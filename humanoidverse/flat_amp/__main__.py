"""Run the Pi flat AMP tasks inside HT_BFM."""

import argparse
import importlib
import math
import sys
from datetime import datetime
from pathlib import Path

from . import CONTROLLERS, PROJECT_ROOT, TASKS, VENDOR_ROOT, activate


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m humanoidverse.flat_amp",
        description=__doc__,
        epilog="Use the existing Python 3.11 / Isaac Sim 5.1 environment.",
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("list-tasks", help="List tasks and supported controllers without starting Isaac Sim")
    for operation in ("train", "play", "distill"):
        help_text = "Train a phase-enabled student with frozen no-phase and phase teachers" if operation == "distill" else f"{operation.capitalize()} a flat AMP policy"
        command = commands.add_parser(operation, help=help_text)
        command.add_argument("--task", choices=TASKS, default=TASKS[0])
        if operation == "distill":
            command.set_defaults(controller="bfm-phase")
        else:
            command.add_argument(
                "--controller", choices=CONTROLLERS, default="amp", help="amp: joint policy; bfm: frozen BFM; bfm-phase: frozen BFM with phase"
            )
        command.add_argument("--num-envs", "--num_envs", dest="num_envs", type=positive_int)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--headless", action="store_true")
        command.add_argument("--actuator-profile", choices=("ht", "legacy"), default="ht")
        command.add_argument("--bfm-weights", type=Path, help="Frozen H0W22 actor bundle or original checkpoint directory")
        command.add_argument("--bfm-robot-config", type=Path, help="Defaults to the config.yaml accompanying the BFM weights")
        command.add_argument(
            "--gait-profile", choices=("fixed", "backward_fast"), default="backward_fast" if operation == "distill" else "fixed",
            help="backward_fast: 0.5s backward / 0.8s other movement; requires bfm-phase and HTMotor",
        )
        command.add_argument("--backward-period", type=finite_float, default=0.5)
        if operation in ("train", "distill"):
            command.add_argument(
                "--max-iterations",
                "--max_iterations",
                dest="max_iterations",
                type=positive_int,
                help="Number of updates in this invocation, including when resuming",
            )
            command.add_argument("--seed", type=int)
            command.add_argument("--resume", type=Path)
            command.add_argument("--log-dir", type=Path, help="New run directory; default: logs/flat_amp/<task>/<timestamp>_<controller>")
            if operation == "train":
                command.add_argument("--amp-preload", type=positive_int, help="Ordinary AMP expert preload override for smoke tests")
            else:
                command.set_defaults(amp_preload=None)
            command.add_argument("--dry-run", action="store_true", help="Validate frozen BFM weights without Isaac Sim")
            if operation == "distill":
                command.add_argument("--distill-no-phase-teacher", "--no-phase-teacher", type=Path, required=True,
                                     help="Schema-1 checkpoint for backward/standing and student initialization")
                command.add_argument("--distill-phase-teacher", "--phase-teacher", type=Path, required=True,
                                     help="Fixed 0.8s schema-2 checkpoint for other movement")
                command.add_argument("--distill-coef", type=finite_float, default=5.0)
                command.add_argument("--distill-final-coef", type=finite_float, default=0.5)
                command.add_argument("--distill-hold", type=nonnegative_int, default=1000)
                command.add_argument("--distill-decay", type=positive_int, default=10000)
                command.set_defaults(num_envs=4096, seed=42, max_iterations=50000)
        else:
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--steps", type=positive_int, default=1000)
            command.add_argument(
                "--velocity", nargs=3, type=float, metavar=("VX", "VY", "YAW"), help="Fixed command for ordinary AMP playback"
            )
    export = commands.add_parser("export", help="Export an ordinary AMP checkpoint to TorchScript and ONNX")
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path, help="New export directory")
    bundle = commands.add_parser("bundle-bfm", help="Extract and verify a compact frozen actor from full BFM weights")
    bundle.add_argument("--bfm-weights", required=True, type=Path)
    bundle.add_argument("--bfm-robot-config", type=Path)
    bundle.add_argument("--output", required=True, type=Path, help="New actor bundle directory")
    return parser


def resolve_path(path):
    return path.expanduser().resolve()


def bfm_options(args):
    if args.bfm_weights is None:
        raise ValueError("--bfm-weights is required for a BFM controller")
    weights = resolve_path(args.bfm_weights)
    if not weights.exists():
        raise ValueError(f"BFM weights do not exist: {weights}")
    if args.bfm_robot_config:
        config = resolve_path(args.bfm_robot_config)
    else:
        directory = weights if weights.is_dir() else weights.parent
        candidates = [directory / "config.yaml", directory.parent / "config.yaml", directory.parent.parent / "config.yaml"]
        config = next((path for path in candidates if path.is_file()), None)
        if config is None:
            raise ValueError("BFM config.yaml was not found beside the weights; supply --bfm-robot-config")
    if not config.is_file():
        raise ValueError(f"BFM robot config does not exist: {config}")
    return ["--bfm-source", str(PROJECT_ROOT), "--bfm-weights", str(weights), "--bfm-robot-config", str(config)]


def build_invocation(args):
    """Map CLI options to the selected training, playback, or export entry point."""
    if args.operation == "export":
        return "legged_lab.scripts.export_policy", [
            "--checkpoint",
            str(resolve_path(args.checkpoint)),
            "--output",
            str(resolve_path(args.output)),
        ]
    if args.operation == "bundle-bfm":
        return "legged_lab.scripts.export_bfm_actor", bfm_options(args) + ["--output", str(resolve_path(args.output))]
    frozen = args.controller != "amp"
    if frozen and args.task != "walk_pi_waq_22":
        raise ValueError("Frozen H0W22 BFM is supported only for walk_pi_waq_22")
    if not frozen and (args.bfm_weights is not None or args.bfm_robot_config is not None):
        raise ValueError("BFM options require --controller bfm or bfm-phase")
    if args.backward_period <= 0:
        raise ValueError("--backward-period must be positive")
    variable_gait = args.gait_profile == "backward_fast"
    if variable_gait and (args.controller != "bfm-phase" or args.actuator_profile != "ht"):
        raise ValueError("backward_fast requires --controller bfm-phase and --actuator-profile ht")
    if args.operation == "distill":
        if not variable_gait:
            raise ValueError("Dual-teacher distillation requires --gait-profile backward_fast")
        if args.distill_coef <= 0 or not 0 <= args.distill_final_coef <= args.distill_coef:
            raise ValueError("Distillation coefficients require 0 <= final <= initial and initial > 0")
    result = ["--task", args.task, "--device", args.device, "--actuator-profile", args.actuator_profile]
    if args.headless:
        result += ["--headless"]
    if args.num_envs is not None:
        result += ["--num_envs", str(args.num_envs)]
    if frozen:
        result += bfm_options(args)
        if args.controller == "bfm":
            result += ["--no-phase"]
        if variable_gait:
            result += ["--gait-profile", "backward_fast", "--backward-period", str(args.backward_period)]
    if args.operation in ("train", "distill"):
        if args.dry_run and not frozen:
            raise ValueError("--dry-run validates BFM weights and requires a BFM controller")
        if args.amp_preload is not None:
            if frozen:
                raise ValueError("--amp-preload is supported only for ordinary AMP")
            result += ["--amp-preload", str(args.amp_preload)]
        if args.operation == "distill":
            for option in ("distill_no_phase_teacher", "distill_phase_teacher"):
                path = resolve_path(getattr(args, option))
                if not path.is_file():
                    raise ValueError(f"Teacher checkpoint does not exist: {path}")
                result += ["--" + option.replace("_", "-"), str(path)]
            for option in ("distill_coef", "distill_final_coef", "distill_hold", "distill_decay"):
                result += ["--" + option.replace("_", "-"), str(getattr(args, option))]
        for option in ("max_iterations", "seed", "resume"):
            value = getattr(args, option)
            if value is not None:
                result += ["--" + option, str(resolve_path(value) if isinstance(value, Path) else value)]
        if args.dry_run:
            result += ["--dry-run"]
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            suffix = "distill" if args.operation == "distill" else args.controller
            log_dir = args.log_dir or PROJECT_ROOT / "logs/flat_amp" / args.task / f"{stamp}_{suffix}"
            result += ["--log-dir", str(resolve_path(log_dir))]
        module = "train_bfm_amp" if frozen else "train"
    else:
        if frozen:
            if args.velocity is not None:
                raise ValueError("--velocity is supported only for ordinary AMP playback")
            if args.num_envs is None:
                result += ["--num_envs", "1"]
            result += ["--play", "--resume", str(resolve_path(args.checkpoint)), "--play-steps", str(args.steps)]
            module = "train_bfm_amp"
        else:
            result += ["--checkpoint", str(resolve_path(args.checkpoint)), "--steps", str(args.steps)]
            if args.velocity is not None:
                result += ["--velocity", *(str(value) for value in args.velocity)]
            module = "play"
    return "legged_lab.scripts." + module, result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.operation == "list-tasks":
        for task in TASKS:
            controllers = CONTROLLERS if task == "walk_pi_waq_22" else ("amp",)
            print(f"{task}: {', '.join(controllers)}")
        return
    try:
        module_name, backend_args = build_invocation(args)
    except ValueError as error:
        parser.error(str(error))
    activate()
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(VENDOR_ROOT):
        raise RuntimeError(f"Flat AMP resolved to an external training core: {module.__file__}")
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *backend_args]
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
