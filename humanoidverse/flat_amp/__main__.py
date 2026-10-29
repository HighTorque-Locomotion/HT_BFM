"""Train, play and export walk_pi_waq_12 inside HT_BFM."""

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

from . import CONTROLLERS, PROJECT_ROOT, TASKS, VENDOR_ROOT, activate
from .keyboard_control import add_keyboard_arguments


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m humanoidverse.flat_amp",
        description=__doc__,
        epilog="Use the HT_BFM Python 3.11 / Isaac Sim 5.1 environment.",
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("list-tasks", help="List the task without starting Isaac Sim")
    for operation in ("train", "play"):
        command = commands.add_parser(operation, help=f"{operation.capitalize()} the Pi12 AMP policy")
        command.add_argument("--task", choices=TASKS, default=TASKS[0])
        command.add_argument("--controller", choices=CONTROLLERS, default="amp")
        command.add_argument("--bfm-weights", type=Path, help="Native Pi12 BFM run, checkpoint or model directory")
        command.add_argument("--bfm-robot-config", type=Path, help="Override the frozen BFM's config.yaml")
        command.add_argument("--no-phase", action="store_true", help="BFM: omit phase sin/cos from command-to-z")
        command.add_argument("--num-envs", "--num_envs", dest="num_envs", type=positive_int)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--headless", action="store_true")
        command.add_argument("--actuator-profile", choices=("ht",), default="ht")
        if operation == "train":
            command.add_argument(
                "--max-iterations", "--max_iterations", dest="max_iterations", type=positive_int,
                help="Number of updates in this invocation, including when resuming",
            )
            command.add_argument("--seed", type=int)
            command.add_argument("--resume", type=Path)
            command.add_argument("--log-dir", type=Path, help="New run directory; default: logs/flat_amp/<task>/<timestamp>_amp")
            command.add_argument("--amp-preload", type=positive_int, help="Expert preload override for smoke tests")
            command.add_argument("--dry-run", action="store_true", help="BFM: validate weights and interfaces without Isaac Sim")
        else:
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--steps", type=positive_int, default=1000)
            command.add_argument("--record-policy", type=Path, help="BFM: save actual policy inputs/outputs in a new directory")
            command.add_argument(
                "--velocity", nargs=3, type=float, metavar=("VX", "VY", "YAW"),
                help="Fixed command, or initial command with keyboard control",
            )
            add_keyboard_arguments(command)
    export = commands.add_parser("export", help="Export AMP or BFM+AMP deployment models without Isaac Sim")
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", type=Path, help="New export directory; required unless --check-only")
    export.add_argument("--controller", choices=CONTROLLERS, help="Default: detect from checkpoint metadata")
    export.add_argument("--bfm-weights", type=Path, help="BFM: original native weights used for training")
    export.add_argument("--bfm-robot-config", type=Path, help="BFM: override the original config.yaml")
    export.add_argument("--env-config", type=Path, help="BFM: default <checkpoint parent>/params/env.yaml")
    export.add_argument("--urdf", type=Path, help="BFM: override the target URDF location saved in env.yaml")
    export.add_argument("--check-only", action="store_true", help="BFM: validate native models/config without exporting")
    return parser


def resolve_path(path):
    return path.expanduser().resolve()


def build_invocation(args):
    """Map CLI options to the local Pi12 training, playback or export entry point."""
    if args.operation == "export":
        if args.output is None and not args.check_only:
            raise ValueError("--output is required unless --check-only is selected")
        result = ["--checkpoint", str(resolve_path(args.checkpoint))]
        for option in ("output", "bfm_weights", "bfm_robot_config", "env_config", "urdf"):
            value = getattr(args, option)
            if value is not None:
                result += ["--" + option.replace("_", "-"), str(resolve_path(value))]
        if args.controller:
            result += ["--controller", args.controller]
        if args.check_only:
            result.append("--check-only")
        return "legged_lab.scripts.export_policy", result
    if args.actuator_profile != "ht":
        raise ValueError("Pi12 uses the HT_lab 5036 motor profile; --actuator-profile must be ht")
    if args.operation == "play" and args.keyboard_control and args.headless:
        raise ValueError("--keyboard-control requires a GUI; omit --headless")
    if args.controller == "bfm":
        return build_bfm_invocation(args)
    if getattr(args, "record_policy", None) is not None:
        raise ValueError("--record-policy requires --controller bfm")
    if args.bfm_weights or args.bfm_robot_config or args.no_phase or getattr(args, "dry_run", False):
        raise ValueError("BFM options require --controller bfm")
    result = ["--task", args.task, "--device", args.device, "--actuator-profile", args.actuator_profile]
    if args.headless:
        result += ["--headless"]
    if args.num_envs is not None:
        result += ["--num_envs", str(args.num_envs)]
    if args.operation == "train":
        if args.amp_preload is not None:
            result += ["--amp-preload", str(args.amp_preload)]
        for option in ("max_iterations", "seed", "resume"):
            value = getattr(args, option)
            if value is not None:
                result += ["--" + option, str(resolve_path(value) if isinstance(value, Path) else value)]
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        log_dir = args.log_dir or PROJECT_ROOT / "logs/flat_amp" / args.task / f"{stamp}_amp"
        result += ["--log-dir", str(resolve_path(log_dir))]
    else:
        result += ["--checkpoint", str(resolve_path(args.checkpoint)), "--steps", str(args.steps)]
        if args.velocity is not None:
            result += ["--velocity", *(str(value) for value in args.velocity)]
        if args.keyboard_control:
            result += [
                "--keyboard-control", "--keyboard-linvel-step", str(args.keyboard_linvel_step),
                "--keyboard-angvel-step", str(args.keyboard_angvel_step),
            ]
    return "legged_lab.scripts." + args.operation, result


def build_bfm_invocation(args):
    if args.bfm_weights is None:
        raise ValueError("--controller bfm requires --bfm-weights with a pretrained Pi12 BFM")
    result = ["--task", args.task, "--device", args.device, "--bfm-weights", str(resolve_path(args.bfm_weights))]
    if args.bfm_robot_config:
        result += ["--bfm-robot-config", str(resolve_path(args.bfm_robot_config))]
    for flag in ("headless", "no_phase"):
        if getattr(args, flag):
            result.append("--" + flag.replace("_", "-"))
    if args.num_envs is not None:
        result += ["--num_envs", str(args.num_envs)]
    if args.operation == "train":
        for option in ("max_iterations", "seed", "resume", "amp_preload"):
            value = getattr(args, option)
            if value is not None:
                name = "--amp-preload" if option == "amp_preload" else "--" + option
                result += [name, str(resolve_path(value) if isinstance(value, Path) else value)]
        if args.dry_run:
            result.append("--dry-run")
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        log_dir = args.log_dir or PROJECT_ROOT / "logs/flat_amp" / args.task / f"{stamp}_bfm"
        result += ["--log-dir", str(resolve_path(log_dir))]
    else:
        result += ["--play", "--resume", str(resolve_path(args.checkpoint)), "--play-steps", str(args.steps)]
        if args.record_policy is not None:
            result += ["--record-policy", str(resolve_path(args.record_policy))]
        if args.velocity is not None:
            result += ["--velocity", *(str(value) for value in args.velocity)]
        if args.keyboard_control:
            result += [
                "--keyboard-control", "--keyboard-linvel-step", str(args.keyboard_linvel_step),
                "--keyboard-angvel-step", str(args.keyboard_angvel_step),
            ]
    return "legged_lab.scripts.train_bfm_amp", result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.operation == "list-tasks":
        print(f"{TASKS[0]}: {', '.join(CONTROLLERS)}")
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
