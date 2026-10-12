import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from humanoidverse.flat_amp import PROJECT_ROOT, VENDOR_ROOT
from humanoidverse.flat_amp.__main__ import build_invocation, build_parser


def test_import_is_lazy_and_backend_selection_is_local(tmp_path):
    # Even with a different RSL-RL on PYTHONPATH, a fresh CLI process must resolve
    # the core shipped with this repository. Existing Stage2 imports stay lazy.
    foreign = tmp_path / "rsl_rl"
    foreign.mkdir()
    (foreign / "__init__.py").write_text('raise RuntimeError("foreign RSL-RL imported")\n')
    code = """
import sys
before = sys.path.copy()
from humanoidverse.flat_amp import activate, VENDOR_ROOT
assert sys.path == before
assert 'rsl_rl' not in sys.modules and 'legged_lab' not in sys.modules
activate()
import rsl_rl, legged_lab
from pathlib import Path
assert Path(rsl_rl.__file__).is_relative_to(VENDOR_ROOT)
assert Path(legged_lab.__file__).is_relative_to(VENDOR_ROOT)
"""
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(tmp_path)}, check=True)


def test_activation_rejects_mixed_training_cores():
    code = """
import sys, types
from humanoidverse.flat_amp import activate
foreign = types.ModuleType('rsl_rl')
foreign.__file__ = '/another/project/rsl_rl/__init__.py'
sys.modules['rsl_rl'] = foreign
try:
    activate()
except RuntimeError as error:
    assert 'fresh process' in str(error)
else:
    raise AssertionError('mixed training cores were accepted')
"""
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def test_training_core_and_assets_match_source_manifest():
    manifest = json.loads((VENDOR_ROOT / "source_manifest.json").read_text())
    assert manifest["source_commit"] == "b2d706a8b1323146a0e58595424b7fe5090a8357"
    assert len(manifest["files"]) >= 142
    assert len({item["path"] for item in manifest["files"]}) == len(manifest["files"])
    for item in manifest["files"]:
        path = VENDOR_ROOT / item["path"]
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]


@pytest.fixture
def distill_cli_paths(tmp_path):
    weights = tmp_path / "bfm_actor"
    weights.mkdir()
    (weights / "config.yaml").write_text("robot: {}\n")
    no_phase = tmp_path / "no_phase.pt"
    phase = tmp_path / "phase.pt"
    no_phase.touch()
    phase.touch()
    return ["--bfm-weights", str(weights), "--no-phase-teacher", str(no_phase), "--phase-teacher", str(phase)]


def test_distill_dispatch_uses_historical_defaults_and_own_backend(distill_cli_paths):
    args = build_parser().parse_args(["distill", *distill_cli_paths])
    module, options = build_invocation(args)
    assert module == "legged_lab.scripts.train_bfm_amp"
    expected = {
        "--gait-profile": "backward_fast", "--backward-period": "0.5", "--num_envs": "4096",
        "--seed": "42", "--max_iterations": "50000", "--distill-coef": "5.0",
        "--distill-final-coef": "0.5", "--distill-hold": "1000", "--distill-decay": "10000",
        "--bfm-source": str(PROJECT_ROOT),
    }
    for option, value in expected.items():
        assert options[options.index(option) + 1] == value
    assert "--no-phase" not in options and "--resume" not in options
    assert Path(options[options.index("--log-dir") + 1]).name.endswith("_distill")


def test_distill_dry_run_does_not_allocate_a_log_directory(distill_cli_paths, tmp_path):
    args = build_parser().parse_args(["distill", *distill_cli_paths, "--dry-run", "--log-dir", str(tmp_path / "run")])
    _, options = build_invocation(args)
    assert "--dry-run" in options and "--log-dir" not in options
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("extra", [
    ["--gait-profile", "fixed"], ["--actuator-profile", "legacy"], ["--task", "walk_pi_waq_23"],
    ["--distill-coef", "0"], ["--distill-final-coef", "6"], ["--backward-period", "0"],
])
def test_distill_rejects_incompatible_modes_before_isaac(distill_cli_paths, extra):
    args = build_parser().parse_args(["distill", *distill_cli_paths, *extra])
    with pytest.raises(ValueError):
        build_invocation(args)


def test_distilled_student_play_needs_gait_but_no_teacher(tmp_path):
    weights = tmp_path / "bfm_actor"
    weights.mkdir()
    (weights / "config.yaml").write_text("robot: {}\n")
    args = build_parser().parse_args([
        "play", "--controller", "bfm-phase", "--gait-profile", "backward_fast",
        "--bfm-weights", str(weights), "--checkpoint", str(tmp_path / "student.pt"),
    ])
    _, options = build_invocation(args)
    assert "--play" in options and "--gait-profile" in options
    assert "--distill-no-phase-teacher" not in options and "--distill-phase-teacher" not in options


@pytest.mark.parametrize("operation", ["train", "play"])
def test_ordinary_amp_cannot_silently_select_variable_bfm_gait(operation, tmp_path):
    options = [operation, "--gait-profile", "backward_fast"]
    if operation == "play":
        options += ["--checkpoint", str(tmp_path / "policy.pt")]
    with pytest.raises(ValueError, match="bfm-phase"):
        build_invocation(build_parser().parse_args(options))


@pytest.mark.parametrize("controller,phase", [("bfm", False), ("bfm-phase", True)])
def test_bfm_dispatch_preserves_phase_resume_and_local_source(tmp_path, controller, phase):
    weights = tmp_path / "checkpoint/model"
    weights.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text("robot: {}\n")
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--controller",
            controller,
            "--bfm-weights",
            str(weights),
            "--resume",
            str(tmp_path / "model.pt"),
            "--max_iterations",
            "2",
        ]
    )
    module, options = build_invocation(args)
    assert module == "legged_lab.scripts.train_bfm_amp"
    assert ("--no-phase" not in options) == phase
    assert options[options.index("--bfm-source") + 1] == str(PROJECT_ROOT)
    assert options[options.index("--bfm-robot-config") + 1] == str(config)
    assert options[options.index("--max_iterations") + 1] == "2"
    assert Path(options[options.index("--log-dir") + 1]).is_relative_to(PROJECT_ROOT / "logs/flat_amp")
    play = parser.parse_args(
        ["play", "--controller", controller, "--bfm-weights", str(weights), "--checkpoint", str(tmp_path / "model.pt")]
    )
    _, options = build_invocation(play)
    assert "--play" in options and "--resume" in options
    assert options[options.index("--num_envs") + 1] == "1"


@pytest.mark.parametrize("task", ["walk_pi_waq_run_22", "walk_pi_waq_23"])
def test_bfm_rejects_unsupported_task_before_loading_weights(task):
    args = build_parser().parse_args(["train", "--controller", "bfm", "--task", task])
    with pytest.raises(ValueError, match="only for walk_pi_waq_22"):
        build_invocation(args)


def test_task_listing_needs_no_isaac_runtime():
    code = """
import sys
from humanoidverse.flat_amp.__main__ import main
main(['list-tasks'])
assert 'isaaclab' not in sys.modules and 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [
        "walk_pi_waq_22: amp, bfm, bfm-phase",
        "walk_pi_waq_run_22: amp",
        "walk_pi_waq_23: amp",
    ]
