import hashlib
import json
import os
import subprocess
import sys

import pytest

from humanoidverse.flat_amp import PROJECT_ROOT, VENDOR_ROOT
from humanoidverse.flat_amp.__main__ import build_invocation, build_parser


def test_import_is_lazy_and_backend_selection_is_local(tmp_path):
    # Even with a different RSL-RL on PYTHONPATH, a fresh CLI process must resolve
    # the core shipped with this repository. Package imports stay lazy.
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
    assert manifest["files"]
    for item in manifest["files"]:
        path = VENDOR_ROOT / item["path"]
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item.get("local_sha256", item["sha256"]), item["path"]






def test_task_listing_needs_no_isaac_runtime():
    code = """
import sys
from humanoidverse.flat_amp.__main__ import main
main(['list-tasks'])
assert 'isaaclab' not in sys.modules and 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [
        "walk_pi_waq_12: amp",
    ]


@pytest.mark.parametrize("arguments", [
    ["train", "--task", "walk_pi_waq_22"],
    ["train", "--task", "walk_pi_waq_run_22"],
    ["play", "--task", "walk_pi_waq_23", "--checkpoint", "model.pt"],
    ["train", "--controller", "bfm"],
    ["train", "--controller", "bfm-phase"],
    ["bundle-bfm"],
])
def test_removed_tasks_and_controllers_are_rejected(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_pi12_training_resume_and_export_dispatch(tmp_path):
    args = build_parser().parse_args([
        "train", "--resume", str(tmp_path / "model.pt"), "--max-iterations", "2", "--amp-preload", "128",
    ])
    module, options = build_invocation(args)
    assert module == "legged_lab.scripts.train"
    assert options[options.index("--task") + 1] == "walk_pi_waq_12"
    assert options[options.index("--resume") + 1] == str(tmp_path / "model.pt")
    assert options[options.index("--max_iterations") + 1] == "2"
    assert options[options.index("--amp-preload") + 1] == "128"
    args = build_parser().parse_args([
        "export", "--checkpoint", str(tmp_path / "model.pt"), "--output", str(tmp_path / "export"),
    ])
    assert build_invocation(args) == ("legged_lab.scripts.export_policy", [
        "--checkpoint", str(tmp_path / "model.pt"), "--output", str(tmp_path / "export"),
    ])
