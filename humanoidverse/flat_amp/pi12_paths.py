"""Locations of the user-supplied Pi12 URDF and bundled mirrored experts."""

import os
from pathlib import Path

from . import PROJECT_ROOT, VENDOR_ROOT


def robot_urdf():
    default = PROJECT_ROOT.parent / "ht_urdf/ht_urdf/Pi_S_12L0A0G0H0W/urdf/Pi_S_12L0A0G0H0W_260909.urdf"
    return Path(os.environ.get("PI12_URDF_PATH", default)).expanduser().resolve()


def motion_yaml():
    default = VENDOR_ROOT / "legged_lab/envs/pi/datasets/motion_amp_expert/pi_12_amp/Pi_S_12L0A0G0H0W_LSE_lafan_walk.yaml"
    return Path(os.environ.get("PI12_AMP_MOTION_YAML", default)).expanduser().resolve()
