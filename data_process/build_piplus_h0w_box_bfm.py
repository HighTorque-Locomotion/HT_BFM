"""Convert an OmniRetarget MuJoCo-qpos climb clip into an H0W MotionLib pkl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.convert_gmr_lafan import convert_motion, load_dof_metadata, resolve_robot_model_path


QPOS_JOINT_NAMES = [
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint",
    "head_yaw_joint", "head_pitch_joint", "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint",
    "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint", "l_hip_pitch_joint", "l_hip_roll_joint",
    "l_thigh_joint", "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "dataset/limb_20_z_scale_1.0_piplus_mapping/climb_20_z_scale_1.0_piplus_mapping.npz")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "dataset/limb_20_z_scale_1.0_piplus_mapping/piplus_h0w_box_bfm.pkl")
    parser.add_argument("--robot-xml", default="package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_motion(args: argparse.Namespace) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    with np.load(args.input, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if qpos.ndim != 2 or qpos.shape[1] != 29:
        raise ValueError(f"Expected qpos [T,29], got {qpos.shape}")
    if not np.isfinite(qpos).all() or fps <= 0.0:
        raise ValueError("qpos/fps must be finite and fps must be positive")
    robot_xml = resolve_robot_model_path(args.robot_xml)
    joint_names, dof_axes, body_names, body_to_joint = load_dof_metadata(robot_xml)
    if joint_names != QPOS_JOINT_NAMES:
        raise ValueError(f"Robot XML joint order differs from RESULTS.md qpos order: xml={joint_names}, qpos={QPOS_JOINT_NAMES}")
    raw = {
        "base_pos_w": qpos[:, :3],
        "base_quat_w": qpos[:, 3:7],  # MuJoCo qpos uses wxyz here.
        "joint_pos": qpos[:, 7:],
        "joint_names": QPOS_JOINT_NAMES,
        "framerate": fps,
    }
    motion = convert_motion(raw, joint_names, dof_axes, body_names, body_to_joint, args.input, quat_order="wxyz", robot="piplus_h0w")
    output = {"box__climb_20_z_scale_1.0_piplus_mapping": motion}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output)
    print(f"Saved {len(output)} box motion with {qpos.shape[0]} frames at {fps:g} FPS to {args.output}")
    return output


if __name__ == "__main__":
    build_motion(parse_args())
