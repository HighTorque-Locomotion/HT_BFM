from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rotation


DEFAULT_INPUT_DIR = Path("data_process/lafan_dataset")
DEFAULT_OUTPUT_DIR = Path("humanoidverse/data")
DEFAULT_ROBOT_XML = Path("humanoidverse/data/robots/g1/g1_29dof.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GMR-retargeted LaFan pkl files into the HumanoidVerse "
            "lafan_29dof.pkl and lafan_29dof_10s-clipped.pkl formats."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--name", default="gmr_lafan")
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dof_axes(robot_xml: Path) -> np.ndarray:
    root = ET.parse(robot_xml).getroot()
    axes = []

    for joint in root.iter("joint"):
        if joint.attrib.get("type") == "free":
            continue
        axis = joint.attrib.get("axis")
        if axis is None:
            raise ValueError(f"Joint {joint.attrib.get('name', '<unnamed>')} has no axis in {robot_xml}.")
        axes.append([float(value) for value in axis.split()])

    axes = np.asarray(axes, dtype=np.float32)
    if axes.shape != (29, 3):
        raise ValueError(f"Expected 29 one-DOF joint axes from {robot_xml}, got shape {axes.shape}.")
    return axes


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("root_rot contains zero-length quaternion(s).")
    return quat / norm


def convert_motion(raw: dict, dof_axes: np.ndarray, source: Path) -> dict:
    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{source} is missing required field(s): {missing}")

    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    root_rot = normalize_quat_xyzw(raw["root_rot"])
    dof = np.asarray(raw["dof_pos"], dtype=np.float32)
    fps = int(raw["fps"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{source}: root_pos must have shape (T, 3), got {root_pos.shape}.")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{source}: root_rot must have shape (T, 4), got {root_rot.shape}.")
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"{source}: dof_pos must have shape (T, 29), got {dof.shape}.")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof.shape[0]):
        raise ValueError(
            f"{source}: root_pos, root_rot, and dof_pos frame counts do not match: "
            f"{root_pos.shape[0]}, {root_rot.shape[0]}, {dof.shape[0]}."
        )
    if fps <= 0:
        raise ValueError(f"{source}: fps must be positive, got {fps}.")

    root_axis_angle = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa = np.concatenate(
        [root_axis_angle[:, None, :], dof_axes[None, :, :] * dof[:, :, None]],
        axis=1,
    ).astype(np.float32)

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot,
        "smpl_joints": np.zeros((root_pos.shape[0], 24, 3), dtype=np.float32),
        "fps": fps,
    }


def load_dataset(input_dir: Path, dof_axes: np.ndarray) -> dict[str, dict]:
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}.")

    dataset = {}
    for path in files:
        dataset[path.stem] = convert_motion(joblib.load(path), dof_axes, path)
    return dataset


def make_clips(dataset: dict[str, dict], clip_seconds: float) -> dict[str, dict]:
    if clip_seconds <= 0.0:
        raise ValueError(f"clip-seconds must be positive, got {clip_seconds}.")

    clips = {}
    for motion_name, motion in dataset.items():
        fps = int(motion["fps"])
        clip_frames = int(round(clip_seconds * fps))
        if clip_frames <= 0:
            raise ValueError(f"{motion_name}: clip length rounded to {clip_frames} frames.")

        num_clips = motion["dof"].shape[0] // clip_frames
        for clip_id in range(num_clips):
            start = clip_id * clip_frames
            end = start + clip_frames
            clip = {
                key: value[start:end].copy() if isinstance(value, np.ndarray) else value
                for key, value in motion.items()
            }
            clip["motion_name"] = motion_name
            clips[f"{motion_name}_clip{clip_id}"] = clip
    return clips


def dump_dataset(data: dict[str, dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def main() -> None:
    args = parse_args()
    output_full = args.output_dir / f"{args.name}.pkl"
    output_clips = args.output_dir / f"{args.name}_10s-clipped.pkl"

    dof_axes = load_dof_axes(args.robot_xml)
    dataset = load_dataset(args.input_dir, dof_axes)
    clips = make_clips(dataset, args.clip_seconds)

    dump_dataset(dataset, output_full, args.overwrite)
    dump_dataset(clips, output_clips, args.overwrite)

    total_frames = sum(motion["dof"].shape[0] for motion in dataset.values())
    clip_frames = sum(motion["dof"].shape[0] for motion in clips.values())
    print(f"Saved {len(dataset)} motions / {total_frames} frames to {output_full}")
    print(f"Saved {len(clips)} clips / {clip_frames} frames to {output_clips}")


if __name__ == "__main__":
    main()
