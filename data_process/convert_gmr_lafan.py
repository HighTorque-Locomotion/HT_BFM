from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HT_URDF_REPO = REPO_ROOT.parent / "ht_urdf"
if HT_URDF_REPO.exists() and str(HT_URDF_REPO) not in sys.path:
    sys.path.insert(0, str(HT_URDF_REPO))

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from humanoidverse.utils.asset_paths import resolve_asset_path


DEFAULT_INPUT_DIR = Path("data_process/dataset/g1_lafan_dataset")
DEFAULT_OUTPUT_DIR = Path("humanoidverse/data")
DEFAULT_ROBOT_XML = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
PIPLUS_LSE_INPUT_DIR = Path("data_process/dataset/pi_LSE_lafan_dataset_20260617")
PIPLUS_LSE_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/PiPlus_S_12L8A0G2H1W_LSE_260611.xml"


def install_numpy_pickle_compat() -> None:
    try:
        importlib.import_module("numpy._core.multiarray")
        return
    except ModuleNotFoundError:
        pass

    import numpy.core as np_core

    sys.modules.setdefault("numpy._core", np_core)
    for module_name in ("multiarray", "_multiarray_umath", "numeric", "fromnumeric"):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)


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
    parser.add_argument(
        "--robot",
        choices=("g1", "piplus_lse"),
        default="g1",
        help="Use piplus_lse to convert ~/HT_BFM/data_process/dataset/pi_LSE_dataset with the PiPlus LSE XML.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default=None,
        help="Quaternion component order in the source files. Defaults to wxyz for piplus_lse, xyzw otherwise.",
    )
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.robot == "piplus_lse":
        if args.input_dir == DEFAULT_INPUT_DIR:
            args.input_dir = PIPLUS_LSE_INPUT_DIR
        if args.robot_xml == DEFAULT_ROBOT_XML:
            args.robot_xml = resolve_asset_path("", PIPLUS_LSE_ROBOT_XML)
        if args.name == "gmr_lafan":
            args.name = "piplus_lse_lafan"
    if args.quat_order is None:
        args.quat_order = "wxyz" if args.robot == "piplus_lse" else "xyzw"
    return args


def load_dof_metadata(robot_xml: Path) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    root = ET.parse(robot_xml).getroot()
    motor_joints = [
        motor.attrib.get("joint", motor.attrib.get("name"))
        for actuator in root.iter("actuator")
        for motor in actuator
    ]
    motor_joints = [name for name in motor_joints if name is not None]

    joint_axes = {}
    joint_order = []
    for joint in root.iter("joint"):
        if joint.attrib.get("type") == "free":
            continue
        axis = joint.attrib.get("axis")
        if axis is None:
            continue
        name = joint.attrib.get("name")
        joint_axes[name] = [float(value) for value in axis.split()]
        joint_order.append(name)

    joint_names = motor_joints or joint_order
    axes = []
    for name in joint_names:
        if name not in joint_axes:
            raise ValueError(f"Actuated joint {name} has no one-DOF axis in {robot_xml}.")
        axes.append(joint_axes[name])

    if not axes:
        raise ValueError(f"No one-DOF actuated joints found in {robot_xml}.")

    body_names = []
    body_to_joint = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{robot_xml}: missing MJCF worldbody.")

    def add_body(body: ET.Element) -> None:
        body_name = body.attrib.get("name")
        if body_name is None:
            return
        body_names.append(body_name)
        for joint in body.findall("joint"):
            joint_name = joint.attrib.get("name")
            if joint_name in joint_names:
                body_to_joint[body_name] = joint_name
        for child in body.findall("body"):
            add_body(child)

    for body in worldbody.findall("body"):
        add_body(body)

    return joint_names, np.asarray(axes, dtype=np.float32), body_names, body_to_joint


def make_body_aligned_pose_aa(
    root_axis_angle: np.ndarray,
    dof: np.ndarray,
    joint_names: list[str],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
) -> np.ndarray:
    joint_pose = {
        joint_name: dof_axes[joint_id][None, :] * dof[:, joint_id, None]
        for joint_id, joint_name in enumerate(joint_names)
    }
    pose_aa = np.zeros((dof.shape[0], len(body_names), 3), dtype=np.float32)
    pose_aa[:, 0, :] = root_axis_angle
    for body_id, body_name in enumerate(body_names[1:], start=1):
        joint_name = body_to_joint.get(body_name)
        if joint_name is not None:
            pose_aa[:, body_id, :] = joint_pose[joint_name]
    return pose_aa


def normalize_quat_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat_order == "wxyz":
        quat = quat[..., [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quat_order}.")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("root_rot contains zero-length quaternion(s).")
    return quat / norm


def get_raw_field(raw: dict, *names: str, source: Path):
    for name in names:
        if name in raw:
            return raw[name]
    raise KeyError(f"{source} is missing required field. Tried aliases: {names}")


def convert_motion(
    raw: dict,
    joint_names: list[str],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
    source: Path,
    quat_order: str,
) -> dict:
    required = (
        ("fps", "framerate"),
        ("root_pos", "base_pos_w"),
        ("root_rot", "base_quat_w"),
        ("dof_pos", "joint_pos"),
    )
    missing = [aliases for aliases in required if not any(alias in raw for alias in aliases)]
    if missing:
        raise KeyError(f"{source} is missing required field alias group(s): {missing}")

    root_pos = np.asarray(get_raw_field(raw, "root_pos", "base_pos_w", source=source), dtype=np.float32)
    root_rot = normalize_quat_xyzw(get_raw_field(raw, "root_rot", "base_quat_w", source=source), quat_order)
    dof = np.asarray(get_raw_field(raw, "dof_pos", "joint_pos", source=source), dtype=np.float32)
    fps = int(get_raw_field(raw, "fps", "framerate", source=source))

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{source}: root_pos must have shape (T, 3), got {root_pos.shape}.")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{source}: root_rot must have shape (T, 4), got {root_rot.shape}.")
    if dof.ndim != 2 or dof.shape[1] != len(joint_names):
        raise ValueError(f"{source}: dof_pos must have shape (T, {len(joint_names)}), got {dof.shape}.")
    if "joint_names" in raw and list(raw["joint_names"]) != joint_names:
        raise ValueError(
            f"{source}: joint_names do not match robot XML motor order.\n"
            f"raw: {raw['joint_names']}\nxml: {joint_names}"
        )
    if not (root_pos.shape[0] == root_rot.shape[0] == dof.shape[0]):
        raise ValueError(
            f"{source}: root_pos, root_rot, and dof_pos frame counts do not match: "
            f"{root_pos.shape[0]}, {root_rot.shape[0]}, {dof.shape[0]}."
        )
    if fps <= 0:
        raise ValueError(f"{source}: fps must be positive, got {fps}.")

    root_axis_angle = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa = make_body_aligned_pose_aa(root_axis_angle, dof, joint_names, dof_axes, body_names, body_to_joint)

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot,
        "smpl_joints": np.zeros((root_pos.shape[0], 24, 3), dtype=np.float32),
        "fps": fps,
        "joint_names": joint_names,
    }


def load_dataset(
    input_dir: Path,
    joint_names: list[str],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
    quat_order: str,
) -> dict[str, dict]:
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}.")

    dataset = {}
    install_numpy_pickle_compat()
    for path in files:
        dataset[path.stem] = convert_motion(
            joblib.load(path),
            joint_names,
            dof_axes,
            body_names,
            body_to_joint,
            path,
            quat_order,
        )
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

    joint_names, dof_axes, body_names, body_to_joint = load_dof_metadata(args.robot_xml)
    dataset = load_dataset(args.input_dir, joint_names, dof_axes, body_names, body_to_joint, args.quat_order)
    clips = make_clips(dataset, args.clip_seconds)

    dump_dataset(dataset, output_full, args.overwrite)
    dump_dataset(clips, output_clips, args.overwrite)

    total_frames = sum(motion["dof"].shape[0] for motion in dataset.values())
    clip_frames = sum(motion["dof"].shape[0] for motion in clips.values())
    print(f"Saved {len(dataset)} motions / {total_frames} frames to {output_full}")
    print(f"Saved {len(clips)} clips / {clip_frames} frames to {output_clips}")


if __name__ == "__main__":
    main()
