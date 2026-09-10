"""Convert the HT_Lab 22-DoF PiPlus NPZ motion set to MotionLib pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.convert_gmr_lafan import convert_motion, load_dof_metadata, resolve_robot_model_path

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset/piplus_nowaist_walk_run"
DEFAULT_MANIFEST = "PiPlus_S_12L8A0G2H0W_mocap_walk_run.yaml"
DEFAULT_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"
DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "piplus_nowaist_walk_run.pkl"


def _motion_category(source_path: Path) -> str:
    name = source_path.stem.lower()
    if "backward" in name:
        return "backward"
    if source_path.parent.name == "turn" or "zhuan" in name:
        return "turn"
    if any(token in name for token in ("youyi", "zuoyi", "paoxiangyou", "paoxiangzuo")):
        return "lateral"
    return "forward"


def _resolve_selected_file(dataset_dir: Path, selected_file: str) -> Path:
    path = Path(selected_file)
    if path.parts and path.parts[0] == dataset_dir.name:
        path = Path(*path.parts[1:])
    resolved = (dataset_dir / path).resolve()
    if dataset_dir.resolve() not in resolved.parents:
        raise ValueError(f"Selected motion escapes dataset directory: {selected_file!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Selected motion does not exist: {resolved}")
    return resolved


def convert_dataset(
    dataset_dir: Path,
    manifest_path: Path,
    output_path: Path,
    robot_xml: Path | str,
    *,
    overwrite: bool = False,
) -> dict:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    manifest = yaml.safe_load(manifest_path.read_text())
    selected_files = [str(value) for value in manifest.get("selected_files", [])]
    if not selected_files:
        raise ValueError(f"Manifest has no selected_files: {manifest_path}")

    robot_xml = resolve_robot_model_path(robot_xml)
    joint_names, dof_axes, body_names, body_to_joint = load_dof_metadata(robot_xml)
    expected_joint_names = tuple(joint_names)
    dataset = {}
    source_records = []
    for selected_file in selected_files:
        source_path = _resolve_selected_file(dataset_dir, selected_file)
        with np.load(source_path, allow_pickle=False) as raw_data:
            required = {"joint_pos", "joint_names", "base_pos_w", "base_quat_w", "framerate"}
            missing = sorted(required.difference(raw_data.files))
            if missing:
                raise KeyError(f"{source_path} is missing fields: {missing}")
            source_joint_names = tuple(str(value) for value in raw_data["joint_names"].tolist())
            if len(source_joint_names) != len(set(source_joint_names)):
                raise ValueError(f"{source_path} contains duplicate joint names")
            missing_joints = [name for name in expected_joint_names if name not in source_joint_names]
            extra_joints = [name for name in source_joint_names if name not in expected_joint_names]
            if missing_joints or extra_joints:
                raise ValueError(
                    f"{source_path} joint set mismatch: missing={missing_joints}, extra={extra_joints}"
                )
            source_index = {name: index for index, name in enumerate(source_joint_names)}
            joint_pos = np.asarray(raw_data["joint_pos"], dtype=np.float32)[:, [source_index[name] for name in expected_joint_names]]
            base_pos = np.asarray(raw_data["base_pos_w"], dtype=np.float32)
            base_quat = np.asarray(raw_data["base_quat_w"], dtype=np.float32)
            framerate = float(np.asarray(raw_data["framerate"]).reshape(-1)[0])
        frame_count = joint_pos.shape[0]
        if joint_pos.shape != (frame_count, len(expected_joint_names)):
            raise ValueError(f"{source_path}: expected joint_pos [T,{len(expected_joint_names)}], got {joint_pos.shape}")
        if base_pos.shape != (frame_count, 3) or base_quat.shape != (frame_count, 4):
            raise ValueError(f"{source_path}: inconsistent root-state dimensions")
        if framerate <= 0.0 or not all(np.isfinite(value).all() for value in (joint_pos, base_pos, base_quat)):
            raise ValueError(f"{source_path}: non-finite motion or invalid framerate")
        quat_norm = np.linalg.norm(base_quat, axis=-1)
        if not np.allclose(quat_norm, 1.0, atol=1.0e-4):
            raise ValueError(f"{source_path}: base_quat_w is not unit-normalized")

        raw = {
            "base_pos_w": base_pos,
            "base_quat_w": base_quat,
            "joint_pos": joint_pos,
            "joint_names": list(expected_joint_names),
            "framerate": framerate,
        }
        # Use a neutral robot label to preserve each NPZ's declared framerate;
        # the generic converter otherwise applies an old filename-based 50 Hz override.
        motion = convert_motion(
            raw,
            list(expected_joint_names),
            dof_axes,
            body_names,
            body_to_joint,
            source_path,
            quat_order="wxyz",
            robot="piplus_h0w_nowaist",
        )
        category = _motion_category(source_path)
        key = f"{category}__{source_path.stem}"
        if key in dataset:
            raise ValueError(f"Duplicate motion key after conversion: {key}")
        dataset[key] = motion
        source_records.append(
            {
                "source": str(source_path),
                "key": key,
                "category": category,
                "frames": frame_count,
                "framerate": framerate,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, output_path)
    total_frames = sum(len(motion["dof"]) for motion in dataset.values())
    category_frames = {
        category: sum(record["frames"] for record in source_records if record["category"] == category)
        for category in ("forward", "backward", "lateral", "turn")
    }
    category_windows = {
        category: sum(max(0, record["frames"] - 7) for record in source_records if record["category"] == category)
        for category in category_frames
    }
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "robot_xml": str(robot_xml.resolve()),
        "quaternion_input": "wxyz (base_quat_w)",
        "quaternion_output": "xyzw (root_rot)",
        "joint_count": len(expected_joint_names),
        "joint_names": list(expected_joint_names),
        "motion_count": len(dataset),
        "total_frames": total_frames,
        "category_frames": category_frames,
        "category_history8_windows": category_windows,
        "output": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "motions": source_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-xml", default=DEFAULT_ROBOT_XML)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.dataset_dir / DEFAULT_MANIFEST
    return args


def main() -> None:
    args = parse_args()
    result = convert_dataset(args.dataset_dir, args.manifest, args.output, args.robot_xml, overwrite=args.overwrite)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
