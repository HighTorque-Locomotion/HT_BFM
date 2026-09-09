"""Add PiPlus backward motions to an HT_Lab motion archive.

The source BFM motion pickle stores root quaternions in xyzw order, while the
HT_Lab retargeted NPZ reader expects Isaac Lab's wxyz convention.  The robot
joint order is preserved and validated against an existing archive sample.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import yaml

DEFAULT_BACKWARD_MOTIONS = (
    "B4_-_Stand_to_Walk_backwards_stageii_piplus_s_LSE",
    "B5_-__Walk_backwards_stageii_piplus_s_LSE",
)
DEFAULT_CONFIG = "piplus_LSE_walk_run/PiPlus_S_12L8A0G2H1W_LSE_lafan_walk_run.yaml"
OUTPUT_CONFIG = "piplus_LSE_walk_run/PiPlus_S_12L8A0G2H1W_LSE_lafan_walk_run_backward_balanced.yaml"
OUTPUT_MOTION_NAMES = (
    "PiPlus_S_12L8A0G2H1W_LSE_B4_Stand_to_Walk_backwards_retargeted.npz",
    "PiPlus_S_12L8A0G2H1W_LSE_B5_Walk_backwards_retargeted.npz",
)
CATEGORIES = ("forward", "stand", "lateral", "turn", "backward")


def _motion_category(path: str) -> str:
    name = Path(path).name.lower()
    if "backward" in name or "backwards" in name or "reverse" in name:
        return "backward"
    if "stand" in name:
        return "stand"
    if "side" in name or "lateral" in name:
        return "lateral"
    if "turn" in name or "yaw" in name or "rotate" in name:
        return "turn"
    return "forward"


def _npz_bytes(motion: dict, expected_joint_names: tuple[str, ...]) -> bytes:
    joint_names = tuple(str(name) for name in motion["joint_names"])
    if joint_names != expected_joint_names:
        raise ValueError("Backward motion joint order does not match the HT_Lab archive")

    joint_pos = np.asarray(motion["dof"], dtype=np.float64)
    base_pos_w = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    frame_count = joint_pos.shape[0]
    if joint_pos.shape != (frame_count, len(expected_joint_names)):
        raise ValueError(f"Invalid joint_pos shape: {joint_pos.shape}")
    if base_pos_w.shape != (frame_count, 3) or root_quat_xyzw.shape != (frame_count, 4):
        raise ValueError("Backward motion root state has inconsistent dimensions")
    if not all(np.isfinite(value).all() for value in (joint_pos, base_pos_w, root_quat_xyzw)):
        raise ValueError("Backward motion contains non-finite values")

    quat_norm = np.linalg.norm(root_quat_xyzw, axis=-1)
    if not np.allclose(quat_norm, 1.0, atol=1.0e-4):
        raise ValueError("Backward motion contains non-unit root quaternions")
    base_quat_w = root_quat_xyzw[:, [3, 0, 1, 2]]

    output = io.BytesIO()
    np.savez_compressed(
        output,
        joint_pos=joint_pos,
        joint_names=np.asarray(expected_joint_names),
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
        framerate=np.asarray(float(np.asarray(motion["fps"]).reshape(-1)[0])),
    )
    return output.getvalue()


def _balanced_motion_weights(selected_files: list[str]) -> tuple[list[float], dict[str, float]]:
    counts = {category: 0 for category in CATEGORIES}
    for path in selected_files:
        counts[_motion_category(path)] += 1
    missing = [category for category, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Cannot balance categories without samples: {missing}")
    weights = [1.0 / counts[_motion_category(path)] for path in selected_files]
    totals = {
        category: sum(weight for path, weight in zip(selected_files, weights, strict=True) if _motion_category(path) == category)
        for category in CATEGORIES
    }
    return weights, totals


def build_archive(source_zip: Path, source_pkl: Path, output_zip: Path, *, overwrite: bool = False) -> dict:
    if output_zip.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_zip}")
    motions = joblib.load(source_pkl)
    if not isinstance(motions, dict):
        raise TypeError(f"Expected a motion mapping, got {type(motions)!r}")

    with ZipFile(source_zip) as source:
        config = yaml.safe_load(source.read(DEFAULT_CONFIG))
        selected_files = list(config["selected_files"])
        sample_path = selected_files[0]
        with np.load(io.BytesIO(source.read(sample_path)), allow_pickle=False) as sample:
            expected_joint_names = tuple(str(name) for name in sample["joint_names"].tolist())

        added_files = []
        added_payloads = []
        for source_name, output_name in zip(DEFAULT_BACKWARD_MOTIONS, OUTPUT_MOTION_NAMES, strict=True):
            if source_name not in motions:
                raise KeyError(f"Missing backward motion: {source_name}")
            archive_path = f"piplus_LSE_walk_run/npz/{output_name}"
            added_files.append(archive_path)
            added_payloads.append((archive_path, _npz_bytes(motions[source_name], expected_joint_names)))

        balanced_selected_files = selected_files + added_files
        motion_weights, category_weight_totals = _balanced_motion_weights(balanced_selected_files)
        balanced_config = {
            "selected_files": balanced_selected_files,
            "motion_weights": motion_weights,
        }
        manifest = {
            "source_zip": str(source_zip.resolve()),
            "source_zip_sha256": hashlib.sha256(source_zip.read_bytes()).hexdigest(),
            "source_motion_pkl": str(source_pkl.resolve()),
            "source_motion_pkl_sha256": hashlib.sha256(source_pkl.read_bytes()).hexdigest(),
            "source_config": DEFAULT_CONFIG,
            "output_config": OUTPUT_CONFIG,
            "added_source_motions": list(DEFAULT_BACKWARD_MOTIONS),
            "added_archive_files": added_files,
            "joint_count": len(expected_joint_names),
            "joint_names": list(expected_joint_names),
            "source_quaternion_order": "xyzw",
            "output_quaternion_order": "wxyz",
            "motion_count": len(balanced_selected_files),
            "category_motion_counts": {
                category: sum(_motion_category(path) == category for path in balanced_selected_files)
                for category in CATEGORIES
            },
            "category_weight_totals": category_weight_totals,
            "normalized_category_sampling_share": {
                category: total / sum(category_weight_totals.values())
                for category, total in category_weight_totals.items()
            },
        }

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as output:
            for info in source.infolist():
                output.writestr(info, source.read(info.filename))
            for archive_path, payload in added_payloads:
                output.writestr(archive_path, payload)
            output.writestr(OUTPUT_CONFIG, yaml.safe_dump(balanced_config, sort_keys=False))
            output.writestr(
                "piplus_LSE_walk_run/backward_balanced_manifest.json",
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-pkl", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_archive(args.source_zip, args.source_pkl, args.output_zip, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
