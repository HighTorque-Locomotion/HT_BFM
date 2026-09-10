"""Build a non-destructive PiPlus H0W BFM dataset with terrain-prefixed motion keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.convert_gmr_lafan import convert_motion, load_dof_metadata, resolve_robot_model_path  # noqa: I001


DEFAULT_ROOT = Path("dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629")
DEFAULT_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-pkl", type=Path, default=DEFAULT_ROOT / "piplus_h0w_lafan_10s-clipped.pkl")
    parser.add_argument("--stairs-dir", type=Path, default=DEFAULT_ROOT / "stairs_0w")
    parser.add_argument("--terrain-metadata", type=Path, default=DEFAULT_ROOT / "stairs_0w/motion_matched_terrain/metadata.yaml")
    parser.add_argument("--robot-xml", default=DEFAULT_ROBOT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "piplus_h0w_flat_stairs_bfm.pkl")
    parser.add_argument("--output-metadata", type=Path, default=DEFAULT_ROOT / "piplus_h0w_flat_stairs_bfm.yaml")
    parser.add_argument(
        "--stair-repeat",
        type=int,
        default=19,
        help="Repeat each stair expert under unique keys; 19 gives about 60%% stair experts versus 76 flat clips.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.stair_repeat < 1:
        raise ValueError("--stair-repeat must be at least 1.")
    flat = joblib.load(args.flat_pkl)
    if not isinstance(flat, dict) or not flat:
        raise ValueError(f"Expected a non-empty motion dictionary in {args.flat_pkl}.")
    robot_xml = resolve_robot_model_path(args.robot_xml)
    joint_names, dof_axes, body_names, body_to_joint = load_dof_metadata(robot_xml)
    terrain_metadata = yaml.safe_load(args.terrain_metadata.read_text())
    terrain_by_motion = {Path(item["motion_file"]).stem: int(item["terrain_id"]) for item in terrain_metadata["motion_files"]}
    terrain_files = {int(item["terrain_id"]): item["terrain_file"] for item in terrain_metadata["terrains"]}

    output = {f"flat__{key}": value for key, value in flat.items()}
    mappings = []
    for raw_path in sorted(args.stairs_dir.glob("*.pkl")):
        if raw_path.stem not in terrain_by_motion:
            # metadata names include the retargetted suffix while raw PKLs do not.
            candidates = [key for key in terrain_by_motion if key.removesuffix("_retargetted") == raw_path.stem]
            if len(candidates) != 1:
                raise ValueError(f"Cannot resolve one terrain entry for {raw_path}.")
            metadata_key = candidates[0]
        else:
            metadata_key = raw_path.stem
        terrain_id = terrain_by_motion[metadata_key]
        motion = convert_motion(
            joblib.load(raw_path), joint_names, dof_axes, body_names, body_to_joint,
            raw_path, quat_order="wxyz", robot="piplus_h0w",
        )
        for repeat_id in range(args.stair_repeat):
            output_key = f"stairs_{terrain_id}__{raw_path.stem}__repeat{repeat_id:02d}"
            output[output_key] = motion
            mappings.append({
                "motion_key": output_key,
                "terrain_id": terrain_id,
                "terrain_file": str((args.terrain_metadata.parent / terrain_files[terrain_id]).resolve()),
                "source_motion": str(raw_path.resolve()),
            })

    metadata = {
        "schema_version": 1,
        "robot": "PiPlus_S_12L8A0G2H0W",
        "num_dof": len(joint_names),
        "joint_names": joint_names,
        "flat_motion_prefix": "flat__",
        "flat_motion_count": len(flat),
        "stair_motion_count": len(mappings),
        "unique_stair_motion_count": len(mappings) // args.stair_repeat,
        "stair_repeat": args.stair_repeat,
        "stairs": mappings,
        "sources": {
            "flat_pkl": str(args.flat_pkl.resolve()),
            "stairs_dir": str(args.stairs_dir.resolve()),
            "terrain_metadata": str(args.terrain_metadata.resolve()),
            "robot_xml": str(robot_xml),
        },
    }
    return output, metadata


def main() -> None:
    args = parse_args()
    for path in (args.output, args.output_metadata):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace generated output.")
    dataset, metadata = build_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, args.output)
    args.output_metadata.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))
    print(f"Saved {metadata['flat_motion_count']} flat + {metadata['stair_motion_count']} stair motions to {args.output}")


if __name__ == "__main__":
    main()
