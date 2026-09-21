"""Copy trusted, named Pi expert NPZs and add sagittal-plane mirrored clips.

Originals are byte-for-byte copies. Mirrors contain only the five raw pose
fields used by AMPLoaderPi12; cached kinematics from other robots are omitted.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import VENDOR_ROOT, activate

RAW_FIELDS = ("joint_names", "joint_pos", "base_pos_w", "base_quat_w", "framerate")
DEFAULT_OUTPUT = VENDOR_ROOT / "legged_lab/envs/pi/datasets/motion_amp_expert/pi_12_amp"
SELECTION_NAME = "Pi_S_12L0A0G0H0W_LSE_lafan_walk.yaml"


def joint_mirror_map(names):
    """Pi joints: swap sides, negate X/Z-axis rotations, retain Y rotations."""
    names = [str(name) for name in names]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate joint names")
    positive = {"hip_pitch", "calf", "ankle_pitch", "shoulder_pitch", "elbow", "head_pitch"}
    negative = {"hip_roll", "thigh", "ankle_roll", "shoulder_roll", "upper_arm", "head_yaw", "waist_yaw"}
    indices, signs = [], []
    for name in names:
        paired = name.startswith(("l_", "r_"))
        partner = ({"l": "r", "r": "l"}[name[0]] + name[1:]) if paired else name
        kind = (name[2:] if paired else name).removesuffix("_joint")
        if kind not in positive | negative:
            raise ValueError(f"Unknown Pi joint axis: {name}")
        if partner not in names:
            raise ValueError(f"Missing mirror joint: {partner}")
        indices.append(names.index(partner))
        signs.append(-1 if kind in negative else 1)
    return indices, np.asarray(signs)


def mirror_motion(data):
    """Reflect across world Y=0; quaternions are WXYZ, not XYZW."""
    missing = set(RAW_FIELDS) - set(data)
    if missing:
        raise ValueError(f"Missing raw motion fields: {sorted(missing)}")
    result = {key: np.array(data[key], copy=True) for key in RAW_FIELDS}
    names = result["joint_names"]
    if names.ndim != 1:
        raise ValueError("joint_names must be a vector")
    indices, signs = joint_mirror_map(names.tolist())
    joints, pos, quat = (result[key] for key in ("joint_pos", "base_pos_w", "base_quat_w"))
    if joints.ndim != 2 or joints.shape[1] != len(names) or len(joints) < 2:
        raise ValueError("Expected at least two frames of named joint_pos")
    if pos.shape != (len(joints), 3) or quat.shape != (len(joints), 4):
        raise ValueError("Root poses must align with joint_pos; base_quat_w must be WXYZ")
    fps = float(result["framerate"].item())
    if fps <= 0 or not np.isfinite(fps) or not all(np.isfinite(x).all() for x in (joints, pos, quat)):
        raise ValueError("Nonfinite poses or invalid framerate")
    if np.any(np.linalg.norm(quat, axis=1) < 1e-6):
        raise ValueError("Zero quaternion")
    result["joint_pos"] = joints[:, indices] * signs.astype(joints.dtype)
    pos[:, 1] *= -1
    # R' = S R S, S = diag(1, -1, 1). WXYZ: negate quaternion X/Z.
    quat[:, [1, 3]] *= -1
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_dataset(source_yaml, output):
    activate()
    from rsl_rl.utils.motion_loader_forpi_12dof import selected_motion_files

    source_yaml, output = Path(source_yaml).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    files = [Path(path) for path in selected_motion_files(source_yaml)]
    if len(files) != len(set(files)):
        raise ValueError("Selection contains duplicate source files")
    # Preserve source-relative subdirectories and validate everything before writing.
    source_root = source_yaml.parent
    if not all(path.is_relative_to(source_root) for path in files):
        source_root = source_root.parent
    if not all(path.is_relative_to(source_root) for path in files):
        raise ValueError("Sources must be under the selection directory or its parent")
    prepared = []
    for path in files:
        # Only use this utility on trusted local experts (some names are pickled).
        with np.load(path, allow_pickle=True) as data:
            mirrored = mirror_motion(data)
            omitted = sorted(set(data.files) - set(RAW_FIELDS))
        prepared.append((path, mirrored, omitted, sha256(path)))
    output.mkdir(parents=True)
    selected, records = [], []
    for path, mirrored, omitted, source_hash in prepared:
        relative = path.relative_to(source_root)
        original = Path("original") / relative
        mirror = Path("mirrored") / relative.with_name(relative.stem + "_mirrored.npz")
        (output / original).parent.mkdir(parents=True, exist_ok=True)
        (output / mirror).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output / original)
        np.savez_compressed(output / mirror, **mirrored)
        if sha256(output / original) != source_hash or sha256(path) != source_hash:
            raise RuntimeError(f"Source changed during copy: {path}")
        selected.extend([original.as_posix(), mirror.as_posix()])
        records.append({
            "source": relative.as_posix(), "original": original.as_posix(), "mirrored": mirror.as_posix(),
            "original_sha256": source_hash, "mirrored_sha256": sha256(output / mirror),
            "frames": len(mirrored["joint_pos"]), "framerate": float(mirrored["framerate"].item()),
            "omitted_cached_fields_in_mirror": omitted,
        })
    selection = output / SELECTION_NAME
    selection.write_text(yaml.safe_dump({"selected_files": selected}, sort_keys=False))
    manifest = {
        "source_selection": source_yaml.name, "source_selection_sha256": sha256(source_yaml),
        "generator": "python -m humanoidverse.flat_amp.augment_pi12_data",
        "reflection": "world Y=0; swap l_/r_; negate X/Z joint rotations; WXYZ quaternion -> (w,-x,y,-z)",
        "sampling": "equal weight per clip, original and mirror paired; preserve framerate and frame count",
        "original_count": len(files), "mirrored_count": len(files), "motions": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(selected)} clips ({len(files)} originals + {len(files)} mirrors): {selection}")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    build_dataset(args.source_yaml, args.output)


if __name__ == "__main__":
    main()
