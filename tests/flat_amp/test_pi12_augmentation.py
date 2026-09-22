"""Offline expert mirroring is separate from PPO policy symmetry."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from rsl_rl.utils.motion_loader_forpi_12dof import LEG_JOINT_NAMES, AMPLoaderPi12, selected_motion_files
from scipy.spatial.transform import Rotation

from humanoidverse.flat_amp.augment_pi12_data import RAW_FIELDS, build_dataset, joint_mirror_map, mirror_motion, sha256
from humanoidverse.flat_amp.pi12_paths import motion_yaml, robot_urdf


def sample_motion():
    rng = np.random.default_rng(42)
    names = np.array([*reversed(LEG_JOINT_NAMES), "head_yaw_joint", "head_pitch_joint", "waist_yaw_joint"])
    return {
        "joint_names": names,
        "joint_pos": rng.normal(size=(20, len(names))),
        "base_pos_w": rng.normal(size=(20, 3)),
        "base_quat_w": Rotation.random(20, random_state=rng).as_quat()[:, [3, 0, 1, 2]],
        "framerate": np.array(50.0),
        "body_pos_w": np.zeros((20, 27, 3)),
    }


def test_mirror_geometry_involution_and_no_mutation():
    source = sample_motion()
    backup = {key: value.copy() for key, value in source.items()}
    mirrored = mirror_motion(source)
    restored = mirror_motion(mirrored)
    assert set(mirrored) == set(RAW_FIELDS)  # Never propagate stale body/FK caches.
    for key in RAW_FIELDS:
        np.testing.assert_array_equal(restored[key], source[key])
    for key in source:
        np.testing.assert_array_equal(source[key], backup[key])
    reflection = np.diag([1, -1, 1])
    before = Rotation.from_quat(source["base_quat_w"][:, [1, 2, 3, 0]]).as_matrix()
    after = Rotation.from_quat(mirrored["base_quat_w"][:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(after, reflection @ before @ reflection, atol=1e-14)
    np.testing.assert_array_equal(mirrored["base_pos_w"], source["base_pos_w"] @ reflection)
    names = source["joint_names"].tolist()
    # Independent axis checks, not merely an involution or comparison to the generator.
    for joint, sign in (("hip_pitch", 1), ("hip_roll", -1), ("thigh", -1), ("calf", 1), ("ankle_pitch", 1), ("ankle_roll", -1)):
        np.testing.assert_array_equal(
            mirrored["joint_pos"][:, names.index(f"l_{joint}_joint")],
            sign * source["joint_pos"][:, names.index(f"r_{joint}_joint")],
        )
    for joint, sign in (("head_yaw_joint", -1), ("head_pitch_joint", 1), ("waist_yaw_joint", -1)):
        index = names.index(joint)
        np.testing.assert_array_equal(mirrored["joint_pos"][:, index], sign * source["joint_pos"][:, index])


@pytest.mark.parametrize("names", [["l_calf_joint"], ["l_mystery_joint", "r_mystery_joint"], ["head_pitch_joint"] * 2])
def test_reject_incomplete_unknown_or_duplicate_joints(names):
    with pytest.raises(ValueError):
        joint_mirror_map(names)


def test_copy_selection_manifest_and_overwrite_protection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.savez(source / "walk.npz", **sample_motion())
    selection = source / "selection.yaml"
    selection.write_text(yaml.safe_dump({"selected_files": ["walk.npz"]}))
    before = sha256(source / "walk.npz")
    output = tmp_path / "output"
    result = build_dataset(selection, output)
    paths = selected_motion_files(result)
    assert len(paths) == 2
    assert sha256(paths[0]) == before == sha256(source / "walk.npz")
    record = json.loads((output / "manifest.json").read_text())["motions"][0]
    assert record["omitted_cached_fields_in_mirror"] == ["body_pos_w"]
    assert sha256(paths[1]) == record["mirrored_sha256"]
    with pytest.raises(FileExistsError):
        build_dataset(selection, output)
    assert sha256(paths[0]) == before


def test_bundled_dataset_pairs_and_hashes():
    root = motion_yaml().parent
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["original_count"] == manifest["mirrored_count"] == len(manifest["motions"])
    selected = selected_motion_files(motion_yaml())
    assert selected and len(selected) == len(set(selected))
    assert len(selected) % 2 == 0
    expected_paths = []
    for record in manifest["motions"]:
        original, mirrored = [root / record[key] for key in ("original", "mirrored")]
        expected_paths.extend([str(original.resolve()), str(mirrored.resolve())])
        assert sha256(original) == record["original_sha256"]
        assert sha256(mirrored) == record["mirrored_sha256"]
        with np.load(original, allow_pickle=True) as source, np.load(mirrored, allow_pickle=True) as target:
            expected = mirror_motion(source)
            for key in RAW_FIELDS:
                np.testing.assert_array_equal(target[key], expected[key])
                np.testing.assert_array_equal(mirror_motion(target)[key], source[key])
    assert set(selected) <= set(expected_paths)
    # The YAML may select a subset, but each original must retain its mirror.
    pairs = {(expected_paths[i], expected_paths[i + 1]) for i in range(0, len(expected_paths), 2)}
    assert all(pair in pairs for pair in zip(selected[::2], selected[1::2]))


def test_all_mirrored_amp_features_and_target_foot_fk():
    if not robot_urdf().is_file():
        pytest.skip("Requires supplied Pi12 URDF")
    files = selected_motion_files(motion_yaml())
    loader = AMPLoaderPi12("cpu", 0.02, motion_files=files, urdf_path=robot_urdf())
    np.testing.assert_allclose(loader.trajectory_weights, np.full(len(files), 1 / len(files)))
    for original, mirrored in zip(files[::2], files[1::2]):
        before, dt, _ = loader._load_npz_motion(original)
        after, mirror_dt, _ = loader._load_npz_motion(mirrored)
        assert dt == mirror_dt
        expected = before.copy()
        expected[:, :3] *= [1, -1, 1]  # Polar vector: local linear velocity.
        expected[:, 3:6] *= [-1, 1, -1]  # Axial vector: local angular velocity.
        order = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
        signs = [1, -1, -1, 1, 1, -1] * 2
        for start in (6, 18):
            expected[:, start:start + 12] = before[:, start + np.array(order)] * signs
        expected[:, 30:36] = before[:, [33, 34, 35, 30, 31, 32]] * [1, -1, 1, 1, -1, 1]
        np.testing.assert_allclose(after[:, :30], expected[:, :30], atol=2e-5, rtol=1e-5, err_msg=Path(original).name)
        # Small geometric asymmetry in the actual left/right URDF is permitted.
        np.testing.assert_allclose(after[:, 30:], expected[:, 30:], atol=1e-4, rtol=1e-4, err_msg=Path(original).name)
