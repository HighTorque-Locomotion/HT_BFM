from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.generate_motion_matched_stairs import fit_staircase_to_contacts, generate, infer_staircase

JOINT_NAMES = np.asarray([f"joint_{index}" for index in range(22)])


def write_motion(path: Path, start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
    frames = 31
    np.savez(
        path,
        joint_pos=np.zeros((frames, 22), dtype=np.float32),
        joint_names=JOINT_NAMES,
        base_pos_w=np.linspace(start, end, frames),
        base_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        framerate=np.asarray(30.0),
    )


def test_infer_staircase_handles_up_and_down_motion():
    up = infer_staircase(np.linspace([0.0, 0.0, 0.38], [0.0, 0.9, 0.88], 31), 0.1, 0.8)
    down = infer_staircase(np.linspace([0.0, 0.0, 0.88], [0.0, -0.9, 0.38], 31), 0.1, 0.8)
    assert up["motion_direction"] == "up"
    assert down["motion_direction"] == "down"
    assert up["num_steps"] == down["num_steps"] == 5
    # Endpoint means intentionally exclude the most transient first/last frames.
    assert np.isclose(up["tread_depth"], 0.1344)


def test_contact_fit_recovers_stair_count_depth_and_phase():
    base_pos = np.linspace([0.0, 0.0, 0.38], [0.0, 0.5, 0.73], 31)
    contacts = [
        {"center_frame": level * 5, "foot": "left", "x": 0.0, "y": level * 0.1, "z": level * 0.07}
        for level in range(6)
    ]
    fitted = fit_staircase_to_contacts(base_pos, contacts, 0.07, 0.85, tread_depth_override=0.1)
    assert fitted["num_steps"] == 5
    assert np.isclose(fitted["tread_depth"], 0.1)
    assert fitted["terrain_contact_vertical_max_error"] < 1e-9


def test_generate_writes_instinctlab_metadata_and_meshes(tmp_path: Path):
    input_dir = tmp_path / "motions"
    output_dir = input_dir / "motion_matched_terrain"
    input_dir.mkdir()
    write_motion(input_dir / "stairs_up_retargetted.npz", (0.0, 0.0, 0.38), (0.0, 0.9, 0.88))
    write_motion(input_dir / "stairs_down_retargetted.npz", (0.0, 0.0, 0.88), (0.0, -0.9, 0.38))
    args = Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        step_height=0.1,
        tread_depth=None,
        stair_width=1.2,
        run_fraction=0.85,
        terrain_size=3.0,
        ground_thickness=0.05,
        root_only=True,
        overwrite=False,
    )

    diagnostics = generate(args)
    metadata = yaml.safe_load((output_dir / "metadata.yaml").read_text())
    assert len(metadata["terrains"]) == len(metadata["motion_files"]) == 2
    assert {entry["terrain_id"] for entry in metadata["terrains"]} == {0, 1}
    assert all((output_dir / entry["terrain_file"]).is_file() for entry in metadata["terrains"])
    assert all((output_dir / entry["motion_file"]).is_file() for entry in metadata["motion_files"])
    assert len(diagnostics["motions"]) == 2
    assert all(motion["num_joints"] == 22 for motion in diagnostics["motions"])
