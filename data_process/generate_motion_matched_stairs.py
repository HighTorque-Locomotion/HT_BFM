"""Generate motion-matched stair meshes and InstinctLab metadata.

The source motion files contain robot kinematics but no recorded terrain.  This
tool estimates a staircase from the root displacement of every motion and keeps
the source files unchanged.  The generated geometry is therefore an explicit,
reproducible approximation, not a reconstruction of the capture scene.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "motion_matched_terrain"
DEFAULT_ROBOT_XML = (
    REPO_ROOT.parent / "ht_urdf/ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument(
        "--step-height",
        type=float,
        default=None,
        help="Override fitted stair rise in metres. By default it is fitted from stationary foot contacts.",
    )
    parser.add_argument(
        "--tread-depth",
        type=float,
        default=None,
        help="Override fitted tread depth in metres. By default one shared depth is fitted across all motions.",
    )
    parser.add_argument("--stair-width", type=float, default=1.20, help="Stair width in metres, along world X.")
    parser.add_argument("--run-fraction", type=float, default=0.85, help="Fraction of endpoint travel occupied by treads.")
    parser.add_argument("--terrain-size", type=float, default=3.0, help="Square terrain side length in metres.")
    parser.add_argument("--ground-thickness", type=float, default=0.05)
    parser.add_argument(
        "--stance-speed-threshold",
        type=float,
        default=None,
        help="Override maximum stance foot speed in m/s. Default: 0.12 for walking and 0.35 for jogging.",
    )
    parser.add_argument("--min-stance-frames", type=int, default=3)
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="Do not run MuJoCo FK; use the older root-endpoint approximation (requires --step-height).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_box(vertices: list[list[float]], faces: list[list[int]], bounds: tuple[float, float, float, float, float, float]) -> None:
    x0, x1, y0, y1, z0, z1 = bounds
    offset = len(vertices)
    vertices.extend(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces.extend(
        [
            [offset + i for i in face]
            for face in (
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            )
        ]
    )


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    header = b"HT_BFM motion-matched stairs".ljust(80, b"\0")
    records = []
    for face in faces:
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        normal = normal / norm if norm > 0.0 else np.zeros(3)
        values = np.concatenate((normal, triangle.reshape(-1))).astype(np.float32)
        records.append(struct.pack("<12fH", *values, 0))
    path.write_bytes(header + struct.pack("<I", len(faces)) + b"".join(records))


def endpoint_window_mean(values: np.ndarray, at_start: bool) -> np.ndarray:
    window = max(3, min(len(values) // 10, 15))
    return values[:window].mean(axis=0) if at_start else values[-window:].mean(axis=0)


def load_motion(path: Path) -> dict[str, np.ndarray | float | list[str]]:
    with np.load(path, allow_pickle=False) as motion:
        required = {"joint_pos", "joint_names", "base_pos_w", "base_quat_w", "framerate"}
        missing = required.difference(motion.files)
        if missing:
            raise KeyError(f"{path} is missing fields: {sorted(missing)}")
        result = {
            "joint_pos": np.asarray(motion["joint_pos"], dtype=np.float64),
            "joint_names": [str(name) for name in motion["joint_names"]],
            "base_pos_w": np.asarray(motion["base_pos_w"], dtype=np.float64),
            "base_quat_w": np.asarray(motion["base_quat_w"], dtype=np.float64),
            "framerate": float(motion["framerate"]),
        }
    joint_pos = result["joint_pos"]
    base_pos = result["base_pos_w"]
    base_quat = result["base_quat_w"]
    if not isinstance(joint_pos, np.ndarray) or joint_pos.ndim != 2 or joint_pos.shape[1] != 22:
        raise ValueError(f"{path}: expected joint_pos (T, 22), got {getattr(joint_pos, 'shape', None)}.")
    if not isinstance(base_pos, np.ndarray) or base_pos.shape != (joint_pos.shape[0], 3):
        raise ValueError(f"{path}: invalid base_pos_w shape: {getattr(base_pos, 'shape', None)}.")
    if not isinstance(base_quat, np.ndarray) or base_quat.shape != (joint_pos.shape[0], 4):
        raise ValueError(f"{path}: invalid base_quat_w shape: {getattr(base_quat, 'shape', None)}.")
    if not np.isfinite(joint_pos).all() or not np.isfinite(base_pos).all() or not np.isfinite(base_quat).all():
        raise ValueError(f"{path}: motion contains non-finite values.")
    return result


def load_mujoco_foot_solver(robot_xml: Path):
    try:
        import mujoco
    except ImportError as error:
        raise ImportError("MuJoCo is required for foot-contact fitting; install project dependencies or use --root-only.") from error

    if not robot_xml.is_file():
        raise FileNotFoundError(f"PiPlus robot XML not found: {robot_xml}")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(1, model.njnt)]
    foot_geom_ids = []
    for body_name in ("l_ankle_roll_link", "r_ankle_roll_link"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_ids = range(model.body_geomadr[body_id], model.body_geomadr[body_id] + model.body_geomnum[body_id])
        collision_geoms = [geom_id for geom_id in geom_ids if model.geom_group[geom_id] == 0]
        if len(collision_geoms) != 1:
            raise ValueError(f"Expected one collision mesh on {body_name}, found {collision_geoms}.")
        foot_geom_ids.append(collision_geoms[0])
    return mujoco, model, data, joint_names, foot_geom_ids


def compute_foot_sole_trajectories(motion: dict, solver) -> np.ndarray:
    mujoco, model, data, model_joint_names, foot_geom_ids = solver
    if motion["joint_names"] != model_joint_names:
        raise ValueError(f"Motion/MJCF joint order mismatch:\nmotion={motion['joint_names']}\nMJCF={model_joint_names}")
    joint_pos = motion["joint_pos"]
    foot_soles = np.empty((joint_pos.shape[0], 2, 3), dtype=np.float64)
    for frame in range(joint_pos.shape[0]):
        data.qpos[:3] = motion["base_pos_w"][frame]
        data.qpos[3:7] = motion["base_quat_w"][frame]
        data.qpos[7:] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        for foot_index, geom_id in enumerate(foot_geom_ids):
            mesh_id = model.geom_dataid[geom_id]
            vertex_start = model.mesh_vertadr[mesh_id]
            vertex_end = vertex_start + model.mesh_vertnum[mesh_id]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            vertices = model.mesh_vert[vertex_start:vertex_end] @ rotation.T + data.geom_xpos[geom_id]
            sole_mask = vertices[:, 2] <= vertices[:, 2].min() + 0.003
            foot_soles[frame, foot_index] = (
                np.median(vertices[sole_mask, 0]),
                np.median(vertices[sole_mask, 1]),
                vertices[:, 2].min(),
            )
    return foot_soles


def extract_stance_contacts(
    foot_soles: np.ndarray, framerate: float, speed_threshold: float, min_frames: int
) -> list[dict[str, float | int | str]]:
    velocity = np.linalg.norm(np.gradient(foot_soles, axis=0) * framerate, axis=2)
    contacts = []
    for foot_index, foot_name in enumerate(("left", "right")):
        stance = velocity[:, foot_index] < speed_threshold
        starts = np.where(np.diff(np.r_[False, stance].astype(np.int8)) == 1)[0]
        ends = np.where(np.diff(np.r_[stance, False].astype(np.int8)) == -1)[0] + 1
        for start, end in zip(starts, ends):
            if end - start < min_frames:
                continue
            position = np.median(foot_soles[start:end, foot_index], axis=0)
            contacts.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "center_frame": int((start + end - 1) // 2),
                    "foot": foot_name,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "median_speed": float(np.median(velocity[start:end, foot_index])),
                }
            )
    return sorted(contacts, key=lambda contact: int(contact["center_frame"]))


def select_stance_speed_threshold(motion: dict, override: float | None) -> tuple[float, float]:
    base_pos = motion["base_pos_w"]
    duration = max((base_pos.shape[0] - 1) / float(motion["framerate"]), 1e-6)
    average_progress_speed = abs(float(base_pos[-1, 1] - base_pos[0, 1])) / duration
    if override is not None:
        if override <= 0.0:
            raise ValueError("stance-speed-threshold must be positive.")
        return override, average_progress_speed
    return (0.35 if average_progress_speed >= 0.20 else 0.12), average_progress_speed


def estimate_step_height(all_contacts: list[list[dict[str, float | int | str]]]) -> tuple[float, float]:
    height_changes = []
    for contacts in all_contacts:
        for previous, current in zip(contacts, contacts[1:]):
            delta_z = abs(float(current["z"]) - float(previous["z"]))
            delta_y = abs(float(current["y"]) - float(previous["y"]))
            if 0.035 <= delta_z <= 0.11 and 0.025 <= delta_y <= 0.20:
                height_changes.append(delta_z)
    if len(height_changes) < 3:
        raise ValueError(f"Only {len(height_changes)} usable stance transitions; cannot fit stair height.")
    raw_height = float(np.median(height_changes))
    # Retargeted capture values contain centimetre-level FK/contact noise. Stair
    # dimensions are authored in whole centimetres, so retain that resolution.
    fitted_height = round(raw_height * 100.0) / 100.0
    return fitted_height, raw_height


def fit_staircase_to_contacts(
    base_pos: np.ndarray,
    contacts: list[dict[str, float | int | str]],
    step_height: float,
    run_fraction: float,
    tread_depth_override: float | None = None,
) -> dict[str, float | int | str | list]:
    inference = infer_staircase(base_pos, step_height, run_fraction)
    direction = float(inference["high_direction"])
    usable = [contact for contact in contacts if float(contact["z"]) >= -0.03]
    for contact in usable:
        contact["level"] = max(0, int(round(float(contact["z"]) / step_height)))
    num_steps = max(int(inference["num_steps"]), max(int(contact["level"]) for contact in usable))
    level_y = {}
    for level in range(num_steps + 1):
        positions = [float(contact["y"]) for contact in usable if int(contact["level"]) == level]
        if positions:
            level_y[level] = float(np.median(positions))

    depth_samples = []
    levels = sorted(level_y)
    for low_index, low_level in enumerate(levels):
        for high_level in levels[low_index + 1 :]:
            depth = direction * (level_y[high_level] - level_y[low_level]) / (high_level - low_level)
            if 0.04 <= depth <= 0.20:
                depth_samples.append(depth)
    if not depth_samples:
        raise ValueError("Stable contacts do not span enough stair levels to fit tread depth.")
    raw_tread_depth = float(np.median(depth_samples))
    tread_depth = tread_depth_override if tread_depth_override is not None else round(raw_tread_depth * 100.0) / 100.0
    initial_center_level_zero = float(np.median([y - direction * level * tread_depth for level, y in level_y.items()]))

    def terrain_level_at_y(y: float, center_level_zero: float) -> int:
        first_step_boundary = center_level_zero + direction * 0.5 * tread_depth
        progress = direction * (y - first_step_boundary)
        if progress < 0.0:
            return 0
        return min(num_steps, int(math.floor(progress / tread_depth)) + 1)

    # Foot placement is not exactly at tread centers. Optimize the global phase
    # of the stair boundaries against the actual sole heights.
    center_candidates = np.linspace(
        initial_center_level_zero - tread_depth,
        initial_center_level_zero + tread_depth,
        401,
    )
    phase_errors = [
        np.mean(
            [
                abs(float(contact["z"]) - terrain_level_at_y(float(contact["y"]), candidate) * step_height)
                for contact in usable
            ]
        )
        for candidate in center_candidates
    ]
    center_level_zero = float(center_candidates[int(np.argmin(phase_errors))])
    run_low = center_level_zero + direction * 0.5 * tread_depth
    run_high = center_level_zero + direction * (num_steps + 0.5) * tread_depth

    vertical_errors = []
    terrain_vertical_errors = []
    horizontal_errors = []
    for contact in usable:
        level = min(int(contact["level"]), num_steps)
        vertical_errors.append(float(contact["z"]) - level * step_height)
        terrain_level = terrain_level_at_y(float(contact["y"]), center_level_zero)
        terrain_vertical_errors.append(float(contact["z"]) - terrain_level * step_height)
        expected_y = center_level_zero + direction * level * tread_depth
        horizontal_errors.append(float(contact["y"]) - expected_y)

    inference.update(
        {
            "fit_method": "MuJoCo foot FK and stationary stance contacts",
            "num_steps": num_steps,
            "total_rise": num_steps * step_height,
            "stair_run": num_steps * tread_depth,
            "tread_depth": tread_depth,
            "raw_tread_depth": raw_tread_depth,
            "run_low": run_low,
            "run_high": run_high,
            "level_zero_center_y": center_level_zero,
            "num_stance_contacts": len(usable),
            "stance_contacts": usable,
            "contact_vertical_mae": float(np.mean(np.abs(vertical_errors))),
            "contact_vertical_max_error": float(np.max(np.abs(vertical_errors))),
            "terrain_contact_vertical_mae": float(np.mean(np.abs(terrain_vertical_errors))),
            "terrain_contact_vertical_max_error": float(np.max(np.abs(terrain_vertical_errors))),
            "contact_horizontal_mae": float(np.mean(np.abs(horizontal_errors))),
            "contact_horizontal_max_error": float(np.max(np.abs(horizontal_errors))),
        }
    )
    return inference


def infer_staircase(base_pos: np.ndarray, step_height: float, run_fraction: float) -> dict[str, float | int | str]:
    start = endpoint_window_mean(base_pos, at_start=True)
    end = endpoint_window_mean(base_pos, at_start=False)
    if abs(float(end[1] - start[1])) < 1e-4:
        raise ValueError("Motion has no usable displacement along world Y.")
    if abs(float(end[2] - start[2])) < step_height * 0.5:
        raise ValueError("Motion root height change is too small to infer stairs.")

    ascending = end[2] > start[2]
    low = start if ascending else end
    high = end if ascending else start
    height_delta = float(high[2] - low[2])
    num_steps = max(1, int(round(height_delta / step_height)))
    total_rise = num_steps * step_height
    low_y, high_y = float(low[1]), float(high[1])
    direction = math.copysign(1.0, high_y - low_y)
    endpoint_run = abs(high_y - low_y)
    stair_run = endpoint_run * run_fraction
    tread_depth = stair_run / num_steps
    center_y = 0.5 * (low_y + high_y)
    run_low = center_y - direction * stair_run * 0.5
    run_high = center_y + direction * stair_run * 0.5
    return {
        "motion_direction": "up" if ascending else "down",
        "num_steps": num_steps,
        "step_height": step_height,
        "total_rise": total_rise,
        "observed_root_height_delta": height_delta,
        "observed_endpoint_run": endpoint_run,
        "stair_run": stair_run,
        "tread_depth": tread_depth,
        "low_y": low_y,
        "high_y": high_y,
        "run_low": run_low,
        "run_high": run_high,
        "high_direction": direction,
    }


def build_stair_mesh(
    inference: dict[str, float | int | str], terrain_size: float, stair_width: float, ground_thickness: float
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    half_terrain = terrain_size * 0.5
    half_width = stair_width * 0.5
    add_box(vertices, faces, (-half_terrain, half_terrain, -half_terrain, half_terrain, -ground_thickness, 0.0))

    num_steps = int(inference["num_steps"])
    step_height = float(inference["step_height"])
    tread_depth = float(inference["tread_depth"])
    direction = float(inference["high_direction"])
    run_low = float(inference["run_low"])
    run_high = float(inference["run_high"])

    # One solid box per tread. The final tread extends into a high landing but
    # stops short of the outer border, keeping InstinctLab's border at z=0.
    high_landing_end = math.copysign(half_terrain - 0.10, direction)
    for level in range(1, num_steps + 1):
        tread_start = run_low + direction * tread_depth * (level - 1)
        tread_end = run_low + direction * tread_depth * level
        if level == num_steps:
            tread_end = high_landing_end
        y0, y1 = sorted((tread_start, tread_end))
        add_box(vertices, faces, (-half_width, half_width, y0, y1, -ground_thickness, level * step_height))

    mesh_vertices = np.asarray(vertices, dtype=np.float64)
    mesh_faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(mesh_vertices).all():
        raise ValueError("Generated mesh contains non-finite vertices.")
    if max(abs(float(run_low)), abs(float(run_high))) >= half_terrain - 0.10:
        raise ValueError(f"Inferred stair run does not fit terrain-size={terrain_size}.")
    return mesh_vertices, mesh_faces


def generate(args: argparse.Namespace) -> dict:
    root_only = getattr(args, "root_only", False)
    if args.step_height is not None and args.step_height <= 0.0:
        raise ValueError("step-height must be positive when provided.")
    if args.tread_depth is not None and args.tread_depth <= 0.0:
        raise ValueError("tread-depth must be positive when provided.")
    if root_only and args.step_height is None:
        raise ValueError("--root-only requires an explicit --step-height.")
    if args.stair_width <= 0.0 or args.ground_thickness <= 0.0:
        raise ValueError("Stair width and ground thickness must be positive.")
    if not 0.0 < args.run_fraction <= 1.0:
        raise ValueError("run-fraction must be in (0, 1].")
    if args.terrain_size <= args.stair_width:
        raise ValueError("terrain-size must be greater than stair-width.")

    motion_files = sorted(args.input_dir.glob("*_retargetted.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No *_retargetted.npz files found in {args.input_dir}.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite to regenerate it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir = args.output_dir / "terrains"
    terrain_dir.mkdir(exist_ok=True)

    motions = [(motion_file, load_motion(motion_file)) for motion_file in motion_files]
    expected_joint_names = motions[0][1]["joint_names"]
    for motion_file, motion in motions[1:]:
        if motion["joint_names"] != expected_joint_names:
            raise ValueError(f"{motion_file}: joint order differs from other motions.")

    contacts_by_motion: list[list[dict[str, float | int | str]]] = [[] for _ in motions]
    stance_thresholds: list[float | None] = [None for _ in motions]
    average_progress_speeds: list[float | None] = [None for _ in motions]
    raw_step_height = None
    if not root_only:
        solver = load_mujoco_foot_solver(args.robot_xml)
        contacts_by_motion = []
        for _, motion in motions:
            stance_threshold, average_progress_speed = select_stance_speed_threshold(
                motion, args.stance_speed_threshold
            )
            stance_thresholds[len(contacts_by_motion)] = stance_threshold
            average_progress_speeds[len(contacts_by_motion)] = average_progress_speed
            contacts_by_motion.append(
                extract_stance_contacts(
                    compute_foot_sole_trajectories(motion, solver),
                    float(motion["framerate"]),
                    stance_threshold,
                    args.min_stance_frames,
                )
            )
        fitted_step_height, raw_step_height = estimate_step_height(contacts_by_motion)
        step_height = args.step_height if args.step_height is not None else fitted_step_height
    else:
        step_height = args.step_height

    raw_tread_depth = None
    fitted_tread_depth = args.tread_depth
    if not root_only:
        preliminary_inferences = [
            fit_staircase_to_contacts(motion["base_pos_w"], contacts, step_height, args.run_fraction)
            for (_, motion), contacts in zip(motions, contacts_by_motion)
        ]
        raw_tread_depth = float(np.median([inference["raw_tread_depth"] for inference in preliminary_inferences]))
        if fitted_tread_depth is None:
            fitted_tread_depth = round(raw_tread_depth * 100.0) / 100.0

    metadata: dict[str, list[dict]] = {"terrains": [], "motion_files": []}
    diagnostics = {
        "schema_version": 2,
        "source_directory": str(args.input_dir.resolve()),
        "robot_xml": None if root_only else str(args.robot_xml.resolve()),
        "method": (
            "stairs fitted to MuJoCo FK stationary foot contacts"
            if not root_only
            else "stairs inferred from averaged root-pose endpoints"
        ),
        "assumption": "Generated geometry approximates the missing capture terrain; contact errors quantify the fit.",
        "parameters": {
            "step_height": step_height,
            "raw_fitted_step_height": raw_step_height,
            "tread_depth": fitted_tread_depth,
            "raw_fitted_tread_depth": raw_tread_depth,
            "stair_width": args.stair_width,
            "run_fraction": args.run_fraction,
            "terrain_size": args.terrain_size,
            "ground_thickness": args.ground_thickness,
            "stance_speed_threshold_override": None if root_only else args.stance_speed_threshold,
            "min_stance_frames": None if root_only else args.min_stance_frames,
        },
        "motions": [],
    }

    for terrain_id, ((motion_file, motion), contacts) in enumerate(zip(motions, contacts_by_motion)):
        base_pos = motion["base_pos_w"]
        if root_only:
            inference = infer_staircase(base_pos, step_height, args.run_fraction)
            inference["fit_method"] = "root endpoints"
        else:
            inference = fit_staircase_to_contacts(
                base_pos, contacts, step_height, args.run_fraction, tread_depth_override=fitted_tread_depth
            )
        vertices, faces = build_stair_mesh(inference, args.terrain_size, args.stair_width, args.ground_thickness)
        terrain_file = terrain_dir / f"{motion_file.stem}_stairs.stl"
        write_binary_stl(terrain_file, vertices, faces)

        metadata["terrains"].append(
            {"terrain_id": terrain_id, "terrain_file": terrain_file.relative_to(args.output_dir).as_posix()}
        )
        metadata["motion_files"].append(
            {
                "terrain_id": terrain_id,
                "motion_file": Path("..", motion_file.name).as_posix(),
                "weight": 1.0,
            }
        )
        diagnostics["motions"].append(
            {
                "terrain_id": terrain_id,
                "motion_file": motion_file.name,
                "terrain_file": terrain_file.relative_to(args.output_dir).as_posix(),
                "num_frames": int(motion["joint_pos"].shape[0]),
                "num_joints": int(motion["joint_pos"].shape[1]),
                "stance_speed_threshold": stance_thresholds[terrain_id],
                "average_progress_speed": average_progress_speeds[terrain_id],
                **inference,
                "mesh_vertices": int(vertices.shape[0]),
                "mesh_triangles": int(faces.shape[0]),
            }
        )

    (args.output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    (args.output_dir / "terrain_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return diagnostics


def main() -> None:
    args = parse_args()
    diagnostics = generate(args)
    print(f"Generated {len(diagnostics['motions'])} motion-matched stair terrains in {args.output_dir}")
    for motion in diagnostics["motions"]:
        print(
            f"  {motion['motion_file']}: {motion['motion_direction']}, "
            f"{motion['num_steps']} steps, rise={motion['total_rise']:.3f} m, "
            f"tread={motion['tread_depth']:.3f} m"
        )


if __name__ == "__main__":
    main()
