"""Play a PiPlus H0W stair motion together with its fitted STL terrain."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml

from humanoidverse.utils.asset_paths import resolve_asset_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w"
DEFAULT_METADATA = DEFAULT_DATASET_DIR / "motion_matched_terrain/metadata.yaml"
DEFAULT_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "stairs_climbing_up_start_R_20260817_PiPlus_S_12L8A0G2H0W_retargetted.npz",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--terrain-file", type=Path, default=None, help="Override terrain selected from metadata.yaml.")
    parser.add_argument("--robot-xml", default=DEFAULT_ROBOT_XML)
    parser.add_argument("--output", type=Path, default=None, help="Write MP4 instead of opening the interactive viewer.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolve_terrain(metadata_path: Path, motion_file: Path, terrain_override: Path | None) -> Path:
    if terrain_override is not None:
        return terrain_override.resolve()
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    terrain_by_id = {int(item["terrain_id"]): item["terrain_file"] for item in metadata["terrains"]}
    matches = [item for item in metadata["motion_files"] if Path(item["motion_file"]).name == motion_file.name]
    if len(matches) != 1:
        raise ValueError(f"Expected one metadata entry for {motion_file.name}, found {len(matches)}.")
    relative_terrain = terrain_by_id[int(matches[0]["terrain_id"])]
    return (metadata_path.parent / relative_terrain).resolve()


def load_motion(motion_file: Path) -> tuple[np.ndarray, list[str], float]:
    with np.load(motion_file, allow_pickle=False) as motion:
        required = {"joint_pos", "joint_names", "base_pos_w", "base_quat_w", "framerate"}
        missing = required.difference(motion.files)
        if missing:
            raise KeyError(f"{motion_file} is missing fields: {sorted(missing)}")
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        joint_names = [str(name) for name in motion["joint_names"]]
        base_pos = np.asarray(motion["base_pos_w"], dtype=np.float64)
        base_quat_wxyz = np.asarray(motion["base_quat_w"], dtype=np.float64)
        framerate = float(motion["framerate"])
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 22:
        raise ValueError(f"Expected joint_pos shape (T, 22), got {joint_pos.shape}.")
    if base_pos.shape != (joint_pos.shape[0], 3) or base_quat_wxyz.shape != (joint_pos.shape[0], 4):
        raise ValueError("Motion root arrays do not match joint_pos frame count.")
    qpos = np.concatenate((base_pos, base_quat_wxyz, joint_pos), axis=1)
    return qpos, joint_names, framerate


def load_model_with_terrain(robot_xml: Path, terrain_file: Path):
    import mujoco

    root = ET.parse(robot_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    mesh_dir = (robot_xml.parent / compiler.attrib.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(mesh_dir))
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"{robot_xml} must contain asset and worldbody elements.")
    ET.SubElement(asset, "mesh", name="motion_matched_stairs", file=str(terrain_file))
    ET.SubElement(
        worldbody,
        "geom",
        name="motion_matched_stairs",
        type="mesh",
        mesh="motion_matched_stairs",
        rgba="0.42 0.46 0.52 1",
        contype="1",
        conaffinity="1",
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml") as temporary_xml:
        ET.ElementTree(root).write(temporary_xml, encoding="utf-8", xml_declaration=True)
        temporary_xml.flush()
        return mujoco.MjModel.from_xml_path(temporary_xml.name)


def validate_contract(model, joint_names: list[str], qpos: np.ndarray) -> None:
    import mujoco

    model_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(1, model.njnt)]
    if joint_names != model_joint_names:
        raise ValueError(f"Motion/MJCF joint order mismatch:\nmotion={joint_names}\nMJCF={model_joint_names}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos has {qpos.shape[1]} values, model expects {model.nq}.")
    if not np.isfinite(qpos).all():
        raise ValueError("Motion qpos contains non-finite values.")


def make_tracking_camera(model):
    import mujoco

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    camera.distance = 2.2
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


def set_frame(model, data, qpos: np.ndarray) -> None:
    import mujoco

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def play_viewer(model, qpos: np.ndarray, framerate: float, stride: int) -> None:
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        for frame in qpos[::stride]:
            if not viewer.is_running():
                break
            frame_start = time.time()
            set_frame(model, data, frame)
            viewer.sync()
            time.sleep(max(0.0, stride / framerate - (time.time() - frame_start)))


def render_video(model, qpos: np.ndarray, framerate: float, stride: int, output: Path, width: int, height: int) -> None:
    import mediapy as media
    import mujoco

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = make_tracking_camera(model)
    frames = []
    for index, frame in enumerate(qpos[::stride]):
        set_frame(model, data, frame)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
        if index == 0 or (index + 1) % 100 == 0:
            print(f"Rendered {index + 1}/{len(qpos[::stride])} frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output), frames, fps=max(1, round(framerate / stride)))
    renderer.close()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("stride must be at least 1.")
    motion_file = args.motion_file.resolve()
    metadata = args.metadata.resolve()
    terrain_file = resolve_terrain(metadata, motion_file, args.terrain_file)
    robot_xml = resolve_asset_path("", args.robot_xml).resolve()
    qpos, joint_names, framerate = load_motion(motion_file)
    end = len(qpos) if args.max_frames is None else min(len(qpos), args.start + args.max_frames)
    qpos = qpos[args.start:end]
    if not len(qpos):
        raise ValueError(f"Selected empty frame range [{args.start}, {end}).")

    if args.output is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
    model = load_model_with_terrain(robot_xml, terrain_file)
    validate_contract(model, joint_names, qpos)
    print(f"Motion: {motion_file}")
    print(f"Terrain: {terrain_file}")
    print(f"Robot XML: {robot_xml}")
    print(f"Frames: {len(qpos)} at {framerate:g} fps; qpos={qpos.shape[1]}, model.nq={model.nq}")
    if args.validate_only:
        print("Motion, robot, and terrain contracts are valid.")
    elif args.output is None:
        play_viewer(model, qpos, framerate, args.stride)
    else:
        render_video(model, qpos, framerate, args.stride, args.output, args.width, args.height)
        print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
