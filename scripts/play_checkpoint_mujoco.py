#!/usr/bin/env python3
"""Play an exported HT-BFM ONNX policy bundle in MuJoCo.

The default bundle is ``checkpoint/bundle.zip``.  The policy accepts velocity
commands ``vx vy wz`` and controls PiPlus joint positions through the PD gains
stored in the bundle manifest.
"""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from humanoidverse.utils.asset_paths import resolve_asset_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO_ROOT / "checkpoint" / "bundle.zip"
DEFAULT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"
ARMATURE_BY_GROUP = {"head": 0.001976, "shoulder": 0.008234, "upper_arm": 0.008234, "elbow": 0.008234, "leg": 0.013212}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="bundle.zip or an extracted bundle directory")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML, help="MuJoCo XML path (local or package://)")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=(0.3, 0.0, 0.0))
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many policy steps; unlimited by default")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace simulation to wall-clock time")
    parser.add_argument("--log-every", type=int, default=100, help="Print state every N policy steps; 0 disables logs")
    parser.add_argument("--base-height", type=float, default=0.385)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def find_bundle_root(path: Path, temporary_dir: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Bundle is not a zip file: {path}")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temporary_dir)
        candidates = [temporary_dir, temporary_dir / "bundle"]
    else:
        candidates = [path, path / "bundle"]
    for candidate in candidates:
        if (candidate / "exported" / "actor.onnx").is_file() and (candidate / "exported" / "policy_manifest.json").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find exported/actor.onnx and policy_manifest.json under {path}")


def joint_addresses(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses, qvel_addresses = [], []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name!r} from the policy manifest is absent from the MuJoCo model")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        qvel_addresses.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addresses), np.asarray(qvel_addresses)


def body_vector(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, world_vector: np.ndarray) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    quaternion_conjugate = data.xquat[body_id].copy()
    quaternion_conjugate[1:] *= -1.0
    mujoco.mju_rotVecQuat(result, world_vector, quaternion_conjugate)
    return result


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="ht_bfm_bundle_") as temporary_name:
        bundle_root = find_bundle_root(args.bundle, Path(temporary_name))
        manifest = json.loads((bundle_root / "exported" / "policy_manifest.json").read_text())
        actor_path = bundle_root / "exported" / "actor.onnx"

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(actor_path), providers=providers)
        input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
        if input_meta.shape[-1] != 363 or output_meta.shape[-1] != 22:
            raise ValueError(f"Unexpected actor contract: input={input_meta.shape}, output={output_meta.shape}")

        xml_path = resolve_asset_path("", args.xml)
        if not xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo robot XML not found: {xml_path}. Set HT_URDF_ROOT or pass --xml.")
        spec = mujoco.MjSpec.from_file(str(xml_path))
        model = spec.compile()
        data = mujoco.MjData(model)

        control = manifest["control"]
        policy_names = manifest["joints"]["policy_order"]
        observation_names = manifest["joints"]["observation_order"]
        policy_qpos, policy_qvel = joint_addresses(model, policy_names)
        obs_qpos, obs_qvel = joint_addresses(model, observation_names)
        for name, dof_address in zip(policy_names, policy_qvel, strict=True):
            group = next((key for key in ARMATURE_BY_GROUP if key in name), "leg")
            model.dof_armature[dof_address] = ARMATURE_BY_GROUP[group]
            model.dof_frictionloss[dof_address] = 0.02
        default_by_name = control["default_pose"]
        default_policy = np.asarray([default_by_name[name] for name in policy_names], dtype=np.float64)
        default_obs = np.asarray([default_by_name[name] for name in observation_names], dtype=np.float32)
        action_scale = np.asarray(control["action_scale"], dtype=np.float64)
        stiffness = np.asarray([control["stiffness"][name] for name in policy_names], dtype=np.float64)
        damping = np.asarray([control["damping"][name] for name in policy_names], dtype=np.float64)
        torque_limits = np.asarray([control["torque_limits"][name] for name in policy_names], dtype=np.float64)
        policy_hz = float(control["frequency_hz"])
        decimation = int(control["decimation"])
        model.opt.timestep = 1.0 / (policy_hz * decimation)

        data.qpos[:3] = (0.0, 0.0, args.base_height)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[policy_qpos] = default_policy
        mujoco.mj_forward(model, data)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_id < 0:
            raise ValueError("MuJoCo model has no 'base_link' body")

        # History is grouped alphabetically exactly like HT-BFM training:
        # actions, base_ang_vel, dof_pos, dof_vel, projected_gravity.
        history: deque[dict[str, np.ndarray]] = deque(maxlen=4)
        last_action_policy = np.zeros(len(policy_names), dtype=np.float32)
        policy_to_observation = np.asarray([policy_names.index(name) for name in observation_names])

        def measure() -> dict[str, np.ndarray]:
            local_velocity = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, local_velocity, 1)
            return {
                "actions": last_action_policy[policy_to_observation].copy(),
                "base_ang_vel": (local_velocity[:3] * 0.25).astype(np.float32),
                "dof_pos": (data.qpos[obs_qpos] - default_obs).astype(np.float32),
                "dof_vel": data.qvel[obs_qvel].astype(np.float32),
                "projected_gravity": body_vector(model, data, base_id, np.asarray([0.0, 0.0, -1.0])).astype(np.float32),
            }

        initial = measure()
        for _ in range(4):
            history.append({key: value.copy() for key, value in initial.items()})

        viewer = None
        if not args.headless:
            mujoco_viewer = importlib.import_module("mujoco.viewer")
            viewer = mujoco_viewer.launch_passive(model, data)
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0

        command = np.asarray(args.command, dtype=np.float32)
        step = 0
        try:
            while (args.max_steps is None or step < args.max_steps) and (viewer is None or viewer.is_running()):
                started = time.perf_counter()
                current = measure()
                grouped_history = np.concatenate(
                    [np.concatenate([frame[key] for frame in history]) for key in ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")]
                )
                actor_obs = np.concatenate(
                    [
                        current["dof_pos"],
                        current["dof_vel"],
                        current["projected_gravity"],
                        current["base_ang_vel"],
                        current["actions"],
                        grouped_history,
                        command,
                    ]
                ).astype(np.float32)[None]
                action = session.run([output_meta.name], {input_meta.name: actor_obs})[0][0].astype(np.float32)
                if not np.isfinite(action).all():
                    raise FloatingPointError("Policy produced NaN/Inf actions")
                last_action_policy = action
                position_target = default_policy + action_scale * action
                for _ in range(decimation):
                    torque = stiffness * (position_target - data.qpos[policy_qpos]) - damping * data.qvel[policy_qvel]
                    data.qfrc_applied[policy_qvel] = np.clip(torque, -torque_limits, torque_limits)
                    mujoco.mj_step(model, data)
                history.appendleft(measure())
                step += 1

                if viewer is not None:
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()
                if args.log_every > 0 and step % args.log_every == 0:
                    print(f"step={step} base_xyz={np.round(data.qpos[:3], 3)} command={command} action_rms={np.sqrt(np.mean(action**2)):.3f}")
                if not args.no_realtime:
                    time.sleep(max(0.0, 1.0 / policy_hz - (time.perf_counter() - started)))
        finally:
            if viewer is not None:
                viewer.close()


if __name__ == "__main__":
    main()
