from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

from humanoidverse.visualize_motion import RobotName, get_robot_spec, _load_motion, _motion_to_qpos


def main(
    data_path: Path | None = None,
    robot: RobotName = "g1",
    motion: int | str = 0,
    headless: bool = False,
    enable_cameras: bool = False,
    max_frames: int | None = 500,
    stride: int = 1,
    device: str = "cpu",
    realtime: bool = True,
    fps: int | None = None,
) -> None:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    robot_spec = get_robot_spec(robot)
    if data_path is None:
        data_path = robot_spec.default_data_path
    key, motion_data, num_motions = _load_motion(data_path.resolve(), motion)
    qpos = _motion_to_qpos(motion_data, robot_spec.dof_size)
    if max_frames is not None:
        qpos = qpos[:max_frames]
    motion_fps = int(motion_data.get("fps", 30) if fps is None else fps)

    env_cfg = HumanoidVerseIsaacConfig(
        lafan_tail_path=str(data_path.resolve()),
        device=device,
        enable_cameras=enable_cameras,
        include_last_action=False,
        include_history_actor=False,
        include_history_noaction=False,
        disable_domain_randomization=True,
        disable_obs_noise=True,
        hydra_overrides=[
            "simulator=isaacsim",
            f"robot={robot_spec.hydra_robot}",
            "env.config.max_episode_length_s=10000",
            f"env.config.headless={headless}",
        ],
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    root_pos = torch.tensor(qpos[:, :3], dtype=torch.float32)
    root_quat = torch.tensor(qpos[:, 3:7], dtype=torch.float32)
    root_vel = torch.zeros((qpos.shape[0], 6), dtype=torch.float32)
    dof_pos = torch.tensor(qpos[:, 7:], dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)

    target_root = torch.zeros((1, 13), dtype=torch.float32, device=env.device)
    target_dof = torch.zeros((1, env.num_dof, 2), dtype=torch.float32, device=env.device)

    print(f"Loaded {data_path}")
    print(f"Robot: {robot_spec.name} / Hydra robot: {robot_spec.hydra_robot}")
    print(f"Motion {motion!r}: {key} ({num_motions} motions in file)")
    print(f"Frames: {len(qpos)} / fps: {motion_fps} / stride: {stride}")

    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    frame_dt = stride / motion_fps

    for i, frame in enumerate(range(0, len(qpos), stride)):
        start_time = time.time()
        target_root[0, :3] = root_pos[frame].to(env.device)
        target_root[0, 3:7] = root_quat[frame].to(env.device)
        target_root[0, 7:13] = root_vel[frame].to(env.device)
        target_dof[0, :, 0] = dof_pos[frame].to(env.device)
        target_dof[0, :, 1] = dof_vel[frame].to(env.device)

        env.simulator.set_actor_root_state_tensor(env_ids, target_root)
        env.simulator.set_dof_state_tensor(env_ids, target_dof)
        env.simulator.scene.write_data_to_sim()
        env.simulator.sim.render()
        env.simulator.scene.update(dt=env.sim_dt)
        if i == 0 and headless:
            print("Isaac Sim headless mode: state applied, no GUI render loop.")
        if realtime:
            sleep_time = frame_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
