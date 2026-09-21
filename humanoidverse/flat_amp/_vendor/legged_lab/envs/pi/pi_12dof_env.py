# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Pi12 独立环境：观测、动作、奖励调用、重置和步态逻辑均在本文件中。

由现有 Pi12 的继承实现展开，只继承 RSL-RL 的 VecEnv 抽象接口。
任务参数和奖励项定义见 pi_12dof_cfg.py；共享奖励函数见 mdp/rewards.py。
"""

from __future__ import annotations

import math
import re

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import EventManager, RewardManager, SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate
from rsl_rl.env import VecEnv

from legged_lab.envs.pi.pi_12dof_cfg import PiDof12WalkFlatEnvCfg
from legged_lab.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from legged_lab.mdp.rewards import command_activity_masks, get_euler_xyz_tensor
from legged_lab.utils.env_utils.scene import SceneCfg


def _tolist(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


class PiDof12Env(VecEnv):
    amp_upper_body = False

    def __init__(self, cfg: PiDof12WalkFlatEnvCfg, headless):
        self.cfg: PiDof12WalkFlatEnvCfg
        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot: Articulation = self.scene["robot"]
        joint_names = list(self.robot.joint_names)
        joint_ids, joint_names_checked = self.robot.find_joints(name_keys=joint_names, preserve_order=True)
        joint_ids = _tolist(joint_ids)
        print("Legged Lab joint names & IDs:")
        for jid, name in zip(joint_ids, joint_names_checked):
            print(f"- ID {jid:3d}: {name}")
        body_names = list(self.robot.body_names)
        print("\nLegged Lab body names & (enum) indices:")
        for bid, name in enumerate(body_names):
            print(f"- ID {bid:3d}: {name}")
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_onestep_envs=self.cfg.commands.rel_onestep_envs,
            rel_turn_only_envs=self.cfg.commands.rel_turn_only_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.init_buffers()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        if self.robot.num_joints != 12:
            raise ValueError(f"Pi12 requires 12 joints, got {self.robot.num_joints}")
        self.extras = {}
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.joint_encoder_bias = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.action_scale = self._resolve_action_scale(self.cfg.robot.action_scale)
        self.action_buffer = DelayBuffer(self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device)
        self.action_buffer.compute(torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False))
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names)
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        self.feet_body_ids, _ = self.robot.find_bodies(name_keys=["l_ankle_roll_link", "r_ankle_roll_link"], preserve_order=True)
        self.torso_body_ids, _ = self.robot.find_bodies(name_keys=["torso_link"], preserve_order=True)
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=["l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint", "l_ankle_pitch_joint"]
            + ([] if self.cfg.amp_exclude_ankle_roll else ["l_ankle_roll_joint"]),
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=["r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint", "r_ankle_pitch_joint"]
            + ([] if self.cfg.amp_exclude_ankle_roll else ["r_ankle_roll_joint"]),
            preserve_order=True,
        )
        print("left_leg_ids:", self.left_leg_ids)
        print("right_leg_ids:", self.right_leg_ids)
        if len(self.torso_body_ids) != 1:
            raise RuntimeError(f"Expected one torso_link for Pi AMP observations, found {len(self.torso_body_ids)}")
        # 根线/角速度 + 左右腿关节位置/速度 + 双脚根坐标系位置，默认共 36 维。
        self.amp_obs_dim = 6 + 2 * (len(self.left_leg_ids) + len(self.right_leg_ids)) + 6
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device) * 0.5
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full((self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()
        self.body_euler = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.is_on_forbidden = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_vel_limited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_reason_buf = {
            "contact_excess": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "bad_orientation": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "bad_euler": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "bad_height": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "time_out": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def _resolve_action_scale(self, action_scale):
        """Expand scalar or regex-keyed action scales to runtime joint order."""
        if isinstance(action_scale, (int, float)):
            return float(action_scale)
        joint_scales = []
        unmatched_joint_names = []
        for joint_name in self.robot.joint_names:
            matched_scale = None
            for joint_pattern, scale in action_scale.items():
                if re.fullmatch(joint_pattern, joint_name):
                    matched_scale = scale
                    break
            if matched_scale is None:
                unmatched_joint_names.append(joint_name)
            else:
                joint_scales.append(matched_scale)
        if unmatched_joint_names:
            raise ValueError(f"Missing action_scale entries for Pi joints: {unmatched_joint_names}")
        return torch.tensor(joint_scales, dtype=torch.float, device=self.device).unsqueeze(0)

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                torch.sin(2 * torch.pi * self.phase).unsqueeze(1),
                torch.cos(2 * torch.pi * self.phase).unsqueeze(1),
            ],
            dim=-1,
        )
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)
        return (current_actor_obs, current_critic_obs)

    def reset(self, env_ids):
        if getattr(self, "_capture_amp_transition", False):
            # 在自动重置改写物理状态之前，保存本次终止转移供 AMP/PPO 使用。
            # 空 env_ids 也要更新为空张量，避免沿用上一控制步的终止状态。
            self.extras["terminal_amp_states"] = self.get_amp_obs_for_expert_trans()[env_ids].clone()
            self.extras["terminal_actor_obs"] = self.compute_current_observations()[0][env_ids].clone()
        if len(env_ids) == 0:
            return
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0
        self.extras["log"] = dict()
        for reason_name, reason_buf in self.reset_reason_buf.items():
            reason_count = torch.count_nonzero(reason_buf[env_ids]).float()
            self.extras["log"][f"Termination/{reason_name}_count"] = reason_count
            self.extras["log"][f"Termination/{reason_name}_rate"] = reason_count / len(env_ids)
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf
        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device) * 0.5
        self.body_euler = get_euler_xyz_tensor(self.robot.data.body_quat_w[:, self.torso_body_ids[0], :])
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        self._capture_amp_transition = True
        try:
            delayed_actions = self.action_buffer.compute(actions)
            self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
            processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos - self.joint_encoder_bias
            self.avg_feet_force_per_step = torch.zeros(
                self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
            )
            self.avg_feet_speed_per_step = torch.zeros(
                self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
            )
            for _ in range(self.cfg.sim.decimation):
                self.sim_step_counter += 1
                self.robot.set_joint_position_target(processed_actions)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
                self.avg_feet_force_per_step += torch.norm(self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1)
                self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)
            self.avg_feet_force_per_step /= self.cfg.sim.decimation
            self.avg_feet_speed_per_step /= self.cfg.sim.decimation
            self.body_euler = get_euler_xyz_tensor(self.robot.data.body_quat_w[:, self.torso_body_ids[0], :])
            if not self.headless:
                self.sim.render()
            self.episode_length_buf += 1
            self.phase_length_buf += 1
            self._calculate_gait_para()
            self.command_generator.compute(self.step_dt)
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
            self.reset_buf, self.time_out_buf = self.check_reset()
            reward_buf = self.reward_manager.compute(self.step_dt)
            play_window = getattr(self, "_play_window", None)
            if play_window is not None:
                play_window.update(actions, processed_actions)
            self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.reset(self.reset_env_ids)
            actor_obs, critic_obs, current_actor_obs_add_nosie, current_actor_obs, current_critic_obs = self.compute_observations()
            lin_vel_index = (self.cfg.robot.critic_obs_history_length - 1) * current_critic_obs.shape[1] + current_actor_obs.shape[1]
            self.extras["observations"] = {
                "critic": critic_obs,
                "obs_history": actor_obs,
                "current_actor_obs_add_nosie": current_actor_obs_add_nosie,
                "current_actor_obs": current_actor_obs,
                "current_critic_obs": current_critic_obs,
                "lin_vel_index": lin_vel_index,
            }
            return (actor_obs, reward_buf, self.reset_buf, self.extras)
        finally:
            self._capture_amp_transition = False

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        contact_excess = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1), dim=1)[0] > 10.0,
            dim=1,
        )
        limit_angle = self.cfg.robot.max_tilt_angle_rad
        if limit_angle is None:
            limit_angle = math.radians(self.cfg.robot.max_tilt_angle_deg)
        tilt = torch.acos(-self.robot.data.projected_gravity_b[:, 2]).abs()
        bad_orientation = tilt > limit_angle
        bad_euler = torch.logical_or(
            torch.abs(self.body_euler[:, 1]) > self.cfg.robot.max_body_pitch_angle_rad,
            torch.abs(self.body_euler[:, 2]) > self.cfg.robot.max_body_yaw_angle_rad,
        )
        minimum_height = 0.33
        terrain_base_height = torch.clamp(self.scene.env_origins[:, 2], max=0.0)
        bad_height = self.is_on_forbidden & (self.robot.data.root_pos_w[:, 2] - terrain_base_height < minimum_height)
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_reason_buf["contact_excess"].copy_(contact_excess)
        self.reset_reason_buf["bad_orientation"].copy_(bad_orientation)
        self.reset_reason_buf["bad_euler"].copy_(bad_euler)
        self.reset_reason_buf["bad_height"].copy_(bad_height)
        self.reset_reason_buf["time_out"].copy_(time_out_buf)
        reset_buf = contact_excess | bad_orientation | time_out_buf | bad_euler | bad_height
        return (reset_buf, time_out_buf)

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = noise_scales.joint_vel * self.obs_scales.joint_vel
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def get_observations(self):
        actor_obs, critic_obs, current_actor_obs_add_nosie, current_actor_obs, current_critic_obs = self.compute_observations()
        lin_vel_index = (self.cfg.robot.critic_obs_history_length - 1) * current_critic_obs.shape[1] + current_actor_obs.shape[1]
        self.extras["observations"] = {
            "critic": critic_obs,
            "obs_history": actor_obs,
            "current_actor_obs_add_nosie": current_actor_obs_add_nosie,
            "current_actor_obs": current_actor_obs,
            "current_critic_obs": current_critic_obs,
            "lin_vel_index": lin_vel_index,
        }
        return (actor_obs, self.extras)

    def get_amp_obs_for_expert_trans(self):
        """Build policy AMP observations using the same layout as the expert loader."""
        root_pos = self.robot.data.root_state_w[:, :3]
        root_quat_inv = quat_conjugate(self.robot.data.root_state_w[:, 3:7])
        left_foot_pos = quat_apply(root_quat_inv, self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - root_pos)
        right_foot_pos = quat_apply(root_quat_inv, self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - root_pos)
        joint_ids = list(self.left_leg_ids) + list(self.right_leg_ids)
        return torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.joint_pos[:, joint_ids],
                self.robot.data.joint_vel[:, joint_ids],
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        _, _, movement_active = command_activity_masks(
            self.command_generator.command,
            lin_command_threshold=self.cfg.gait.lin_command_threshold,
            yaw_command_threshold=self.cfg.gait.yaw_command_threshold,
        )
        stand_command = ~movement_active
        self.phase_length_buf[stand_command] = 0.0
        self.phase = self.phase_length_buf * self.step_dt % self.cfg.gait.gait_cycle / self.cfg.gait.gait_cycle
        self.phase = (self.phase + self.gait_start) % 1.0
        self.phase *= ~stand_command
        offset = 0.5
        self.gait_phase[:, 0] = self.phase[:] % 1.0
        self.gait_phase[:, 1] = (self.phase[:] + offset) % 1.0
        self.gait_phase[stand_command, :] = 0.0

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        noisy = current_actor_obs.clone()
        if self.add_noise:
            noisy += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec
        self.actor_obs_buffer.append(noisy)
        self.critic_obs_buffer.append(current_critic_obs)
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        return (
            torch.clip(actor_obs, -self.clip_obs, self.clip_obs),
            torch.clip(critic_obs, -self.clip_obs, self.clip_obs),
            noisy,
            current_actor_obs,
            current_critic_obs,
        )
