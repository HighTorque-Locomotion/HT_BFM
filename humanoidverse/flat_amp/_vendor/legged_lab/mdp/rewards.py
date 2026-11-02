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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_lab.envs.pi.pi_12dof_env import PiDof12Env
import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from humanoidverse.flat_amp.foot_height import FootHeightShortfallEvents, SoleClearance


class SwingHeightShortfallPenalty(ManagerTermBase):
    """Collect substep shortfalls; charge once per foot/cycle without dt scaling.

    Adapted from HT_lab's SwingHeightShortfallPenalty, using Pi12's shared
    gait clock and command generator. Only enabled terms register the hooks.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.events = FootHeightShortfallEvents(
            env.num_envs, env.device, phase_window=cfg.params["phase_window"], minimum_height=cfg.params["minimum_height"]
        )
        self.geometry = SoleClearance.from_stage(env.scene.stage, env.device)
        self.feet_ids, _ = env.robot.find_bodies(["l_ankle_roll_link", "r_ankle_roll_link"], preserve_order=True)
        self.lin_threshold = cfg.params["lin_command_threshold"]
        self.yaw_threshold = cfg.params["yaw_command_threshold"]
        self.step_events = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.last_events = torch.zeros_like(self.step_events)
        self.last_clearance = torch.zeros(env.num_envs, 2, device=env.device)
        self.previous_moving = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.offsets = torch.tensor([0.0, 0.5], device=env.device)
        env.swing_height_shortfall_penalty = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.events.reset(env_ids)
        self.step_events[ids] = False
        self.last_events[ids] = False
        self.last_clearance[ids] = 0
        self.previous_moving[ids] = -1

    def begin_step(self, env):
        self.step_events.zero_()
        _, _, self.moving = command_activity_masks(env.command_generator.command, self.lin_threshold, self.yaw_threshold)
        self.events.reset(self.previous_moving != self.moving.long())
        self.previous_moving.copy_(self.moving)
        # Derive from the live clock, since gait_phase can be stale after reset
        # or a command change. Match _calculate_gait_para at the last substep.
        self.start_phase = env.phase_length_buf * env.step_dt / env.cfg.gait.gait_cycle + env.gait_start

    def record_substep(self, env, substep):
        data = env.robot.data
        self.last_clearance.copy_(
            self.geometry(data.body_pos_w[:, self.feet_ids], data.body_quat_w[:, self.feet_ids], env.scene.env_origins[:, 2])
        )
        phase = (
            self.start_phase[:, None] + self.offsets + (substep + 1) * env.physics_dt / env.cfg.gait.gait_cycle
        ) % 1.0
        phase = torch.where(self.moving[:, None], phase, 0.0)
        self.step_events |= self.events.update(self.last_clearance, phase, None, self.moving)

    def __call__(self, env, phase_window, minimum_height, lin_command_threshold, yaw_command_threshold):
        self.last_events.copy_(self.step_events)
        self.step_events.zero_()
        # RewardManager multiplies by dt: preserve an actual -1 event at weight -1.
        return self.last_events.sum(dim=-1).float() / env.step_dt


def command_activity_masks(
    command: torch.Tensor, lin_command_threshold: float = 0.1, yaw_command_threshold: float | None = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify planar velocity commands without mixing linear and angular units.

    Returns ``(lin_active, yaw_active, movement_active)``. A command is truly
    standing only when both the linear and yaw components are inside their
    respective deadbands.
    """
    lin_active = torch.linalg.vector_norm(command[:, :2], dim=1) >= lin_command_threshold
    if yaw_command_threshold is None:
        yaw_active = torch.zeros_like(lin_active)
    else:
        yaw_active = torch.abs(command[:, 2]) >= yaw_command_threshold
    return (lin_active, yaw_active, lin_active | yaw_active)


def straight_command_mask(
    command: torch.Tensor, lin_command_threshold: float = 0.1, yaw_command_threshold: float = 0.1
) -> torch.Tensor:
    """Select forward/backward commands with lateral and yaw commands inside their deadbands."""
    return (
        (command[:, 0].abs() >= lin_command_threshold)
        & (command[:, 1].abs() < lin_command_threshold)
        & (command[:, 2].abs() < yaw_command_threshold)
    )


def idle_when_commanded(
    env: PiDof12Env,
    lin_command_threshold: float = 0.1,
    yaw_command_threshold: float = 0.1,
    lin_velocity_threshold: float = 0.1,
    yaw_velocity_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize failure to move in each commanded planar velocity mode.

    Linear and yaw motion are evaluated independently. In particular, a robot
    that remains still under a pure-yaw command is no longer treated as idle-free.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_generator.command
    lin_command_active, yaw_command_active, _ = command_activity_masks(
        command, lin_command_threshold=lin_command_threshold, yaw_command_threshold=yaw_command_threshold
    )
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_velocity_idle = torch.linalg.vector_norm(vel_yaw[:, :2], dim=1) < lin_velocity_threshold
    yaw_velocity_idle = torch.abs(asset.data.root_ang_vel_w[:, 2]) < yaw_velocity_threshold
    lin_idle_when_commanded = lin_command_active & lin_velocity_idle
    yaw_idle_when_commanded = yaw_command_active & yaw_velocity_idle
    return (lin_idle_when_commanded | yaw_idle_when_commanded).float()


def body_orientation_euler(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    body_euler = get_euler_xyz_tensor(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :])
    # Correct WXYZ conversion returns roll, pitch, yaw; heading must remain free.
    quat_mismatch = torch.exp(-torch.sum(torch.abs(body_euler[:, :2]), dim=1) * 10)
    orientation = torch.exp(-torch.norm(body_orientation[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.0


def track_command_tilt_exp(
    env: PiDof12Env,
    std: float,
    tolerance: float = 0.0,
    gravity: float = 9.81,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward torso roll/pitch around the commanded steady-turn reference.

    Commands use horizontal heading axes (x forward, y left, z up). No command
    derivatives are taken: ax = -wz*vy and ay = wz*vx. Align the torso z axis
    with (ax, ay, gravity) to obtain a soft roll/pitch reference; yaw is free.

    Each axis has an optional flat tolerance band (disabled by default). Outside it, the reward is
    exp(-0.5 * sum((excess_error / std)**2)). Both angles are in radians;
    std is the Gaussian standard deviation, unlike the velocity rewards'
    exp(-error/std**2) convention. This term is stateless, including on reset.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity must be finite and positive")
    command = env.command_generator.command
    ax = -command[:, 2] * command[:, 1]
    ay = command[:, 2] * command[:, 0]
    pitch_ref = torch.atan2(ax, torch.full_like(ax, gravity))
    roll_ref = torch.atan2(-ay, torch.sqrt(gravity**2 + ax.square()))
    target = torch.stack((roll_ref, pitch_ref), dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch = get_euler_xyz_tensor(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :])[:, :2]
    difference = roll_pitch - target
    error = torch.atan2(torch.sin(difference), torch.cos(difference)).abs()
    excess = torch.clamp(error - tolerance, min=0.0)
    return torch.exp(-0.5 * torch.sum(torch.square(excess / std), dim=-1))


def base_com_feet_midpoint_exp(
    env: PiDof12Env,
    std: float,
    base_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    foot_x_offset: float = -0.03,
    tolerance: float = 0.005,
) -> torch.Tensor:
    """Align base COM with offset foot midpoint along the robot heading x axis.

    Shift each ankle-roll link origin along its local x axis before averaging,
    including the swing foot regardless of contact or phase. Project the error
    onto the root yaw frame's x axis; lateral and vertical errors are ignored.
    The base position is its actual COM, not its
    link origin or the whole-robot COM. foot_x_offset is in meters.
    Return exp(-(max(abs(error_x) - tolerance, 0) / std)**2), with lengths in meters.
    With tolerance=0.005 and std=0.03, errors up to 5 mm score 1; 35 mm scores exp(-1).
    This width is sqrt(2) times the conventional Gaussian standard deviation.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and positive")
    if not math.isfinite(foot_x_offset):
        raise ValueError("foot_x_offset must be finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    base: Articulation = env.scene[base_cfg.name]
    feet: Articulation = env.scene[feet_cfg.name]
    base_xy = base.data.body_com_pos_w[:, base_cfg.body_ids, :2]
    feet_xy = feet.data.body_link_pos_w[:, feet_cfg.body_ids, :2]
    if base_xy.shape[1] != 1 or feet_xy.shape[1] != 2:
        raise ValueError("Expected exactly one base body and two foot bodies")
    # First column of the foot rotation matrix, restricted to world XY (WXYZ).
    w, x, y, z = feet.data.body_link_quat_w[:, feet_cfg.body_ids].unbind(dim=-1)
    foot_x_axis_xy = torch.stack((1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1)
    midpoint_xy = (feet_xy + foot_x_offset * foot_x_axis_xy).mean(dim=1)
    error_xy = base_xy[:, 0] - midpoint_xy
    yaw = _yaw_from_quat_wxyz(base.data.root_quat_w)
    error_x = error_xy[:, 0] * torch.cos(yaw) + error_xy[:, 1] * torch.sin(yaw)
    excess = torch.clamp(error_x.abs() - tolerance, min=0.0)
    return torch.exp(-excess.square() / std**2)


def stand_still_joint_exp(
    env: PiDof12Env,
    lin_command_threshold: float = 0.06,
    yaw_command_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    if command_threshold is not None:
        lin_command_threshold = command_threshold
    command = env.command_generator.command
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, movement_active = command_activity_masks(
        command, lin_command_threshold=lin_command_threshold, yaw_command_threshold=yaw_command_threshold
    )
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * (~movement_active).float()


def track_lin_vel_xy_yaw_frame_exp(
    env: PiDof12Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: PiDof12Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def root_heading_yaw_l2(
    env: PiDof12Env,
    target_yaw: float = 0.0,
    only_vel_limited: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_asset_cfg: SceneEntityCfg | None = None,
    root_weight: float = 1.0,
    body_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize root and body yaw deviation from a fixed world-frame heading."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)
    yaw_error = torch.atan2(torch.sin(yaw - target_yaw), torch.cos(yaw - target_yaw))
    penalty = root_weight * torch.square(yaw_error)
    if body_asset_cfg is not None:
        body_asset: Articulation = env.scene[body_asset_cfg.name]
        body_yaw = _yaw_from_quat_wxyz(body_asset.data.body_quat_w[:, body_asset_cfg.body_ids, :])
        body_yaw_error = body_yaw - target_yaw
        body_yaw_error = torch.atan2(torch.sin(body_yaw_error), torch.cos(body_yaw_error))
        body_penalty = torch.square(body_yaw_error)
        if body_penalty.ndim > 1:
            body_penalty = torch.mean(body_penalty, dim=-1)
        penalty = penalty + body_weight * body_penalty
    if only_vel_limited:
        vel_limited_mask = getattr(env, "is_vel_limited", torch.zeros_like(penalty, dtype=torch.bool))
        penalty = penalty * vel_limited_mask.to(device=penalty.device, dtype=penalty.dtype)
    return penalty


def lin_vel_z_l2(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def ang_vel_xy_l2_body(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    )
    return torch.sum(torch.square(body_ang_vel_b[:, :2]), dim=1)


def energy(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: PiDof12Env) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def action_smoothness(env: PiDof12Env) -> torch.Tensor:
    buf = env.action_buffer._circular_buffer.buffer
    a_t = buf[:, -1, :]
    a_t1 = buf[:, -2, :]
    a_t2 = buf[:, -3, :]
    term_1 = torch.sum((a_t - a_t1) ** 2, dim=1)
    term_2 = torch.sum((a_t + a_t2 - 2 * a_t1) ** 2, dim=1)
    term_3 = 0.05 * torch.sum(torch.abs(a_t), dim=1)
    return term_1 + term_2 + term_3


def undesired_contacts(env: PiDof12Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(
    env: PiDof12Env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    lin_command_threshold: float | None = None,
    yaw_command_threshold: float | None = None,
    max_lin_command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize flight only in command modes where flight is undesired.

    A brief no-foot-contact interval is normal for running.  The optional
    command gates let a task retain the historical walking ``fly`` penalty
    while disabling it for the run command regime.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    no_foot_contact = torch.sum(is_contact, dim=-1) < 0.5
    if max_lin_command_threshold is None:
        command_active = torch.ones_like(no_foot_contact, dtype=torch.bool)
    else:
        _, _, command_active = command_activity_masks(
            env.command_generator.command,
            lin_command_threshold=0.1 if lin_command_threshold is None else lin_command_threshold,
            yaw_command_threshold=yaw_command_threshold,
        )
        planar_command_speed = torch.linalg.vector_norm(env.command_generator.command[:, :2], dim=1)
        command_active &= planar_command_speed <= max_lin_command_threshold
    return no_foot_contact * command_active.float() * (~env.is_on_forbidden).float()


def flat_orientation_l2(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: PiDof12Env) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_slide(
    env: PiDof12Env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize foot planar speed and world-z angular speed during contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    body_ang_vel_z = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    reward += torch.sum(body_ang_vel_z.abs() * contacts, dim=1)
    return reward


def feet_x_offset_lateral(
    env: PiDof12Env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_command_threshold: float = 0.1,
    forward_command_threshold: float = 0.1,
    yaw_command_threshold: float = 0.1,
    contact_threshold: float = 1.0,
    x_deadband: float = 0.03,
) -> torch.Tensor:
    """Penalize fore-aft foot offset during pure lateral locomotion.

    The command gate only activates for a meaningful ``y`` command while the
    forward and yaw commands remain inside their deadbands.  Foot positions are
    expressed relative to the root in the yaw frame, so the penalty follows
    the robot heading instead of the world axes.  It is evaluated only when
    both feet are in contact, allowing the swing foot to reposition freely.
    """
    command = env.command_generator.command
    pure_lateral = (
        (torch.abs(command[:, 1]) >= lateral_command_threshold)
        & (torch.abs(command[:, 0]) < forward_command_threshold)
        & (torch.abs(command[:, 2]) < yaw_command_threshold)
    )
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_x_offset_lateral expects exactly two foot bodies")
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_pos_rel_w = foot_pos_w - asset.data.root_pos_w[:, None, :]
    root_yaw = math_utils.yaw_quat(asset.data.root_quat_w).unsqueeze(1)
    root_yaw = root_yaw.expand(-1, foot_pos_rel_w.shape[1], -1)
    foot_pos_yaw = math_utils.quat_apply_inverse(root_yaw, foot_pos_rel_w)
    x_offset = torch.abs(foot_pos_yaw[:, 0, 0] - foot_pos_yaw[:, 1, 0])
    offset_penalty = torch.square(torch.relu(x_offset - x_deadband))
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3].norm(dim=-1)
    both_contact = (contact_force > contact_threshold).all(dim=1)
    return offset_penalty * both_contact.float() * pure_lateral.float() * (~env.is_on_forbidden).float()


def body_force(
    env: PiDof12Env, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(
    env: PiDof12Env,
    lin_command_threshold: float = 0.1,
    yaw_command_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    _, _, movement_active = command_activity_masks(
        env.command_generator.command,
        lin_command_threshold=lin_command_threshold,
        yaw_command_threshold=yaw_command_threshold,
    )
    return torch.sum(torch.abs(angle), dim=1) * (~movement_active).float()


def joint_deviation_l2(
    env: PiDof12Env,
    lin_command_threshold: float = 0.1,
    yaw_command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint offsets only during straight forward/backward locomotion."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_generator.command
    straight = straight_command_mask(
        command,
        lin_command_threshold=lin_command_threshold,
        yaw_command_threshold=yaw_command_threshold,
    )
    return torch.sum(torch.square(angle), dim=1) * straight.float()


def feet_stumble(env: PiDof12Env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_y_distance(
    env: PiDof12Env, lin_command_threshold: float = 0.1, yaw_command_threshold: float = 0.1
) -> torch.Tensor:
    """Penalize deviation from 0.16 m foot spacing only when standing or moving straight forward/backward."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.16)
    command = env.command_generator.command
    straight = straight_command_mask(command, lin_command_threshold, yaw_command_threshold)
    _, _, movement_active = command_activity_masks(command, lin_command_threshold, yaw_command_threshold)
    return y_distance_b * (straight | ~movement_active).float()


def _gauss_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def stance_leg_pitch_sum_exp(
    env: PiDof12Env,
    std: float,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward zero hip-pitch + calf + ankle-pitch sum in each foot's stance.

    Use actual joint angles in radians and exp(-0.5 * (error / std)**2).
    Each leg is gated by its own local phase in [0.5, 1); sum both rewards.
    Standing commands set both phases to zero and therefore receive no reward.
    Joint IDs are resolved by name separately for each leg by the reward manager.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and positive")
    left_asset: Articulation = env.scene[left_asset_cfg.name]
    right_asset: Articulation = env.scene[right_asset_cfg.name]
    left_error = left_asset.data.joint_pos[:, left_asset_cfg.joint_ids].sum(dim=-1)
    right_error = right_asset.data.joint_pos[:, right_asset_cfg.joint_ids].sum(dim=-1)
    error = torch.stack((left_error, right_error), dim=-1)
    in_stance = (env.gait_phase >= 0.5) & (env.gait_phase < 1.0)
    return (torch.exp(-0.5 * (error / std).square()) * in_stance).sum(dim=-1)


def gait_clock_smooth(phase: torch.Tensor, air_ratio: torch.Tensor, delta_t: float):
    """
    平滑版步态时钟：
      - I_frc: swing 权重（~1 在 swing）
      - I_spd: stance 权重（= 1 - I_frc）
    使用高斯 CDF 构造"软矩形"，并做周期扩展保证相位 0/1 处连续。

    参数
    ----
    phase:     [N]，相位∈[0,1]
    air_ratio: [N] 或标量，swing 占比（0..1）
    delta_t:   平滑强度（作为 σ 使用；越大越平滑，越小越接近硬切换）

    返回
    ----
    I_frc, I_spd: [N]，范围约 [0,1]
    """
    if not torch.is_tensor(air_ratio):
        air_ratio = torch.tensor(air_ratio, device=phase.device, dtype=phase.dtype)
    sigma = torch.as_tensor(delta_t, device=phase.device, dtype=phase.dtype).clamp(min=1e-06)
    start = torch.zeros_like(phase)
    end = torch.clamp(air_ratio, 1e-06, 1.0 - 1e-06)

    def win(phi):
        return _gauss_cdf((phi - start) / sigma) - _gauss_cdf((phi - end) / sigma)

    I_swing = win(phase) + win(phase - 1.0) + win(phase + 1.0)
    I_swing = I_swing.clamp(0.0, 1.0)
    I_frc = I_swing
    I_spd = 1.0 - I_frc
    return (I_frc, I_spd)


def gait_feet_frc_perio(
    env: PiDof12Env,
    delta_t: float = 0.02,
    lin_command_threshold: float | None = None,
    yaw_command_threshold: float | None = None,
    max_lin_command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * torch.exp(-50 * torch.square(env.avg_feet_force_per_step[:, 0]))
    right_frc_score = right_frc_swing_mask * torch.exp(-50 * torch.square(env.avg_feet_force_per_step[:, 1]))
    if lin_command_threshold is None and yaw_command_threshold is None:
        movement_active = torch.ones_like(left_frc_score, dtype=torch.bool)
    else:
        _, _, movement_active = command_activity_masks(
            env.command_generator.command,
            lin_command_threshold=0.1 if lin_command_threshold is None else lin_command_threshold,
            yaw_command_threshold=yaw_command_threshold,
        )
    if max_lin_command_threshold is not None:
        planar_command_speed = torch.linalg.vector_norm(env.command_generator.command[:, :2], dim=1)
        movement_active &= planar_command_speed <= max_lin_command_threshold
    return (left_frc_score + right_frc_score) * movement_active.float() * (~env.is_on_forbidden).float()


# Original continuous, smoothed implementation retained for reference.
# def gait_feet_frc_perio_penalize(
#     env: PiDof12Env,
#     delta_t: float = 0.02,
#     lin_command_threshold: float | None = None,
#     yaw_command_threshold: float | None = None,
#     max_lin_command_threshold: float | None = None,
# ) -> torch.Tensor:
#     """Penalize foot force during the swing phase of the gait."""
#     left_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
#     right_frc_swing_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
#     left_force = env.avg_feet_force_per_step[:, 0]
#     right_force = env.avg_feet_force_per_step[:, 1]
#     left_frc_score = left_frc_swing_mask * (torch.abs(left_force) > 5.0).float()
#     right_frc_score = right_frc_swing_mask * (torch.abs(right_force) > 5.0).float()
#     if lin_command_threshold is None and yaw_command_threshold is None:
#         movement_active = torch.ones_like(left_frc_score, dtype=torch.bool)
#     else:
#         _, _, movement_active = command_activity_masks(
#             env.command_generator.command,
#             lin_command_threshold=0.1 if lin_command_threshold is None else lin_command_threshold,
#             yaw_command_threshold=yaw_command_threshold,
#         )
#     if max_lin_command_threshold is not None:
#         planar_command_speed = torch.linalg.vector_norm(env.command_generator.command[:, :2], dim=1)
#         movement_active &= planar_command_speed <= max_lin_command_threshold
#     return (left_frc_score + right_frc_score) * movement_active.float() * (~env.is_on_forbidden).float()


def gait_feet_frc_perio_penalize(
    env: PiDof12Env,
    delta_t: float = 0.02,
    lin_command_threshold: float | None = 0.1,
    yaw_command_threshold: float | None = 0.1,
    max_lin_command_threshold: float | None = None,
    phase_start: float = 0.10,
    phase_end: float = 0.40,
) -> torch.Tensor:
    """Charge once per foot/swing for >5 N contact inside the local phase window.

    With weight=-1, division by step_dt cancels RewardManager's dt scaling,
    giving a discrete -1 task-reward contribution per event. delta_t remains
    accepted for old configs but no longer smooths the phase boundaries.
    """
    if not 0.0 <= phase_start < phase_end <= 0.5:
        raise ValueError("Expected 0 <= phase_start < phase_end <= 0.5")
    phase = env.gait_phase
    # Rearm in stance or on wrap/restart; never rearm just because contact ends.
    new_swing = phase < env.swing_contact_previous_phase
    in_stance = phase >= env.phase_ratio
    env.swing_contact_penalized &= ~(new_swing | in_stance)
    env.swing_contact_previous_phase.copy_(phase)

    _, _, movement_active = command_activity_masks(
        env.command_generator.command,
        lin_command_threshold=0.1 if lin_command_threshold is None else lin_command_threshold,
        yaw_command_threshold=yaw_command_threshold,
    )
    if max_lin_command_threshold is not None:
        planar_command_speed = torch.linalg.vector_norm(env.command_generator.command[:, :2], dim=1)
        movement_active &= planar_command_speed <= max_lin_command_threshold
    in_window = (phase >= phase_start) & (phase <= phase_end) & ~in_stance
    triggered = (
        in_window
        & (env.avg_feet_force_per_step > 5.0)
        & movement_active[:, None]
        & ~env.swing_contact_penalized
    )
    env.swing_contact_penalized |= triggered
    return triggered.sum(dim=1).to(phase.dtype) / env.step_dt


def gait_feet_spd_perio(
    env: PiDof12Env,
    delta_t: float = 0.02,
    lin_command_threshold: float | None = None,
    yaw_command_threshold: float | None = None,
    max_lin_command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock_smooth(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock_smooth(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * torch.exp(-50 * torch.square(env.avg_feet_speed_per_step[:, 0]))
    right_spd_score = right_spd_support_mask * torch.exp(-50 * torch.square(env.avg_feet_speed_per_step[:, 1]))
    if lin_command_threshold is None and yaw_command_threshold is None:
        movement_active = torch.ones_like(left_spd_score, dtype=torch.bool)
    else:
        _, _, movement_active = command_activity_masks(
            env.command_generator.command,
            lin_command_threshold=0.1 if lin_command_threshold is None else lin_command_threshold,
            yaw_command_threshold=yaw_command_threshold,
        )
    if max_lin_command_threshold is not None:
        planar_command_speed = torch.linalg.vector_norm(env.command_generator.command[:, :2], dim=1)
        movement_active &= planar_command_speed <= max_lin_command_threshold
    return (left_spd_score + right_spd_score) * movement_active.float() * (~env.is_on_forbidden).float()










def mjlab_feet_gait(
    env: PiDof12Env,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str | None,
    sensor_cfg: SceneEntityCfg,
    yaw_command_threshold: float | None = None,
    max_lin_command_threshold: float | None = None,
) -> torch.Tensor:
    """Reward agreement between gait-clock stance phase and foot contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    use_env_gait_phase = hasattr(env, "gait_phase") and env.gait_phase.shape[1] >= len(sensor_cfg.body_ids)
    if use_env_gait_phase:
        leg_phase = env.gait_phase[:, : len(sensor_cfg.body_ids)]
    else:
        global_phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
        offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
        leg_phase = (global_phase + offsets) % 1.0
    leg_phase = leg_phase[:, : is_contact.shape[1]]
    is_stance = leg_phase >= threshold if use_env_gait_phase else leg_phase < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)
    if hasattr(env, "command_generator"):
        command = env.command_generator.command
    elif command_name is not None and hasattr(env, "command_manager"):
        command = env.command_manager.get_command(command_name)
    else:
        raise AttributeError("mjlab_feet_gait requires env.command_generator or env.command_manager.")
    if yaw_command_threshold is None:
        yaw_command_threshold = command_threshold
    _, _, movement_active = command_activity_masks(
        command, lin_command_threshold=command_threshold, yaw_command_threshold=yaw_command_threshold
    )
    if max_lin_command_threshold is not None:
        planar_command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
        movement_active &= planar_command_speed <= max_lin_command_threshold
    return reward * movement_active.float()


def get_euler_xyz_tensor(quat):
    """Convert Isaac Lab WXYZ quaternions to signed roll, pitch, yaw in radians."""
    r, p, w = get_euler_rpy(quat)
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
    return euler_xyz


def get_euler_rpy(q):
    """Convert WXYZ quaternions to roll, pitch, yaw in [0, 2*pi)."""
    qw, qx, qy, qz = (0, 1, 2, 3)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(torch.pi / 2.0, sinp), torch.asin(sinp))
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return (roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi))


def copysign_new(a, b):
    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)


def joint_torque_l2(env: PiDof12Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
