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

"""Left-right symmetry transforms for the Pi 22/23-DOF policies.

The standard simple-phase policy frame is laid out as::

    ang_vel(3), projected_gravity(3), command(3),
    joint_pos(N), joint_vel(N), previous_action(N), sin_phase, cos_phase

``N`` is resolved from the environment's runtime articulation. A sagittal
reflection swaps left/right joints and negates rotations about the X and Z
axes. For moving commands it also advances the observed gait phase by half a
cycle so the left and right legs exchange gait roles.
"""

from __future__ import annotations
import torch

SUPPORTED_NUM_JOINTS = (22, 23)
OBS_PREFIX_DIM = 9
PHASE_DIM = 2
_PI_JOINT_MIRROR = {
    "l_hip_pitch_joint": ("r_hip_pitch_joint", 1.0),
    "r_hip_pitch_joint": ("l_hip_pitch_joint", 1.0),
    "l_hip_roll_joint": ("r_hip_roll_joint", -1.0),
    "r_hip_roll_joint": ("l_hip_roll_joint", -1.0),
    "l_thigh_joint": ("r_thigh_joint", -1.0),
    "r_thigh_joint": ("l_thigh_joint", -1.0),
    "l_calf_joint": ("r_calf_joint", 1.0),
    "r_calf_joint": ("l_calf_joint", 1.0),
    "l_ankle_pitch_joint": ("r_ankle_pitch_joint", 1.0),
    "r_ankle_pitch_joint": ("l_ankle_pitch_joint", 1.0),
    "l_ankle_roll_joint": ("r_ankle_roll_joint", -1.0),
    "r_ankle_roll_joint": ("l_ankle_roll_joint", -1.0),
    "l_shoulder_pitch_joint": ("r_shoulder_pitch_joint", 1.0),
    "r_shoulder_pitch_joint": ("l_shoulder_pitch_joint", 1.0),
    "l_shoulder_roll_joint": ("r_shoulder_roll_joint", -1.0),
    "r_shoulder_roll_joint": ("l_shoulder_roll_joint", -1.0),
    "l_upper_arm_joint": ("r_upper_arm_joint", -1.0),
    "r_upper_arm_joint": ("l_upper_arm_joint", -1.0),
    "l_elbow_joint": ("r_elbow_joint", 1.0),
    "r_elbow_joint": ("l_elbow_joint", 1.0),
    "waist_yaw_joint": ("waist_yaw_joint", -1.0),
    "head_yaw_joint": ("head_yaw_joint", -1.0),
    "head_pitch_joint": ("head_pitch_joint", 1.0),
}


def _actor_frame_dim(num_joints: int) -> int:
    return OBS_PREFIX_DIM + 3 * num_joints + PHASE_DIM


def _critic_frame_dim(num_joints: int) -> int:
    return _actor_frame_dim(num_joints) + 3 + 2


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _joint_mirror_tensors(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base_env = _unwrap_env(env)
    robot = getattr(base_env, "robot", None)
    if robot is None or not hasattr(robot, "joint_names"):
        raise ValueError("Pi symmetry requires env.robot.joint_names to resolve the runtime joint order.")
    joint_names = list(robot.joint_names)
    if len(joint_names) not in SUPPORTED_NUM_JOINTS:
        raise ValueError(f"Pi symmetry supports {SUPPORTED_NUM_JOINTS} joints, got {len(joint_names)}: {joint_names}")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"Pi runtime joint names contain duplicates: {joint_names}")
    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    perm = []
    sign = []
    missing = []
    for joint_name in joint_names:
        mirror_spec = _PI_JOINT_MIRROR.get(joint_name)
        if mirror_spec is None:
            missing.append(joint_name)
            continue
        mirror_name, mirror_sign = mirror_spec
        if mirror_name not in joint_name_to_idx:
            missing.append(mirror_name)
            continue
        perm.append(joint_name_to_idx[mirror_name])
        sign.append(mirror_sign)
    if missing:
        raise ValueError(f"Pi symmetry has no runtime mirror counterpart for joints: {sorted(set(missing))}")
    return (torch.tensor(perm, dtype=torch.long, device=device), torch.tensor(sign, dtype=torch.float32, device=device))


def _command_thresholds(env) -> tuple[float, float]:
    base_env = _unwrap_env(env)
    gait_cfg = getattr(getattr(base_env, "cfg", None), "gait", None)
    return (
        float(getattr(gait_cfg, "lin_command_threshold", 0.1)),
        float(getattr(gait_cfg, "yaw_command_threshold", 0.1)),
    )


def _mirror_joints(data: torch.Tensor, perm_idx: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Mirror a batch of joint values in the runtime articulation order."""
    return data[:, perm_idx] * sign.to(dtype=data.dtype).unsqueeze(0)


def _mirror_single_actor_obs(
    obs: torch.Tensor,
    perm_idx: torch.Tensor,
    sign: torch.Tensor,
    lin_command_threshold: float,
    yaw_command_threshold: float,
) -> torch.Tensor:
    """Mirror one simple-phase actor-observation frame."""
    num_joints = perm_idx.numel()
    expected_dim = _actor_frame_dim(num_joints)
    if obs.shape[-1] != expected_dim:
        raise ValueError(f"Expected a {expected_dim}-D Pi-{num_joints} actor frame, got {obs.shape[-1]}.")
    mirrored = obs.clone()
    mirrored[:, 0] *= -1.0
    mirrored[:, 2] *= -1.0
    mirrored[:, 4] *= -1.0
    mirrored[:, 7] *= -1.0
    mirrored[:, 8] *= -1.0
    for start in (OBS_PREFIX_DIM, OBS_PREFIX_DIM + num_joints, OBS_PREFIX_DIM + 2 * num_joints):
        mirrored[:, start : start + num_joints] = _mirror_joints(obs[:, start : start + num_joints], perm_idx, sign)
    command = obs[:, 6:9]
    moving = (torch.linalg.vector_norm(command[:, :2], dim=1) > lin_command_threshold) | (
        torch.abs(command[:, 2]) > yaw_command_threshold
    )
    phase_start = OBS_PREFIX_DIM + 3 * num_joints
    mirrored[moving, phase_start : phase_start + PHASE_DIM] *= -1.0
    return mirrored


def _mirror_actor_frames(
    obs: torch.Tensor,
    perm_idx: torch.Tensor,
    sign: torch.Tensor,
    lin_command_threshold: float,
    yaw_command_threshold: float,
) -> torch.Tensor:
    batch_size, total_dim = obs.shape
    frame_dim = _actor_frame_dim(perm_idx.numel())
    if total_dim == 0 or total_dim % frame_dim != 0:
        raise ValueError(f"Pi actor observations must be a positive multiple of {frame_dim}, got {total_dim}.")
    frames = obs.reshape(batch_size, -1, frame_dim)
    mirrored = torch.stack(
        [
            _mirror_single_actor_obs(frames[:, frame_idx], perm_idx, sign, lin_command_threshold, yaw_command_threshold)
            for frame_idx in range(frames.shape[1])
        ],
        dim=1,
    )
    return mirrored.reshape_as(obs)


def _mirror_single_critic_obs(
    obs: torch.Tensor,
    perm_idx: torch.Tensor,
    sign: torch.Tensor,
    lin_command_threshold: float,
    yaw_command_threshold: float,
) -> torch.Tensor:
    """Mirror one critic frame (actor frame, root velocity, contacts)."""
    num_joints = perm_idx.numel()
    actor_dim = _actor_frame_dim(num_joints)
    expected_dim = _critic_frame_dim(num_joints)
    if obs.shape[-1] != expected_dim:
        raise ValueError(f"Expected a {expected_dim}-D Pi-{num_joints} critic frame, got {obs.shape[-1]}.")
    mirrored = obs.clone()
    mirrored[:, :actor_dim] = _mirror_single_actor_obs(
        obs[:, :actor_dim], perm_idx, sign, lin_command_threshold, yaw_command_threshold
    )
    mirrored[:, actor_dim + 1] *= -1.0
    mirrored[:, actor_dim + 3] = obs[:, actor_dim + 4]
    mirrored[:, actor_dim + 4] = obs[:, actor_dim + 3]
    return mirrored


@torch.no_grad()
def compute_symmetric_states(obs=None, actions=None, env=None, obs_type: str = "policy"):
    """Append left-right mirrored samples using the RSL-RL symmetry API."""
    if obs is None and actions is None:
        return (None, None)
    device = obs.device if obs is not None else actions.device
    perm_idx, sign = _joint_mirror_tensors(env, device)
    num_joints = perm_idx.numel()
    lin_command_threshold, yaw_command_threshold = _command_thresholds(env)
    obs_aug = None
    actions_aug = None
    if obs is not None:
        if obs_type == "base_lin_vel":
            if obs.shape[-1] != 3:
                raise ValueError(f"Expected a 3-D base linear velocity, got {obs.shape[-1]}.")
            mirrored = obs.clone()
            mirrored[:, 1] *= -1.0
        elif obs_type == "critic":
            mirrored = _mirror_single_critic_obs(obs, perm_idx, sign, lin_command_threshold, yaw_command_threshold)
        elif obs_type in ("policy", "policy_history"):
            mirrored = _mirror_actor_frames(obs, perm_idx, sign, lin_command_threshold, yaw_command_threshold)
        else:
            mirrored = obs.clone()
        obs_aug = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        if actions.shape[-1] != num_joints:
            raise ValueError(f"Pi-{num_joints} symmetry expected {num_joints} actions, got {actions.shape[-1]}.")
        mirrored_actions = _mirror_joints(actions, perm_idx, sign)
        actions_aug = torch.cat([actions, mirrored_actions], dim=0)
    return (obs_aug, actions_aug)
