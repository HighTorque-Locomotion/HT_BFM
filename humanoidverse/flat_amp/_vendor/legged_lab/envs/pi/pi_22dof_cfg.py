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

import copy
from pathlib import Path
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.pi import PI_PLUS_22DOF_CFG, PI_PLUS_22DOF_ACTION_SCALE
from legged_lab.envs.pi.pi_cfg import (
    PI_LIN_COMMAND_THRESHOLD,
    PI_YAW_COMMAND_THRESHOLD,
    PiDof23RewardCfg,
    PiDof23WalkAgentCfg,
    PiDof23WalkFlatEnvCfg,
)

PI_22DOF_SIMPLE_AMP_MOTION_DIR = str(
    Path(LEGGED_LAB_ROOT_DIR) / "legged_lab/envs/pi/datasets/motion_amp_expert/pi_22_amp"
)


def _make_pi_22dof_scene_cfg():
    scene = copy.deepcopy(PiDof23WalkFlatEnvCfg().scene)
    scene.robot = PI_PLUS_22DOF_CFG
    return scene


PI_22DOF_RUN_COMMAND_THRESHOLD = 0.6
PI_22DOF_RUN_AIR_RATIO = 0.6


def _make_pi_22dof_simple_amp_motion_cfg(include_run: bool = False) -> tuple[list[str], dict[str, float]]:
    """Use the complete neutral-head 22-DOF AMP package for simple-phase WAQ."""
    motion_weights = {
        "walk_forward_retargetted": 3.0,
        "walk_forward_retargetted_mirrored": 3.0,
        "walk_back_edited_105_180_retargetted": 1.0,
        "walk_back_edited_105_180_retargetted_mirrored": 1.0,
        "PiPlus_S_12L8A0G2H1W_LSE_Left_Turn_180_retargeted": 0.5,
        "PiPlus_S_12L8A0G2H0W_Right_Turn_180_retargeted": 0.5,
        "TURN_LEFT180DEGREES_retargeted": 0.5,
        "TURN_RIGHT180DEGREES_retargeted": 0.5,
        "walk_leftside_20260710_PiPlus_S_12L8A0G2H0W_retargeted": 1.0,
        "walk_rightside_20260713_PiPlus_S_12L8A0G2H0W_retargeted": 1.0,
    }
    if include_run:
        motion_weights["walk_forward_retargetted"] = 1.0
        motion_weights["walk_forward_retargetted_mirrored"] = 1.0
        motion_weights.update(
            {
                "run_20260710_PiPlus_S_12L8A0G2H0W_retargeted": 1.0,
                "run_20260710_PiPlus_S_12L8A0G2H0W_retargeted_mirrored": 1.0,
                "paoxiangyou_retargetted_0_21_fps29__retargetted": 1.0,
                "paoxiangyou_retargetted_0_21_fps29__retargetted_mirrored": 1.0,
                "paoxiangzuo_retargetted_135_167_fps29_retargetted": 1.0,
                "paoxiangzuo_retargetted_135_167_fps29_retargetted_mirrored": 1.0,
                "walk_back_edited_105_180_retargetted": 1.0,
                "walk_back_edited_105_180_retargetted_mirrored": 1.0,
            }
        )
    motion_files = [f"{PI_22DOF_SIMPLE_AMP_MOTION_DIR}/{name}.npz" for name in motion_weights]
    return (motion_files, motion_weights)


def _make_pi_22dof_robot_cfg():
    robot = copy.deepcopy(PiDof23WalkFlatEnvCfg().robot)
    robot.action_scale = PI_PLUS_22DOF_ACTION_SCALE
    return robot


@configclass
class PiDof22RewardCfg(PiDof23RewardCfg):
    """Pi rewards with terms that require a movable waist removed."""

    joint_deviation_waist_walk = None
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)
    feet_x_offset_lateral = RewTerm(
        func=mdp.feet_x_offset_lateral,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
            "lateral_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "forward_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "contact_threshold": 1.0,
            "x_deadband": 0.03,
        },
    )
    dof_torque_l2 = RewTerm(
        func=mdp.joint_torque_l2, weight=-5e-05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll.*", ".*_upper_arm.*", "head_.*"]),
        },
    )


@configclass
class PiDof22WalkFlatEnvCfg(PiDof23WalkFlatEnvCfg):
    """WAQ environment configuration using the 22-DOF Pi asset."""

    scene = _make_pi_22dof_scene_cfg()
    robot = _make_pi_22dof_robot_cfg()
    reward = PiDof22RewardCfg()


def _make_pi_22dof_walk_run_commands_cfg():
    """Keep the original 22-DOF command setup while exposing running speeds."""
    commands = copy.deepcopy(PiDof22WalkFlatEnvCfg().commands)
    commands.ranges.lin_vel_x = (-0.6, 1.2)
    return commands


@configclass
class PiDof22WalkRunRewardCfg(PiDof22RewardCfg):
    """22-DOF mixed walk/run rewards with disjoint command-speed gates."""

    fly = RewTerm(
        func=mdp.fly,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 1.0,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "max_lin_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
        },
    )
    foot_gait = RewTerm(
        func=mdp.mjlab_feet_gait,
        weight=0.5,
        params={
            "period": 1.0,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "max_lin_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
            "command_name": None,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
        },
    )
    gait_feet_frc_perio = RewTerm(
        func=mdp.gait_feet_frc_perio,
        weight=1.0,
        params={
            "delta_t": 0.015,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "max_lin_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
        },
    )
    gait_feet_frc_perio_penalize = RewTerm(
        func=mdp.gait_feet_frc_perio_penalize,
        weight=-1.0,
        params={
            "delta_t": 0.015,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "max_lin_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
        },
    )
    gait_feet_spd_perio = RewTerm(
        func=mdp.gait_feet_spd_perio,
        weight=1.0,
        params={
            "delta_t": 0.015,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "max_lin_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
        },
    )
    gait_feet_frc_perio_run = RewTerm(
        func=mdp.gait_feet_frc_perio_run,
        weight=1.0,
        params={
            "delta_t": 0.015,
            "run_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
            "run_air_ratio": PI_22DOF_RUN_AIR_RATIO,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
        },
    )
    gait_feet_frc_perio_penalize_run = RewTerm(
        func=mdp.gait_feet_frc_perio_penalize_run,
        weight=-1.0,
        params={
            "delta_t": 0.015,
            "run_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
            "run_air_ratio": PI_22DOF_RUN_AIR_RATIO,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
        },
    )
    gait_feet_spd_perio_run = RewTerm(
        func=mdp.gait_feet_spd_perio_run,
        weight=1.0,
        params={
            "delta_t": 0.015,
            "run_command_threshold": PI_22DOF_RUN_COMMAND_THRESHOLD,
            "run_air_ratio": PI_22DOF_RUN_AIR_RATIO,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
        },
    )


@configclass
class PiDof22WalkRunFlatEnvCfg(PiDof22WalkFlatEnvCfg):
    """Plane-only mixed walk/run environment for the 22-DOF Pi asset."""

    commands = _make_pi_22dof_walk_run_commands_cfg()
    reward = PiDof22WalkRunRewardCfg()


@configclass
class PiDof22WalkAgentCfg(PiDof23WalkAgentCfg):
    """WAQ agent configuration with 22 actions and 22-DOF AMP data."""

    experiment_name = "walk_pi_waq_simple_22"
    run_name = "new_amp_neutral_head"
    neptune_project = "walk_pi_waq_simple_22"
    wandb_project = "walk_pi_waq_simple_22"
    resume = False
    load_run = ""
    load_checkpoint = "model_.*.pt"
    policy = copy.deepcopy(PiDof23WalkAgentCfg().policy)
    amp_loader_class_name = "AMPLoaderPi22"
    amp_motion_files, amp_motion_weights = _make_pi_22dof_simple_amp_motion_cfg()
    min_normalized_std = [0.05] * 22


@configclass
class PiDof22WalkRunAgentCfg(PiDof22WalkAgentCfg):
    """Mixed walk/run agent using the existing simple-phase log namespace."""

    experiment_name = "walk_pi_waq_simple_22"
    neptune_project = "walk_pi_waq_simple_22"
    wandb_project = "walk_pi_waq_simple_22"
    run_name = "walk_run_amp_1p5"
    amp_motion_files, amp_motion_weights = _make_pi_22dof_simple_amp_motion_cfg(include_run=True)
