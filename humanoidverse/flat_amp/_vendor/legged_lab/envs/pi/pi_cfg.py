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
import math
from pathlib import Path
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
import legged_lab.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.pi import PI_PLUS_CFG, PI_PLUS_ACTION_SCALE, LEG_JOINT_NAMES
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)

AMP_EXCLUDE_ANKLE_ROLL = False
AMP_MOTION_FORMAT = "npz"
AMP_MOTION_DIR = str(Path(LEGGED_LAB_ROOT_DIR) / "legged_lab/envs/pi/datasets/motion_amp_expert/new_npz")
PI_LIN_COMMAND_THRESHOLD = 0.1
PI_YAW_COMMAND_THRESHOLD = 0.1


@configclass
class PiAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    beta: float = 1e-4


def _amp_motion_path(name: str) -> str:
    return f"{AMP_MOTION_DIR}/{name}.{AMP_MOTION_FORMAT}"


def _make_amp_motion_cfg() -> tuple[list[str], dict[str, float]]:
    motion_weights = {"walk_left_head_pitch_0p9": 1.5, "walk_right_head_pitch_0p9": 1.5, "walk_lhy_head_pitch_0p9": 3.0}
    motion_weights.update({"left": 1.0, "right": 1.0, "walk_back": 2.0})
    return ([_amp_motion_path(name) for name in motion_weights], motion_weights)


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.5
    gait_air_ratio_r: float = 0.5
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.8
    lin_command_threshold: float = PI_LIN_COMMAND_THRESHOLD
    yaw_command_threshold: float = PI_YAW_COMMAND_THRESHOLD


@configclass
class PiDof23RewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.5})
    root_heading_yaw_l2 = RewTerm(
        func=mdp.root_heading_yaw_l2,
        weight=-2.0,
        params={
            "target_yaw": 0.0,
            "only_vel_limited": True,
            "body_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_xy_l2_body = RewTerm(
        func=mdp.ang_vel_xy_l2_body, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-0.05
    )
    energy = RewTerm(func=mdp.energy, weight=-0.001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_rate_smooth = RewTerm(func=mdp.action_smoothness, weight=-0.003)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 200,
            "max_reward": 200,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"]
            ),
        },
    )
    joint_deviation_hip_walk = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"]
            ),
        },
    )
    joint_deviation_waist_walk = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"]),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_upper_arm.*", "head_.*"]
            ),
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_calf.*", ".*_ankle.*"]),
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
        },
    )
    gait_feet_frc_perio_penalize = RewTerm(
        func=mdp.gait_feet_frc_perio_penalize,
        weight=-1.0,
        params={
            "delta_t": 0.015,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
        },
    )
    gait_feet_spd_perio = RewTerm(
        func=mdp.gait_feet_spd_perio,
        weight=1.0,
        params={
            "delta_t": 0.015,
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
        },
    )
    body_orientation_euler = RewTerm(
        func=mdp.body_orientation_euler,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
        weight=1.0,
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_exp,
        weight=-2.0,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-5.0,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "lin_velocity_threshold": 0.1,
            "yaw_velocity_threshold": 0.1,
        },
    )


@configclass
class PiEventCfg:
    """H2-style domain randomization adapted to the Pi robot."""

    foot_friction = EventTerm(
        func=mdp.SharedBodyFrictionRandomizer,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"), "friction_range": (0.3, 1.6)},
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_joint_encoder_bias,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "bias_range": (-0.015, 0.015)},
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.9, 1.1),
            "dynamic_friction_distribution_params": (0.9, 1.1),
            "viscous_friction_distribution_params": (0.9, 1.1),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class PiDof23WalkFlatEnvCfg:
    amp_exclude_ankle_roll = AMP_EXCLUDE_ANKLE_ROLL
    amp_motion_files_display = []
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=PI_PLUS_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=1,
        action_scale=PI_PLUS_ACTION_SCALE,
        terminate_contacts_body_names=[
            "base_link",
            "torso_link",
            "head_.*_link",
            ".*_shoulder_.*_link",
            ".*_elbow_link",
            ".*_hip_.*_link",
            ".*_wrist_link",
        ],
        feet_body_names=[".*ankle_roll.*"],
    )
    reward = PiDof23RewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0, ang_vel=1.0, projected_gravity=1.0, commands=1.0, joint_pos=1.0, joint_vel=1.0, actions=1.0
        ),
        clip_observations=100.0,
        clip_actions=100.0,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_onestep_envs=0.05,
        rel_turn_only_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(lin_vel=0.2, ang_vel=0.2, projected_gravity=0.05, joint_pos=0.01, joint_vel=1.5),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=PiEventCfg(), action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0})
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class PiDof23WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticWaq",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = PiAmpAlgorithmCfg(
        class_name="AMPWAQPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            data_augmentation_func="legged_lab.envs.pi.symmetry:compute_symmetric_states",
            use_mirror_loss=True,
            mirror_loss_coeff=10.0,
        ),
        rnd_cfg=None,
        beta=0.0001,
    )
    clip_actions = None
    save_interval = 500
    runner_class_name = "AmpWaqOnPolicyRunner"
    experiment_name = "walk_pi_waq_23"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_pi_waq_23"
    wandb_project = "walk_pi_waq_23"
    resume = False
    load_run = ""
    load_checkpoint = "model_.*.pt"
    amp_loader_class_name = "AMPLoaderPi"
    amp_reward_coef = 0.3
    amp_exclude_ankle_roll = AMP_EXCLUDE_ANKLE_ROLL
    amp_motion_files, amp_motion_weights = _make_amp_motion_cfg()
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.65
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23
