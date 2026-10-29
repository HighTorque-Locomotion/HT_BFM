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

"""walk_pi_waq_12 的独立任务配置，参照已验证的 Pi12 delivery 版本展开。

奖励函数/权重/参数：PiDof12RewardCfg；步态：GaitCfg；随机化：PiEventCfg。
环境、指令、噪声、控制频率：PiDof12WalkFlatEnvCfg；PPO/AMP：PiDof12WalkAgentCfg。
仅复用通用配置类型和 mdp 函数，不继承 Pi22/Pi23 的任务配置。
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

import legged_lab.mdp as mdp
from humanoidverse.flat_amp.pi12_paths import motion_yaml, robot_urdf
from humanoidverse.flat_amp.velocity_curriculum import VelocityCurriculumCfg
from legged_lab.assets.pi.pi_12dof import PI_12DOF_ACTION_SCALE, PI_12DOF_CFG
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
from legged_lab.mdp.lateral_window import LateralVelocityWindow

LEG_JOINT_NAMES = [f".*_{joint}_joint" for joint in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")]

# AMP 使用全部 12 个腿部关节。
AMP_EXCLUDE_ANKLE_ROLL = False
PI_LIN_COMMAND_THRESHOLD = 0.1
PI_YAW_COMMAND_THRESHOLD = 0.1


@configclass
class PiAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    beta: float = 1e-4


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
class PiDof12RewardCfg:
    """在此添加/删除奖励项，或修改 func、weight、params；None 或零权重禁用该项。"""

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
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
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
            "threshold": 150,
            "max_reward": 400,
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh.*", ".*_hip_roll.*"]),
        },
    )
    joint_deviation_hip_walk = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh.*", ".*_hip_roll.*"]),
        },
    )
    joint_deviation_waist_walk = None
    joint_deviation_arms = None
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
            # One discrete penalty per swing; each foot uses its own local phase.
            "phase_start": 0.10,
            "phase_end": 0.40,
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
        weight=0.0,
    )
    track_command_tilt_exp = RewTerm(
        func=mdp.track_command_tilt_exp,
        weight=1.0,
        params={
            "std": math.radians(5.0),
            "tolerance": 0.0,
            "gravity": 9.81,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
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
        # weight=-5.0,
        weight=0.0,  # Disable idle penalties for the lateral drift ablation.
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
            "lin_velocity_threshold": 0.1,
            "yaw_velocity_threshold": 0.1,
        },
    )

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={
            "lin_command_threshold": PI_LIN_COMMAND_THRESHOLD,
            "yaw_command_threshold": PI_YAW_COMMAND_THRESHOLD,
        },
    )
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
    lateral_vx_center = RewTerm(
        func=LateralVelocityWindow, weight=-2.0,
        params={"channel": "vx", "window_seconds": 1.0},
    )
    lateral_wz_center = RewTerm(
        func=LateralVelocityWindow, weight=-2.0,
        params={"channel": "wz", "window_seconds": 1.0},
    )
    dof_torque_l2 = RewTerm(func=mdp.joint_torque_l2, weight=-5e-05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})


@configclass
class PiEventCfg:
    """Pi12 的启动随机化、重置和周期推力配置。"""

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
class FootContactRandomizationCfg:
    """Startup-only compliance: 30% rigid, 70% log-uniform soft contact."""

    enabled: bool = True
    rigid_fraction: float = 0.3
    # reproduction_fraction: float = 0.1  # Disabled fixed k=2000 N/m, c=0 group.
    reproduction_fraction: float = 0.0
    reproduction_stiffness: float = 2000.0
    stiffness_range: tuple[float, float] = (2000.0, 200000.0)
    damping_ratio_range: tuple[float, float] = (0.0, 0.5)
    reference_mass: float = 8.0  # Tuning reference, not the effective mass of each contact point.


@configclass
class PiDof12WalkFlatEnvCfg:
    amp_exclude_ankle_roll = AMP_EXCLUDE_ANKLE_ROLL
    amp_motion_files_display = []
    device: str = "cuda:0"
    foot_contact_randomization = FootContactRandomizationCfg()
    # Fixed values used only when foot_contact_randomization.enabled=False.
    foot_contact_stiffness: float = 0.0  # N/m
    foot_contact_damping: float = 0.0  # N·s/m
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=PI_12DOF_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=1,
        action_scale=PI_12DOF_ACTION_SCALE,
        terminate_contacts_body_names=["base_link", "torso_link", ".*_hip_.*_link"],
        feet_body_names=[".*ankle_roll.*"],
        max_tilt_angle_deg=45.0,
        max_tilt_angle_rad=None,
    )
    reward = PiDof12RewardCfg()
    gait = GaitCfg()
    velocity_curriculum = VelocityCurriculumCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(lin_vel=1.0, ang_vel=1.0, projected_gravity=1.0, commands=1.0, joint_pos=1.0, joint_vel=1.0, actions=1.0),
        clip_observations=100.0,
        clip_actions=100.0,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_onestep_envs=0.15,
        rel_turn_only_envs=0.15,
        rel_straight_envs=0.15,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,  # Allows the heading controller to reach the final +/-2 rad/s cap.
        debug_vis=True,
        ranges=CommandRangesCfg(lin_vel_x=(-1.0, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)),
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
class PiDof12WalkAgentCfg(RslRlOnPolicyRunnerCfg):
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
    save_interval = 1000
    runner_class_name = "AmpWaqOnPolicyRunner"
    experiment_name = "walk_pi_waq_12"
    run_name = "pi12_ht5036_amp"
    logger = "tensorboard"
    neptune_project = "walk_pi_waq_12"
    wandb_project = "walk_pi_waq_12"
    resume = False
    load_run = ""
    load_checkpoint = "model_.*.pt"
    amp_loader_class_name = "AMPLoaderPi12"
    amp_reward_coef = 0.3
    amp_exclude_ankle_roll = AMP_EXCLUDE_ANKLE_ROLL
    amp_motion_files = []
    amp_motion_weights = None
    # 保留 HT_BFM 的资产/数据路径解析及环境变量覆盖；由 loader 延迟读取。
    amp_loader_kwargs = {"motion_yaml": str(motion_yaml()), "urdf_path": str(robot_urdf())}
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.65
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 12
