# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pi12 asset and identified 5047 motors, matching HT_lab's Pi12 configuration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from humanoidverse.flat_amp.pi12_paths import robot_urdf

ARMATURE_5047 = 0.012910
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS_5047 = ARMATURE_5047 * NATURAL_FREQ**2
DAMPING_5047 = 2.0 * DAMPING_RATIO * ARMATURE_5047 * NATURAL_FREQ


class Pi12HTMotor(DelayedPDActuator):
    """Pi12 delayed PD actuator with the identified 5047 torque-speed curve."""

    def compute(self, control_action, joint_pos, joint_vel):
        self._joint_vel[:] = joint_vel
        return DelayedPDActuator.compute(self, control_action, joint_pos, joint_vel)


    def __init__(self, cfg: Pi12HTMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)

        # Store curve parameters
        self._curve_a = cfg.curve_param_a  # -0.0141
        self._curve_b = cfg.curve_param_b  # -0.0709
        self._curve_c = cfg.curve_param_c  # 6.2756
        self._max_torque = cfg.max_torque  # 20.0
        self._max_velocity = cfg.max_velocity  # 6.0

        # Override parent class limits with identified values
        self._saturation_effort = self._max_torque

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip efforts based on identified torque-speed curve.

        Args:
            effort: Computed effort from PD controller. Shape: (num_envs, num_joints)

        Returns:
            Clipped effort tensor with same shape as input.
        """
        # Get current joint velocities
        vel = self._joint_vel

        # Compute absolute values
        abs_vel = torch.abs(vel)
        abs_effort = torch.abs(effort)

        # Determine operating mode: True for motoring, False for braking
        # Motoring: torque and velocity have same sign (product > 0)
        # Braking: torque and velocity have opposite signs (product < 0)
        is_motoring = (effort * vel) >= 0.0

        # --- Compute maximum torque for motoring mode ---
        # Solve the curve equation for maximum torque at current velocity
        # Given: |ω| = -0.0141|T|² - 0.0709|T| + 6.2756
        # Rearrange to: 0.0141|T|² + 0.0709|T| + (|ω| - 6.2756) = 0
        # Use quadratic formula: T = (-b + sqrt(b² - 4ac)) / (2a)
        # Note: We take the positive root to get the valid positive solution

        a = -self._curve_a  # 0.0141
        b = -self._curve_b  # 0.0709
        c = abs_vel - self._curve_c  # |ω| - 6.2756

        # Discriminant
        discriminant = b * b - 4 * a * c
        # Clamp discriminant to avoid sqrt of negative numbers (numerical issues)
        discriminant = torch.clamp(discriminant, min=0.0)

        # Maximum torque from curve (use positive root to get valid solution)
        max_torque_motoring = (-b + torch.sqrt(discriminant)) / (2 * a)

        # Ensure within absolute limits
        max_torque_motoring = torch.clamp(max_torque_motoring, min=0.0, max=self._max_torque)

        # For velocities exceeding max velocity, torque should be zero in motoring mode
        max_torque_motoring = torch.where(
            abs_vel > self._max_velocity, torch.zeros_like(max_torque_motoring), max_torque_motoring
        )

        # --- Compute maximum torque for braking mode ---
        # In braking mode, simple rectangular limit applies
        max_torque_braking = torch.full_like(abs_vel, self._max_torque)

        # For velocities exceeding max velocity, allow full braking torque
        # (This is physically reasonable - you can brake harder at high speeds)

        # --- Select appropriate limit based on operating mode ---
        max_torque = torch.where(is_motoring, max_torque_motoring, max_torque_braking)

        # --- Apply the limits ---
        # Clip absolute effort, then restore sign
        effort_sign = torch.sign(effort)
        clipped_abs_effort = torch.clamp(abs_effort, max=max_torque)
        clamped_effort = effort_sign * clipped_abs_effort

        return clamped_effort


@configclass
class Pi12HTMotorCfg(DelayedPDActuatorCfg):
    class_type: type = Pi12HTMotor
    curve_param_a: float = -0.0141
    curve_param_b: float = -0.0709
    curve_param_c: float = 6.2756
    max_torque: float = 20.0
    max_velocity: float = 6.0
    saturation_effort: float = 20.0


def _motor(patterns):
    return Pi12HTMotorCfg(
        class_type=Pi12HTMotor,
        joint_names_expr=patterns,
        effort_limit_sim=20.0,
        velocity_limit_sim=60.0,
        stiffness=STIFFNESS_5047,
        damping=DAMPING_5047,
        armature=ARMATURE_5047,
        max_torque=20.0,
        max_velocity=6.0,
        saturation_effort=20.0,
        min_delay=0,
        max_delay=4,
    )


PI_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(robot_urdf()),
        fix_base=False,
        root_link_name="base_link",
        # Rewards and randomization address torso_link explicitly.
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.351),
        joint_pos={".*_hip_pitch_joint": -0.25, ".*_calf_joint": 0.65, ".*_ankle_pitch_joint": -0.4},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": _motor([".*_thigh_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_calf_joint"]),
        "feet": _motor([".*_ankle_pitch_joint", ".*_ankle_roll_joint"]),
    },
)

PI_12DOF_ACTION_SCALE = 0.25 * 20.0 / STIFFNESS_5047
