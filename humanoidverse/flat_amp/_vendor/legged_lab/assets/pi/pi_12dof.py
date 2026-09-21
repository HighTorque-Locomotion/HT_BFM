"""Pi12 asset and identified 5047 motors, matching HT_lab's Pi12 configuration."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuator
from isaaclab.assets import ArticulationCfg

from humanoidverse.flat_amp.pi12_paths import robot_urdf
from legged_lab.assets.pi.ht_motor import HTMotor, HTMotorCfg_5047
from legged_lab.assets.pi.ht_motor_cfg import ARMATURE_5047, DAMPING_5047, STIFFNESS_5047


class Pi12HTMotor(HTMotor):
    """Use HT_lab's delayed PD path without changing the older PiPlus motors."""

    def compute(self, control_action, joint_pos, joint_vel):
        self._joint_vel[:] = joint_vel
        return DelayedPDActuator.compute(self, control_action, joint_pos, joint_vel)


def _motor(patterns):
    # HT_lab's 5047 profile differs from the bundled PiPlus 18.7 Nm profile.
    return HTMotorCfg_5047(
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
