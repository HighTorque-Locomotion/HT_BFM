"""HT colleague actuator parameters for the existing Pi 22-DoF ZedMini asset.

Imported 2026-09-10 from HT_lab/assets/PiPlus_S_12L8A0G2H0W.py.
The robot URDF and runtime joint order remain those of pi_depth_amp.
Motor parameters, default pose, and action-scale formula follow HT_lab.
Do not use the source file's fixed symmetry indices with this URDF.
"""

import copy

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets.pi.ht_motor import HTMotorCfg_4438, HTMotorCfg_5036
from legged_lab.assets.pi.ht_motor_cfg import (
    ARMATURE_3536,
    ARMATURE_4438,
    ARMATURE_5036,
    DAMPING_3536,
    DAMPING_4438,
    DAMPING_5036,
    STIFFNESS_3536,
    STIFFNESS_4438,
    STIFFNESS_5036,
)
from legged_lab.assets.pi.pi_22dof_legacy import PI_PLUS_22DOF_CFG as _LEGACY_CFG

PiPlus_S_12L8A0G2H0W_CFG = copy.deepcopy(_LEGACY_CFG)
PiPlus_S_12L8A0G2H0W_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.351),
    joint_pos={
        ".*_hip_pitch_joint": -0.25,
        ".*_calf_joint": 0.65,
        ".*_ankle_pitch_joint": -0.4,
        ".*_elbow_joint": 0.0,
        "l_shoulder_roll_joint": 0.0,
        "l_shoulder_pitch_joint": 0.0,
        "r_shoulder_roll_joint": 0.0,
        "r_shoulder_pitch_joint": 0.0,
        "head_yaw_joint": 0.0,
        "head_pitch_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

PiPlus_S_12L8A0G2H0W_CFG.actuators = {
    "legs": HTMotorCfg_5036(
        joint_names_expr=[
            ".*_thigh_joint",
            ".*_hip_roll_joint",
            # ".*_hip_pitch_joint",
            ".*_calf_joint",
        ],
        effort_limit_sim={
            ".*_thigh_joint": 23.7,
            ".*_hip_roll_joint": 23.7,
            # ".*_hip_pitch_joint":  23.7,
            ".*_calf_joint": 23.7,
        },
        velocity_limit_sim={
            ".*_thigh_joint": 60.0,
            ".*_hip_roll_joint": 60.0,
            # ".*_hip_pitch_joint": 60.0,
            ".*_calf_joint": 60.0,
        },
        stiffness={
            # ".*_hip_pitch_joint": STIFFNESS_5047,
            ".*_hip_roll_joint": STIFFNESS_5036,
            ".*_thigh_joint": STIFFNESS_5036,
            ".*_calf_joint": STIFFNESS_5036,
        },
        damping={
            # ".*_hip_pitch_joint": DAMPING_5047,
            ".*_hip_roll_joint": DAMPING_5036,
            ".*_thigh_joint": DAMPING_5036,
            ".*_calf_joint": DAMPING_5036,
        },
        armature={
            # ".*_hip_pitch_joint": ARMATURE_5047,
            ".*_hip_roll_joint": ARMATURE_5036,
            ".*_thigh_joint": ARMATURE_5036,
            ".*_calf_joint": ARMATURE_5036,
        },
    ),
    "hip_pitch": HTMotorCfg_5036(
        effort_limit_sim=23.7,
        velocity_limit_sim=60.0,
        joint_names_expr=[".*_hip_pitch_joint"],
        stiffness=STIFFNESS_5036,
        damping=DAMPING_5036,
        armature=ARMATURE_5036,
    ),
    "feet": HTMotorCfg_5036(
        effort_limit_sim=23.7,
        velocity_limit_sim=60.0,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=STIFFNESS_5036,
        damping=DAMPING_5036,
        armature=ARMATURE_5036,
    ),
    "head": ImplicitActuatorCfg(
        effort_limit_sim=3.0,
        velocity_limit_sim=60.0,
        joint_names_expr=[".*head_yaw_joint", ".*head_pitch_joint"],
        stiffness=STIFFNESS_3536,
        damping=DAMPING_3536,
        armature=ARMATURE_3536,
    ),
    "arms": HTMotorCfg_4438(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_upper_arm_joint",
            ".*_elbow_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 20.0,
            ".*_shoulder_roll_joint": 20.0,
            ".*_upper_arm_joint": 20.0,
            ".*_elbow_joint": 20.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 60.0,
            ".*_shoulder_roll_joint": 60.0,
            ".*_upper_arm_joint": 60.0,
            ".*_elbow_joint": 60.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": STIFFNESS_4438,
            ".*_shoulder_roll_joint": STIFFNESS_4438,
            ".*_upper_arm_joint": STIFFNESS_4438,
            ".*_elbow_joint": STIFFNESS_4438,
        },
        damping={
            ".*_shoulder_pitch_joint": DAMPING_4438,
            ".*_shoulder_roll_joint": DAMPING_4438,
            ".*_upper_arm_joint": DAMPING_4438,
            ".*_elbow_joint": DAMPING_4438,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_4438,
            ".*_shoulder_roll_joint": ARMATURE_4438,
            ".*_upper_arm_joint": ARMATURE_4438,
            ".*_elbow_joint": ARMATURE_4438,
        },
    ),
}

PiPlus_S_12L8A0G2H0W_ACTION_SCALE = {}
for a in PiPlus_S_12L8A0G2H0W_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PiPlus_S_12L8A0G2H0W_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
