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

"""Configuration for the fixed-waist, 22-DOF Pi Plus robot."""

import copy

from legged_lab.assets import ISAAC_ASSET_DIR
from legged_lab.assets.pi.pi import PI_PLUS_ACTION_SCALE, PI_PLUS_CFG


PI_PLUS_22DOF_CFG = copy.deepcopy(PI_PLUS_CFG)
PI_PLUS_22DOF_CFG.spawn.asset_path = (
    f"{ISAAC_ASSET_DIR}/pi/zedmini_260804/urdf/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_260804_22dof.urdf"
)
PI_PLUS_22DOF_CFG.spawn.usd_dir = f"{ISAAC_ASSET_DIR}/pi/zedmini_260804/usd"
PI_PLUS_22DOF_CFG.spawn.usd_file_name = "pi_22dof_zedmini_260804.usd"
PI_PLUS_22DOF_CFG.spawn.root_link_name = "base_link"
PI_PLUS_22DOF_CFG.spawn.merge_fixed_joints = False
PI_PLUS_22DOF_CFG.spawn.replace_cylinders_with_capsules = True
PI_PLUS_22DOF_CFG.spawn.force_usd_conversion = True
PI_PLUS_22DOF_CFG.actuators.pop("waist")

# Match the effort limits in the 22-DOF URDF.  Every joint within each
# actuator group has the same limit in this asset:
#   legs/ankles: 21 Nm, arms: 10 Nm, head: 3.7 Nm.
PI_PLUS_22DOF_CFG.actuators["legs"].effort_limit_sim = 21.0
PI_PLUS_22DOF_CFG.actuators["feet"].effort_limit_sim = 21.0
PI_PLUS_22DOF_CFG.actuators["arms"].effort_limit_sim = 10.0
PI_PLUS_22DOF_CFG.actuators["head"].effort_limit_sim = 3.7

# The new simple-phase AMP data keeps head_pitch_joint near neutral (0 rad).
# Do not inherit the 23-DOF asset's forward-looking head-pitch initialization.
PI_PLUS_22DOF_CFG.init_state.joint_pos["head_pitch_joint"] = 0.0
PI_PLUS_22DOF_CFG.actuators["head"].stiffness = {
    "head_yaw_joint": 80.0,
    "head_pitch_joint": 80.0,
}
PI_PLUS_22DOF_CFG.actuators["head"].damping = {
    "head_yaw_joint": 1.1,
    "head_pitch_joint": 1.1,
}

PI_PLUS_22DOF_ACTION_SCALE = {
    joint_pattern: scale for joint_pattern, scale in PI_PLUS_ACTION_SCALE.items() if joint_pattern != "waist_yaw_joint"
}
