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

from legged_lab.utils.task_registry import task_registry
from .pi.pi_env import PiDof23Env
from .pi.pi_22dof_env import PiDof22Env
from .pi.pi_cfg import PiDof23WalkFlatEnvCfg, PiDof23WalkAgentCfg
from .pi.pi_22dof_cfg import (
    PiDof22WalkFlatEnvCfg,
    PiDof22WalkAgentCfg,
    PiDof22WalkRunFlatEnvCfg,
    PiDof22WalkRunAgentCfg,
)

task_registry.register("walk_pi_waq_22", PiDof22Env, PiDof22WalkFlatEnvCfg(), PiDof22WalkAgentCfg())
task_registry.register("walk_pi_waq_run_22", PiDof22Env, PiDof22WalkRunFlatEnvCfg(), PiDof22WalkRunAgentCfg())
task_registry.register("walk_pi_waq_23", PiDof23Env, PiDof23WalkFlatEnvCfg(), PiDof23WalkAgentCfg())

from .pi.pi_12dof_cfg import PiDof12WalkAgentCfg, PiDof12WalkFlatEnvCfg
from .pi.pi_12dof_env import PiDof12Env

task_registry.register("walk_pi_waq_12", PiDof12Env, PiDof12WalkFlatEnvCfg(), PiDof12WalkAgentCfg())
