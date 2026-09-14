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

from isaaclab.terrains import TerrainImporterCfg
from typing import TYPE_CHECKING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

if TYPE_CHECKING:
    from legged_lab.envs.base.base_config import BaseSceneCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        )
        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt
        )
        self.light = AssetBaseCfg(
            prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
        )
        self.sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
