# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Flat velocity-command adaptation, extracted from the local Isaac Lab extension.

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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
import omni.log
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from humanoidverse.flat_amp.velocity_curriculum import VelocityCurriculum, VelocityCurriculumCfg


class UniformVelocityCommand(CommandTerm):
    """Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \\omega_z = \\frac{1}{2} \\text{wrap_to_pi}(\\theta_{\\text{target}} - \\theta_{\\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    "The configuration of the command generator."

    def __init__(self, cfg: UniformVelocityCommandCfg, env):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        super().__init__(cfg, env)
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading` parameter is set to None."
            )
        if self.cfg.ranges.heading and (not self.cfg.heading_command):
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}' but the heading command is not active. Consider setting the flag for the heading command to True."
            )
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_onestep_env = torch.zeros_like(self.is_heading_env)
        self.is_turn_only_env = torch.zeros_like(self.is_heading_env)
        self.is_straight_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.velocity_curriculum = None
        if cfg.velocity_curriculum is not None and cfg.velocity_curriculum.enabled:
            self.velocity_curriculum = VelocityCurriculum(cfg.velocity_curriculum, cfg.ranges, self.num_envs, self.device)
        self.current_velocity_ranges = torch.tensor(
            [cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z], device=self.device
        ).expand(self.num_envs, -1, -1).clone()

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    "\n    Properties\n    "

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    "\n    Implementation specific functions.\n    "

    def _update_metrics(self):
        if self.velocity_curriculum is not None:
            self.velocity_curriculum.observe(
                self.vel_command_b, self.robot.data.root_lin_vel_b[:, :2], self.robot.data.root_ang_vel_b[:, 2],
                self._curriculum_modes(), self.is_standing_env,
            )
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _curriculum_modes(self):
        # The command sampler makes these categories mutually exclusive.
        modes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        modes[self.is_onestep_env] = 1
        modes[self.is_turn_only_env] = 2
        modes[self.is_straight_env] = 3
        return modes

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if isinstance(ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[ids]
        if self.velocity_curriculum is not None:
            self.velocity_curriculum.reset(ids, self._env.max_episode_length)
        extras = super().reset(ids)
        self._update_command()
        if self.velocity_curriculum is not None:
            extras.update(self.velocity_curriculum.logs())
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        rnums = torch.rand(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = False
        self.is_onestep_env[env_ids] = False
        self.is_turn_only_env[env_ids] = False
        self.is_straight_env[env_ids] = False
        self.is_standing_env[env_ids] = rnums <= self.cfg.rel_standing_envs
        self.is_onestep_env[env_ids] = (rnums > self.cfg.rel_standing_envs) & (
            rnums <= self.cfg.rel_standing_envs + self.cfg.rel_onestep_envs
        )
        self.is_turn_only_env[env_ids] = (rnums > self.cfg.rel_standing_envs + self.cfg.rel_onestep_envs) & (
            rnums <= self.cfg.rel_standing_envs + self.cfg.rel_onestep_envs + self.cfg.rel_turn_only_envs
        )
        straight_start = self.cfg.rel_standing_envs + self.cfg.rel_onestep_envs + self.cfg.rel_turn_only_envs
        self.is_straight_env[env_ids] = (rnums > straight_start) & (rnums <= straight_start + self.cfg.rel_straight_envs)
        if self.velocity_curriculum is not None:
            modes = self._curriculum_modes()[env_ids]
            ranges = self.velocity_curriculum.ranges(env_ids, modes)
            self.current_velocity_ranges[env_ids] = ranges
            self.vel_command_b[env_ids] = ranges[..., 0] + torch.rand(
                ranges.shape[:2], device=self.device
            ) * (ranges[..., 1] - ranges[..., 0])

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            if self.velocity_curriculum is not None:
                # Heading control must obey the same per-environment course limits.
                self.vel_command_b[env_ids, 2] = torch.clamp(
                    self.vel_command_b[env_ids, 2],
                    min=self.current_velocity_ranges[env_ids, 2, 0],
                    max=self.current_velocity_ranges[env_ids, 2, 1],
                )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        onestep_env_ids = self.is_onestep_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[onestep_env_ids, 0] = 0.0
        self.vel_command_b[onestep_env_ids, 2] = 0.0
        turn_only_env_ids = self.is_turn_only_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[turn_only_env_ids, 0] = 0.0
        self.vel_command_b[turn_only_env_ids, 1] = 0.0
        # Apply after heading control so straight commands retain zero yaw rate on every step.
        straight_env_ids = self.is_straight_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[straight_env_ids, 1:] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        elif hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.set_visibility(False)
            self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    "\n    Internal helpers.\n    "

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return (arrow_scale, arrow_quat)


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand
    asset_name: str = MISSING
    "Name of the asset in the environment for which the commands are generated."
    heading_command: bool = False
    "Whether to use heading command or angular velocity command. Defaults to False.\n\n    If True, the angular velocity command is computed from the heading error, where the\n    target heading is sampled uniformly from provided range. Otherwise, the angular velocity\n    command is sampled uniformly from provided range.\n    "
    heading_control_stiffness: float = 1.0
    "Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."
    rel_standing_envs: float = 0.0
    "The sampled probability of environments that should be standing still. Defaults to 0.0."
    rel_onestep_envs: float = 0.0
    rel_turn_only_envs: float = 0.0
    rel_straight_envs: float = 0.0
    "Probability of pure forward/backward commands with lateral and yaw velocities set to zero."
    rel_heading_envs: float = 1.0
    "The sampled probability of environments where the robots follow the heading-based angular velocity command\n    (the others follow the sampled angular velocity command). Defaults to 1.0.\n\n    This parameter is only used if :attr:`heading_command` is True.\n    "

    velocity_curriculum: VelocityCurriculumCfg | None = None

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        "Range for the linear-x velocity command (in m/s)."
        lin_vel_y: tuple[float, float] = MISSING
        "Range for the linear-y velocity command (in m/s)."
        ang_vel_z: tuple[float, float] = MISSING
        "Range for the angular-z velocity command (in rad/s)."
        heading: tuple[float, float] | None = None
        "Range for the heading command (in rad). Defaults to None.\n\n        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.\n        "

    ranges: Ranges = MISSING
    "Distribution ranges for the velocity commands."
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    "The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    "The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
