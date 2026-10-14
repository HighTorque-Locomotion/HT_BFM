"""Pure lateral velocity centering, adapted from hi_walk_23dof extrema history."""

import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class LateralVelocityWindow(ManagerTermBase):
    """Return |maximum + minimum| after a full, uninterrupted lateral window.

    Each instance owns one velocity channel. Reset and command-mode/direction
    changes discard history; extrema from another maneuver cannot enter it.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        duration = cfg.params.get("window_seconds", 1.0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self.window_steps = max(1, math.ceil(duration / env.step_dt - 1e-9))
        self.history = torch.zeros(env.num_envs, self.window_steps, device=env.device)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.direction = torch.zeros_like(self.count)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0
        self.count[env_ids] = 0
        self.direction[env_ids] = 0

    def __call__(self, env, channel: str, command_name: str | None = None,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 window_seconds: float = 1.0, lateral_command_threshold: float = 0.1,
                 forward_command_threshold: float = 0.1, yaw_command_threshold: float = 0.1,
                 walk_only: bool = False):
        command = (env.command_generator.command if command_name is None
                   else env.command_manager.get_command(command_name))
        active = ((command[:, 1].abs() >= lateral_command_threshold)
                  & (command[:, 0].abs() < forward_command_threshold)
                  & (command[:, 2].abs() < yaw_command_threshold))
        if walk_only:
            active &= env._is_walk_env | (env._is_approach_kick_env & ~env._in_kick_zone)
        direction = torch.where(active, command[:, 1].sign().long(), 0)
        changed = direction != self.direction
        self.history[changed] = 0
        self.count[changed] = 0
        self.direction.copy_(direction)

        data = env.scene[asset_cfg.name].data
        if channel == "vx":
            velocity = math_utils.quat_apply_inverse(
                math_utils.yaw_quat(data.root_quat_w), data.root_lin_vel_w
            )[:, 0]
        elif channel == "wz":
            # Same world-z yaw rate as the existing angular-velocity tracker.
            velocity = data.root_ang_vel_w[:, 2]
        else:
            raise ValueError(f"Unknown lateral-window velocity channel: {channel}")
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = torch.where(active, velocity, 0.0)
        self.count = torch.where(active, (self.count + 1).clamp(max=self.window_steps), 0)
        centered = (self.history.amax(dim=1) + self.history.amin(dim=1)).abs()
        return torch.where(active & (self.count == self.window_steps), centered, 0.0)
