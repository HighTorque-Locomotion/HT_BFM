"""Performance-based Pi12 command ranges, independent of Isaac Sim.

Reference: HT_lab hi_walk_23dof@701e2e0, hi_walk_amp velocity_ranges_curriculum.
The flat task uses command modes in place of terrain types. Each environment keeps
independent mode levels, so standing and turning cannot advance straight walking.
"""

from dataclasses import dataclass

import torch

MODES = ("mixed", "lateral", "turn", "straight")
AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


@dataclass
class VelocityCurriculumCfg:
    enabled: bool = True
    tracking_threshold: float = 0.7
    downgrade_threshold: float = 0.55
    turn_tracking_threshold: float = 0.5
    turn_downgrade_threshold: float = 0.35
    increase_step: float = 0.05
    decrease_step: float = 0.02
    tracking_std: float = 0.5
    initial_lin_vel_x: tuple[float, float] = (-0.15, 0.30)
    initial_lin_vel_y: tuple[float, float] = (-0.15, 0.15)
    initial_ang_vel_z: tuple[float, float] = (-0.30, 0.30)
    initial_turn_ang_vel_z: tuple[float, float] = (-0.50, 0.50)
    initial_straight_lin_vel_x: tuple[float, float] = (0.10, 0.50)


class VelocityCurriculum:
    def __init__(self, cfg, target_ranges, num_envs, device):
        self.cfg = cfg
        self.device = device
        self.levels = torch.zeros(num_envs, len(MODES), device=device)
        self.score_sums = torch.zeros_like(self.levels)
        self.counts = torch.zeros_like(self.levels)
        self.episode_steps = torch.zeros(num_envs, device=device)
        self.env_ids = torch.arange(num_envs, device=device)
        self.initial = torch.tensor([
            [cfg.initial_lin_vel_x, cfg.initial_lin_vel_y, cfg.initial_ang_vel_z],
            [(0, 0), cfg.initial_lin_vel_y, (0, 0)],
            [(0, 0), (0, 0), cfg.initial_turn_ang_vel_z],
            [cfg.initial_straight_lin_vel_x, (0, 0), (0, 0)],
        ], dtype=torch.float32, device=device)
        x, y, yaw = (getattr(target_ranges, key) for key in AXES)
        self.target = torch.tensor([
            [x, y, yaw], [(0, 0), y, (0, 0)], [(0, 0), (0, 0), yaw], [x, (0, 0), (0, 0)],
        ], dtype=torch.float32, device=device)
        if (not torch.isfinite(self.initial).all() or not torch.isfinite(self.target).all()
                or torch.any(self.initial[..., 0] > self.initial[..., 1])
                or torch.any(self.target[..., 0] > self.initial[..., 0])
                or torch.any(self.target[..., 1] < self.initial[..., 1])):
            raise ValueError("Initial velocity ranges must be finite, ordered and inside final ranges")
        if not (0 <= cfg.downgrade_threshold < cfg.tracking_threshold <= 1
                and 0 <= cfg.turn_downgrade_threshold < cfg.turn_tracking_threshold <= 1
                and 0 < cfg.increase_step <= 1 and 0 < cfg.decrease_step <= 1
                and 0 < cfg.tracking_std < float("inf")):
            raise ValueError("Invalid velocity curriculum thresholds, steps or tracking_std")

    def ranges(self, env_ids, modes):
        level = self.levels[env_ids, modes, None, None]
        return self.initial[modes] + level * (self.target[modes] - self.initial[modes])

    def observe(self, command, lin_vel_xy, yaw_rate, modes, standing):
        xy_error = (command[:, :2] - lin_vel_xy).square().sum(dim=-1)
        yaw_error = (command[:, 2] - yaw_rate).square()
        error = torch.where(modes == MODES.index("turn"), yaw_error, xy_error)
        score = torch.exp(-error / self.cfg.tracking_std**2)
        active = ~standing
        ids, mode = self.env_ids[active], modes[active]
        self.score_sums[ids, mode] += score[active]
        self.counts[ids, mode] += 1
        self.episode_steps += 1

    def reset(self, env_ids, max_episode_steps):
        counts = self.counts[env_ids]
        # HT_lab divides tracking sums by the full episode length, so early falls
        # cannot earn a high score from a few good frames. Apply the same duration
        # factor to each mode's mean when this task switches modes mid-episode.
        coverage = (self.episode_steps[env_ids] / max_episode_steps).clamp(max=1).unsqueeze(-1)
        score = self.score_sums[env_ids] / counts.clamp(min=1) * coverage
        up = score.new_full((len(MODES),), self.cfg.tracking_threshold)
        down = score.new_full((len(MODES),), self.cfg.downgrade_threshold)
        up[2], down[2] = self.cfg.turn_tracking_threshold, self.cfg.turn_downgrade_threshold
        level = self.levels[env_ids]
        level += ((counts > 0) & (score > up)) * self.cfg.increase_step
        level -= ((counts > 0) & (score < down)) * self.cfg.decrease_step
        self.levels[env_ids] = level.clamp(0, 1)
        self.score_sums[env_ids] = 0
        self.counts[env_ids] = 0
        self.episode_steps[env_ids] = 0

    def logs(self):
        result = {}
        mean_levels = self.levels.mean(dim=0)
        mean_ranges = (self.initial + mean_levels[:, None, None] * (self.target - self.initial)).detach().cpu()
        mean_levels = mean_levels.detach().cpu()
        for mode, name in enumerate(MODES):
            level = mean_levels[mode]
            ranges = mean_ranges[mode]
            result[f"Curriculum/{name}/level"] = level.item()
            for axis, key in enumerate(AXES):
                result[f"Curriculum/{name}/{key}_min"] = ranges[axis, 0].item()
                result[f"Curriculum/{name}/{key}_max"] = ranges[axis, 1].item()
        return result

    def state_dict(self):
        return {"schema": 1, "modes": list(MODES), "levels": self.levels.detach().cpu().clone()}

    def load_state_dict(self, state):
        if state["schema"] != 1 or state["modes"] != list(MODES):
            raise ValueError("Unsupported velocity curriculum checkpoint")
        levels = torch.as_tensor(state["levels"], dtype=self.levels.dtype, device=self.device)
        if (levels.ndim != 2 or levels.shape[1] != len(MODES) or levels.shape[0] == 0
                or not torch.isfinite(levels).all() or torch.any((levels < 0) | (levels > 1))):
            raise ValueError("Invalid saved velocity curriculum levels")
        if levels.shape != self.levels.shape:
            # Environment identity is unavailable when resuming with a different count.
            levels = levels.mean(dim=0, keepdim=True).expand_as(self.levels)
        self.levels.copy_(levels)
        self.score_sums.zero_()
        self.counts.zero_()
        self.episode_steps.zero_()


def save_velocity_curriculum(env):
    curriculum = getattr(getattr(env, "command_generator", None), "velocity_curriculum", None)
    return curriculum.state_dict() if isinstance(curriculum, VelocityCurriculum) else None


def load_velocity_curriculum(env, state):
    curriculum = getattr(getattr(env, "command_generator", None), "velocity_curriculum", None)
    if isinstance(curriculum, VelocityCurriculum) and state is not None:
        curriculum.load_state_dict(state)
        # Current startup commands remain valid (inside the restored wider ranges).
        # The next normal command resample uses the restored levels.
