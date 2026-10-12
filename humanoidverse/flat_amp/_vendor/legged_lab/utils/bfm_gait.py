"""Command-dependent BFM gait clock and reward-neutral locomotion diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class BackwardFastGait:
    backward_period: float = 0.5
    default_period: float = 0.8
    lin_threshold: float = 0.1
    yaw_threshold: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Gait periods and command thresholds must be finite and positive")

    def metadata(self):
        return {
            "name": "backward_fast", "version": 1, **asdict(self),
            "backward_gate": "command_vx <= -lin_threshold; includes backward mixed commands",
            "phase_update": "progress += dt/T(executed_command); modulo one; standing progress and phase zero",
            "reset_phase_offsets": [0.0, 0.5], "right_leg_offset": 0.5,
            "command_sampling": "original task, unchanged",
            "reward_parameters": "original task, unchanged; all phase rewards use this same gait_phase",
        }


def command_masks(command, cfg):
    x, y, yaw = command.unbind(-1)
    moving = command[:, :2].norm(dim=-1) >= cfg.lin_threshold
    turning = yaw.abs() >= cfg.yaw_threshold
    lateral = (x.abs() < cfg.lin_threshold) & (y.abs() >= cfg.lin_threshold) & ~turning
    return {
        "backward": x <= -cfg.lin_threshold,
        "forward": (x >= cfg.lin_threshold) & (y.abs() < cfg.lin_threshold) & ~turning,
        "left": lateral & (y > 0), "right": lateral & (y < 0),
        "turn": ~moving & turning, "standing": ~moving & ~turning,
    }


def yaw_velocity(data):
    w, x, y, z = data.root_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y.square() + z.square()))
    vx, vy = data.root_lin_vel_w[:, :2].unbind(-1)
    return torch.stack((yaw.cos()*vx + yaw.sin()*vy, -yaw.sin()*vx + yaw.cos()*vy), -1)


class CommandGaitClock:
    def __init__(self, num_envs, device, cfg):
        self.cfg = cfg
        self.progress = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.period = torch.full((num_envs,), cfg.default_period, device=device)

    def reset(self, ids):
        self.progress[ids] = 0
        self.period[ids] = self.cfg.default_period

    def advance(self, command, dt, start_offset):
        masks = command_masks(command, self.cfg)
        self.period.fill_(self.cfg.default_period)
        self.period[masks["backward"]] = self.cfg.backward_period
        self.progress.add_(dt / self.period.double()).remainder_(1.0)
        self.progress[masks["standing"]] = 0
        phase = (self.progress + start_offset.double()).remainder(1.0).float()
        phase[masks["standing"]] = 0
        return phase


class ContactCycleTracker:
    """Measure complete same-foot cycles; no cadence reward or policy input."""
    def __init__(self, num_envs, device):
        self.time = 0.0
        self.off_time = torch.zeros(num_envs, 2, device=device)
        self.on_time = torch.zeros_like(self.off_time)
        self.armed = torch.zeros_like(self.off_time, dtype=torch.bool)
        self.last_touchdown = torch.full_like(self.off_time, -1)

    def reset(self, ids):
        self.off_time[ids] = self.on_time[ids] = 0
        self.armed[ids] = False
        self.last_touchdown[ids] = -1

    def update(self, force, dt, valid):
        self.time += dt
        self.reset(~valid)
        self.off_time = torch.where((force < 2) & ~self.armed, self.off_time + dt, 0)
        self.armed |= self.off_time >= .06 - 1e-6
        self.on_time = torch.where(self.armed & (force >= 5), self.on_time + dt, 0)
        touchdown = self.armed & (self.on_time >= .04 - 1e-6) & valid[:, None]
        now = self.time - self.on_time + dt
        period = now - self.last_touchdown
        event = touchdown & (self.last_touchdown >= 0)
        self.last_touchdown = torch.where(touchdown, now, self.last_touchdown)
        self.armed &= ~touchdown
        self.on_time[touchdown] = self.off_time[touchdown] = 0
        return event, period


def with_gait_environment(base_class, profile=None):
    """Optional variable clock; all original physical/task rewards remain unchanged."""
    class GaitEnvironment(base_class):
        def __init__(self, *args, **kwargs):
            self.gait_profile = profile
            self.command_gait_clock = None
            self.cycle_tracker = None
            self._gait_sums = {}
            super().__init__(*args, **kwargs)
            self._metric_cfg = profile or BackwardFastGait()
            if profile is not None:
                self.command_gait_clock = CommandGaitClock(self.num_envs, self.device, profile)
                self.phase = self.gait_start.clone()
                self._write_leg_phase()
            self.cycle_tracker = ContactCycleTracker(self.num_envs, self.device)
            self._previous_metric_command = self.command_generator.command.clone()
            self._stop_elapsed = torch.zeros(self.num_envs, device=self.device)
            self._arm_ids = [
                i for i, name in enumerate(self.robot.joint_names) if any(part in name for part in ("shoulder", "upper_arm", "elbow"))
            ]

        def _write_leg_phase(self):
            self.gait_phase[:, 0] = self.phase
            self.gait_phase[:, 1] = (self.phase + .5) % 1
            standing = command_masks(self.command_generator.command, self._metric_cfg)["standing"]
            self.phase[standing] = 0
            self.gait_phase[standing] = 0

        def _calculate_gait_para(self):
            if self.command_gait_clock is None:
                return super()._calculate_gait_para()
            self.phase = self.command_gait_clock.advance(self.command_generator.command, self.step_dt, self.gait_start)
            self.gait_cycle.copy_(self.command_gait_clock.period)
            self._write_leg_phase()

        def reset(self, ids):
            super().reset(ids)
            if self.command_gait_clock is not None and len(ids):
                self.command_gait_clock.reset(ids)
                self.phase[ids] = self.gait_start[ids]
                self.gait_cycle[ids] = profile.default_period
                self._write_leg_phase()
            if self.cycle_tracker is not None and len(ids):
                self.cycle_tracker.reset(ids)
                self._previous_metric_command[ids] = self.command_generator.command[ids]
                self._stop_elapsed[ids] = 0

        def _record(self, key, value, mask):
            pair = torch.stack(((value * mask).sum(), mask.sum().to(value.dtype)))
            if key in self._gait_sums:
                self._gait_sums[key].add_(pair)
            else:
                self._gait_sums[key] = pair

        def pop_gait_metrics(self):
            result = {}
            for key, pair in self._gait_sums.items():
                total, count = pair.detach().cpu().tolist()
                if count:
                    result[key] = total / count
            self._gait_sums.clear()
            return result

        def step(self, actions):
            command = self.command_generator.command.clone()
            changed = ((command - self._previous_metric_command).abs() > command.new_tensor([.05, .05, .1])).any(-1)
            self.cycle_tracker.reset(changed)
            self._previous_metric_command.copy_(command)
            result = super().step(actions)
            done = result[2].bool()
            valid = ~done
            force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2].abs()
            self._cycle_event, self._cycle_period = self.cycle_tracker.update(force, self.step_dt, valid)
            velocity = yaw_velocity(self.robot.data)
            for key, mask in command_masks(command, self._metric_cfg).items():
                self._record(f"Direction/{key}/termination", done.float(), mask)
                mask = mask & valid
                self._record(f"Direction/{key}/vx_signed", velocity[:, 0], mask)
                self._record(f"Direction/{key}/vy_signed", velocity[:, 1], mask)
                self._record(f"Direction/{key}/vx_error", (velocity[:, 0] - command[:, 0]).abs(), mask)
                self._record(f"Direction/{key}/vy_error", (velocity[:, 1] - command[:, 1]).abs(), mask)
                self._record(f"Direction/{key}/yaw_error", (self.robot.data.root_ang_vel_w[:, 2] - command[:, 2]).abs(), mask)
                self._record(f"Gait/{key}/actual_same_foot_period", self._cycle_period, self._cycle_event & mask[:, None])
                self._record(f"Gait/{key}/scheduled_period", self.gait_cycle, mask)
            standing = command_masks(command, self._metric_cfg)["standing"] & valid
            self._stop_elapsed = torch.where(standing, self._stop_elapsed + self.step_dt, 0)
            self._record("Stop/arm_dq_squared_after_2s", self.robot.data.joint_vel[:, self._arm_ids].square().mean(-1),
                         self._stop_elapsed > 2)
            return result

    return GaitEnvironment
