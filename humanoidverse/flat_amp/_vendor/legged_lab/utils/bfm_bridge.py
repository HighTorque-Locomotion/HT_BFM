"""Frozen Pi12 BFM interface. No Isaac Sim imports; usable in CPU contract tests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class BFMContract:
    joint_names: tuple[str, ...]
    default_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    action_multiplier: float = 32.0
    action_clip: float = 32.0
    history_length: int = 4
    z_dim: int = 256
    norm_z: bool = True
    angular_velocity_scale: float = 0.25
    control_dt: float = 0.02

    def __post_init__(self):
        if len(self.joint_names) != 12 or len(set(self.joint_names)) != 12:
            raise ValueError("BFM requires 12 unique Pi12 joint names")
        for field in (self.default_pos, self.action_scale):
            if len(field) != 12 or not np.isfinite(field).all():
                raise ValueError("BFM pose/scales must contain 12 finite values")
        values = (self.action_multiplier, self.action_clip, self.control_dt, self.angular_velocity_scale)
        if not np.isfinite(values).all() or min(self.action_scale) <= 0 or min(values) <= 0:
            raise ValueError("BFM scales, action bounds and control_dt must be positive")
        if self.history_length <= 0 or self.z_dim <= 0:
            raise ValueError("BFM history_length/z_dim must be positive")
        if not self.norm_z:
            raise ValueError("This migration supports norm_z=True checkpoints; raw latents require expert statistics")

    @classmethod
    def from_robot_yaml(cls, path: str | Path):
        """Read the *BFM checkpoint's* robot configuration, not the target PD profile."""
        with Path(path).expanduser().open() as stream:
            data = yaml.safe_load(stream)
        robot = data["robot"]
        names = tuple(robot["dof_names"])
        control = robot["control"]
        if control["control_type"] != "P" or control.get("isaac_pd_scale", False):
            raise ValueError("Expected Pi12 position control with isaac_pd_scale=False")
        scales = []
        for index, name in enumerate(names):
            scale = float(control["action_scale"])
            if control.get("action_rescale", False):
                gains = [float(value) for key, value in control["stiffness"].items() if key in name]
                if len(gains) != 1 or gains[0] <= 0:
                    raise ValueError(f"Cannot resolve unique BFM stiffness for {name}")
                # HumanoidVerse uses the original effort list here, not its torque clipping scale.
                scale *= float(robot["dof_effort_limit_list"][index]) / gains[0]
            scales.append(scale)
        multiplier = 1.0
        if control.get("normalize_action", False):
            multiplier = float(control["normalize_action_to"]) / float(control["normalize_action_from"])
        obs = data.get("obs", {})
        histories = obs.get("obs_auxiliary", {}).get("history_actor", {})
        if histories and (set(histories) != {"actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"}
                          or len(set(histories.values())) != 1):
            raise ValueError("BFM requires equally long grouped histories for all five actor fields")
        obs_scales = obs.get("obs_scales", {})
        for key in ("actions", "dof_pos", "dof_vel", "projected_gravity", "history_actor"):
            if obs_scales.get(key, 1.0) != 1.0:
                raise ValueError(f"Unsupported frozen BFM observation scale: {key}")
        if control.get("normalize_action_to_overrides"):
            raise ValueError("Per-joint action normalization overrides are not supported")
        sim = data.get("simulator", {}).get("config", {}).get("sim", {})
        return cls(
            joint_names=names,
            default_pos=tuple(float(robot["init_state"]["default_joint_angles"][name]) for name in names),
            action_scale=tuple(scales),
            action_multiplier=multiplier,
            action_clip=float(control["action_clip_value"]),
            history_length=int(next(iter(histories.values()), 4)),
            angular_velocity_scale=float(obs_scales.get("base_ang_vel", 0.25)),
            control_dt=float(sim.get("control_decimation", 4)) / float(sim.get("fps", 200)),
        )

    @property
    def state_dim(self):
        return 30

    @property
    def observation_dim(self):
        return self.state_dim + 12 + 42 * self.history_length

    def metadata(self):
        # JSON round-trip gives stable tuples/lists when comparing saved checkpoints.
        return json.loads(json.dumps(asdict(self)))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BFMBridge:
    """Maintain source-order observations and convert BFM actions to target PD coordinates."""

    def __init__(self, contract, target_joint_names, num_envs, device):
        self.contract = contract
        self.device = torch.device(device)
        self.target_joint_names = tuple(target_joint_names)
        if len(target_joint_names) != 12 or len(set(target_joint_names)) != 12 or set(target_joint_names) != set(contract.joint_names):
            raise ValueError("Target articulation and BFM must have exactly the same 12 joint names")
        self.to_bfm = torch.tensor([target_joint_names.index(n) for n in contract.joint_names], device=device)
        self.to_target = torch.argsort(self.to_bfm)
        self.default_pos = torch.tensor(contract.default_pos, device=device, dtype=torch.float32)
        self.action_scale = torch.tensor(contract.action_scale, device=device, dtype=torch.float32)
        self.last_action = torch.zeros(num_envs, 12, device=device)
        # Source history is grouped by field, newest first, excluding the current frame.
        self.history = {
            key: torch.zeros(num_envs, contract.history_length, dim, device=device)
            for key, dim in (
                ("actions", 12),
                ("base_ang_vel", 3),
                ("dof_pos", 12),
                ("dof_vel", 12),
                ("projected_gravity", 3),
            )
        }

    def observe(self, joint_pos, joint_vel, gravity, ang_vel):
        state = torch.cat(
            (
                joint_pos[:, self.to_bfm] - self.default_pos,
                joint_vel[:, self.to_bfm],
                gravity,
                ang_vel * self.contract.angular_velocity_scale,
            ),
            dim=-1,
        )
        return {
            "state": state,
            "last_action": self.last_action.clone(),
            "history_actor": torch.cat([value.flatten(1) for value in self.history.values()], dim=-1),
        }

    def encoder_observation(self, obs, command, phase_sin_cos=None):
        """Append the task's explicit clock only to the trainable encoder, not the frozen BFM."""
        state = obs["state"].clone()
        state[:, 12:24] *= 0.05
        history = obs["history_actor"].clone()
        offset = self.contract.history_length * (12 + 3 + 12)
        history[:, offset : offset + self.contract.history_length * 12] *= 0.05
        fields = [command, state, obs["last_action"], history]
        if phase_sin_cos is not None:
            if phase_sin_cos.shape != (state.shape[0], 2):
                raise ValueError("Expected phase sin/cos with shape [num_envs, 2]")
            fields.append(phase_sin_cos)
        return torch.cat(fields, dim=-1)

    def project_z(self, raw_z):
        if raw_z.shape != (self.last_action.shape[0], self.contract.z_dim) or not torch.isfinite(raw_z).all():
            raise ValueError("Command encoder returned an invalid latent batch")
        return math.sqrt(self.contract.z_dim) * torch.nn.functional.normalize(raw_z, dim=-1)

    def target_actions(self, actor_output, target_default_pos, target_action_scale):
        if actor_output.shape != self.last_action.shape or not torch.isfinite(actor_output).all():
            raise ValueError("BFM actor must return finite [num_envs, 12] actions")
        sim_action = (actor_output * self.contract.action_multiplier).clamp(
            -self.contract.action_clip, self.contract.action_clip
        )
        desired_pos = (self.default_pos + sim_action * self.action_scale)[:, self.to_target]
        if torch.any(torch.as_tensor(target_action_scale, device=self.device) <= 0):
            raise ValueError("Target action scales must be positive")
        # The target environment still applies its PD, encoder bias and delay.
        return (desired_pos - target_default_pos) / target_action_scale, sim_action

    def advance(self, previous_obs, sim_action, done_ids):
        state = previous_obs["state"]
        fields = {
            "actions": previous_obs["last_action"],
            "base_ang_vel": state[:, -3:],
            "dof_pos": state[:, :12],
            "dof_vel": state[:, 12:24],
            "projected_gravity": state[:, 24:27],
        }
        for key, history in self.history.items():
            history[:, 1:] = history[:, :-1].clone()
            history[:, 0] = fields[key]
            history[done_ids] = 0
        self.last_action.copy_(sim_action)
        self.last_action[done_ids] = 0


class BFMLatentEnv:
    """Expose 256 PPO actions while the wrapped reward manager still sees 12 joint actions."""

    def __init__(self, env, actor, contract, use_phase=True):
        self.env = env
        self.actor = actor
        self.use_phase = bool(use_phase)
        self.num_actions = contract.z_dim
        self.bridge = BFMBridge(contract, list(env.robot.joint_names), env.num_envs, env.device)
        if not math.isclose(env.step_dt, contract.control_dt, abs_tol=1e-8):
            raise ValueError(f"BFM requires {contract.control_dt}s policy steps, target has {env.step_dt}s")
        self._observations = None
        self._actor_obs = None
        self._extras = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def _refresh(self, extras):
        data = self.env.robot.data
        # Reuse the task's observation noise and scales, but undo its normalization before
        # building BFM state. This keeps clean privileged critic inputs independent.
        frame = extras["observations"]["current_actor_obs_add_nosie"]
        if frame.shape != (self.env.num_envs, 47):
            raise ValueError("Expected Pi12 frame [N,47], ending in phase sin/cos")
        scales = self.env.obs_scales
        q = frame[:, 9:21] / scales.joint_pos + data.default_joint_pos
        dq = frame[:, 21:33] / scales.joint_vel + data.default_joint_vel
        self._observations = self.bridge.observe(
            q,
            dq,
            frame[:, 3:6] / scales.projected_gravity,
            frame[:, :3] / scales.ang_vel,
        )
        self._actor_obs = self.bridge.encoder_observation(
            self._observations,
            frame[:, 6:9],
            phase_sin_cos=frame[:, 45:47] if self.use_phase else None,
        )
        # Append BFM memory to the target's privileged critic so its value is conditioned on latent-control history.
        extras = {**extras, "observations": dict(extras["observations"])}
        extras["observations"]["critic"] = torch.cat((extras["observations"]["critic"], self._actor_obs), dim=-1)
        self._extras = extras
        return self._actor_obs, extras

    def get_observations(self):
        if self._observations is None:
            _, extras = self.env.get_observations()
            return self._refresh(extras)
        return self._actor_obs, self._extras

    def refresh_play_observations(self):
        """Apply a changed command/phase before inference without advancing BFM memory."""
        from humanoidverse.flat_amp.keyboard_control import refresh_play_observations

        self.get_observations()
        extras = self.env.extras
        observations = extras["observations"]
        refresh_play_observations(self.env, observations["obs_history"], observations)
        # Use the physical critic, not our cached critic with BFM memory appended.
        return self._refresh(extras)

    @torch.no_grad()
    def step(self, raw_z):
        self.get_observations()
        output = self.actor(self._observations, self.bridge.project_z(raw_z))
        action, sim_action = self.bridge.target_actions(
            output, self.env.robot.data.default_joint_pos, self.env.action_scale
        )
        _, reward, done, extras = self.env.step(action)
        # Raw source action (32*x), before target action delay, is the BFM feedback convention.
        self.bridge.advance(self._observations, sim_action, self.env.reset_env_ids)
        obs, extras = self._refresh(extras)
        return obs, reward, done, extras

