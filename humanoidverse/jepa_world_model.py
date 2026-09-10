"""JEPA runtime, online adaptation, and latent-residual MPC utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from humanoidverse.pretrain_jepa_world_model import JEPAWorldModel


@dataclass
class FrozenJEPAWorldModel:
    model: JEPAWorldModel
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    checkpoint: Path | None
    trainable: bool = False
    initialization: str = "checkpoint"
    normalization_updates: int = 0

    @classmethod
    def from_scratch(
        cls,
        *,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        context_len: int = 8,
    ) -> "FrozenJEPAWorldModel":
        """Create a trainable JEPA using only future simulator samples."""
        model = JEPAWorldModel(obs_dim, action_dim, hidden_dim, embedding_dim, context_len)
        model.to(device=device, dtype=torch.float32).eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith("target_encoder."))
        return cls(
            model=model,
            obs_mean=torch.zeros(obs_dim, device=device),
            obs_std=torch.ones(obs_dim, device=device),
            action_mean=torch.zeros(action_dim, device=device),
            action_std=torch.ones(action_dim, device=device),
            checkpoint=None,
            trainable=True,
            initialization="scratch_simulator",
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        *,
        trainable: bool = False,
    ) -> "FrozenJEPAWorldModel":
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"JEPA checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        obs_mean = torch.as_tensor(checkpoint["obs_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(checkpoint["obs_std"], dtype=torch.float32).clamp_min(1.0e-6)
        action_mean = torch.as_tensor(checkpoint["action_mean"], dtype=torch.float32)
        action_std = torch.as_tensor(checkpoint["action_std"], dtype=torch.float32).clamp_min(1.0e-6)
        if obs_mean.ndim != 1 or action_mean.ndim != 1:
            raise ValueError("JEPA checkpoint normalization statistics must be one-dimensional")
        model = JEPAWorldModel(
            obs_dim=int(obs_mean.numel()),
            action_dim=int(action_mean.numel()),
            hidden_dim=int(config.get("hidden_dim", 512)),
            embedding_dim=int(config.get("embedding_dim", 256)),
            context_len=int(config.get("context_len", 8)),
        )
        try:
            incompatible = model.load_state_dict(checkpoint["model"], strict=False)
            invalid_missing = [key for key in incompatible.missing_keys if not key.startswith("motion_head.")]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}")
        except RuntimeError as exc:
            raise ValueError(
                "JEPA checkpoint is incompatible with frozen MPC. "
                "Re-run pretrain_jepa_world_model.py after enabling its observation decoder."
            ) from exc
        model.to(device=device, dtype=torch.float32).eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(bool(trainable) and not name.startswith("target_encoder."))
        return cls(
            model=model,
            obs_mean=obs_mean.to(device),
            obs_std=obs_std.to(device),
            action_mean=action_mean.to(device),
            action_std=action_std.to(device),
            checkpoint=checkpoint_path,
            trainable=bool(trainable),
            initialization="legacy_checkpoint_no_motion_head" if incompatible.missing_keys else "checkpoint",
        )

    @property
    def obs_dim(self) -> int:
        return int(self.obs_mean.numel())

    @property
    def action_dim(self) -> int:
        return int(self.action_mean.numel())

    @property
    def context_len(self) -> int:
        return int(self.model.context_len)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.embedding_dim)

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.obs_mean) / self.obs_std

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    @torch.no_grad()
    def update_normalization(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        momentum: float = 0.95,
        statistic_sync: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Update input statistics from simulator samples, optionally across ranks."""
        if not 0.0 <= momentum < 1.0:
            raise ValueError("normalization momentum must be in [0, 1)")
        flat_obs = observations.reshape(-1, self.obs_dim).float()
        flat_actions = actions.reshape(-1, self.action_dim).float()
        obs_mean, obs_var = flat_obs.mean(0), flat_obs.var(0, unbiased=False)
        action_mean, action_var = flat_actions.mean(0), flat_actions.var(0, unbiased=False)
        if statistic_sync is not None:
            obs_mean, obs_var = statistic_sync(obs_mean), statistic_sync(obs_var)
            action_mean, action_var = statistic_sync(action_mean), statistic_sync(action_var)
        obs_std = obs_var.clamp_min(1.0e-12).sqrt()
        action_std = action_var.clamp_min(1.0e-12).sqrt()
        blend = 0.0 if self.normalization_updates == 0 else float(momentum)
        self.obs_mean.lerp_(obs_mean, 1.0 - blend)
        self.obs_std.lerp_(obs_std, 1.0 - blend).clamp_(min=1.0e-3)
        self.action_mean.lerp_(action_mean, 1.0 - blend)
        self.action_std.lerp_(action_std, 1.0 - blend).clamp_(min=1.0e-3)
        self.normalization_updates += 1


def train_jepa_world_model(
    *,
    world_model: FrozenJEPAWorldModel,
    optimizer: torch.optim.Optimizer,
    context_observations: torch.Tensor,
    action_sequences: torch.Tensor,
    target_observations: torch.Tensor,
    target_motion: torch.Tensor,
    batch_size: int,
    updates: int,
    target_momentum: float,
    observation_reconstruction_coef: float,
    motion_prediction_coef: float,
    grad_clip: float,
    gradient_sync: Callable[[Iterable[torch.nn.Parameter]], None] | None = None,
    update_normalization: bool = False,
    normalization_momentum: float = 0.95,
    statistic_sync: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Adapt JEPA on simulator transitions using its self-supervised objective.

    The online target is a future normalized actor observation. Simulator
    ``base_lin_vel`` and yaw rate are privileged labels for the motion head,
    never inputs. The context
    encoder/action predictor and observation decoder receive gradients, while
    the target encoder follows the context encoder through EMA updates.
    """
    if not world_model.trainable:
        raise ValueError("Online JEPA updates require a trainable world-model runtime")
    if context_observations.ndim != 3 or context_observations.shape[1:] != (
        world_model.context_len,
        world_model.obs_dim,
    ):
        raise ValueError(
            f"Expected online JEPA contexts [B, {world_model.context_len}, {world_model.obs_dim}], "
            f"got {tuple(context_observations.shape)}"
        )
    if action_sequences.ndim != 3 or action_sequences.shape[0] != context_observations.shape[0]:
        raise ValueError("Online JEPA action sequences must share the context batch dimension")
    if action_sequences.shape[1] < world_model.context_len or action_sequences.shape[2] != world_model.action_dim:
        raise ValueError(
            f"Expected online JEPA actions [B, >= {world_model.context_len}, {world_model.action_dim}], "
            f"got {tuple(action_sequences.shape)}"
        )
    if target_observations.shape != (context_observations.shape[0], world_model.obs_dim):
        raise ValueError(
            f"Expected online JEPA targets [{context_observations.shape[0]}, {world_model.obs_dim}], "
            f"got {tuple(target_observations.shape)}"
        )
    if target_motion.shape != (context_observations.shape[0], 3):
        raise ValueError(f"Expected motion labels [{context_observations.shape[0]}, 3], got {tuple(target_motion.shape)}")
    if batch_size <= 0 or updates <= 0:
        raise ValueError("Online JEPA batch size and update count must be positive")
    if not 0.0 < target_momentum <= 1.0:
        raise ValueError("Online JEPA target momentum must be in (0, 1]")
    if observation_reconstruction_coef < 0.0 or motion_prediction_coef < 0.0 or grad_clip < 0.0:
        raise ValueError("Online JEPA loss coefficients and gradient clip must be non-negative")

    if update_normalization:
        world_model.update_normalization(
            torch.cat((context_observations, target_observations[:, None]), dim=1),
            action_sequences,
            momentum=normalization_momentum,
            statistic_sync=statistic_sync,
        )

    context = world_model.normalize_observations(context_observations)
    actions = world_model.normalize_actions(action_sequences)
    targets = world_model.normalize_observations(target_observations)
    sample_count = int(context.shape[0])
    if sample_count == 0:
        return {
            "world_model_online_loss": 0.0,
            "world_model_online_latent_loss": 0.0,
            "world_model_online_reconstruction_loss": 0.0,
            "world_model_online_motion_loss": 0.0,
            "world_model_online_motion_mae": 0.0,
            "world_model_online_vx_mae": 0.0,
            "world_model_online_vy_mae": 0.0,
            "world_model_online_yaw_rate_mae": 0.0,
            "world_model_online_grad_norm": 0.0,
            "world_model_online_sample_count": 0.0,
        }

    model = world_model.model
    model.train()
    totals = torch.zeros(9, device=context.device, dtype=torch.float64)
    for _ in range(int(updates)):
        indices = torch.randperm(sample_count, device=context.device)[: min(int(batch_size), sample_count)]
        batch_context = context[indices]
        batch_actions = actions[indices]
        batch_targets = targets[indices]
        batch_motion = target_motion[indices]
        prediction = model(batch_context, batch_actions)
        target = model.target(batch_targets)
        prediction_direction = F.normalize(prediction, dim=-1)
        target_direction = F.normalize(target.detach(), dim=-1)
        latent_loss = (2.0 - 2.0 * (prediction_direction * target_direction).sum(dim=-1)).mean()
        predicted_observation = model.decode(prediction)
        target_observation_prediction = model.decode(target.detach())
        reconstruction_loss = F.smooth_l1_loss(predicted_observation, batch_targets)
        reconstruction_loss = reconstruction_loss + 0.5 * F.smooth_l1_loss(
            target_observation_prediction,
            batch_targets,
        )
        predicted_motion = model.predict_motion(prediction)
        motion_loss = F.smooth_l1_loss(predicted_motion, batch_motion)
        motion_error = (predicted_motion - batch_motion).abs()
        motion_mae = motion_error.mean()
        loss = (
            latent_loss
            + float(observation_reconstruction_coef) * reconstruction_loss
            + float(motion_prediction_coef) * motion_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_sync is not None:
            gradient_sync(model.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)) if grad_clip > 0.0 else 0.0
        optimizer.step()
        model.update_target(float(target_momentum))
        totals += torch.tensor(
            (
                float(loss.detach()),
                float(latent_loss.detach()),
                float(reconstruction_loss.detach()),
                float(motion_loss.detach()),
                float(motion_mae.detach()),
                float(motion_error[:, 0].mean().detach()),
                float(motion_error[:, 1].mean().detach()),
                float(motion_error[:, 2].mean().detach()),
                float(grad_norm),
            ),
            device=context.device,
            dtype=torch.float64,
        )
    model.eval()
    totals /= float(updates)
    return {
        "world_model_online_loss": float(totals[0]),
        "world_model_online_latent_loss": float(totals[1]),
        "world_model_online_reconstruction_loss": float(totals[2]),
        "world_model_online_motion_loss": float(totals[3]),
        "world_model_online_motion_mae": float(totals[4]),
        "world_model_online_vx_mae": float(totals[5]),
        "world_model_online_vy_mae": float(totals[6]),
        "world_model_online_yaw_rate_mae": float(totals[7]),
        "world_model_online_grad_norm": float(totals[8]),
        "world_model_online_sample_count": float(sample_count),
    }


def _repeat_observation(observation: Mapping[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    repeated: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        if value.ndim < 1:
            raise ValueError(f"BFM observation {key!r} must have a batch dimension")
        repeated[key] = value.unsqueeze(1).expand(-1, repeats, *value.shape[1:]).reshape(value.shape[0] * repeats, *value.shape[1:])
    return repeated


def _bfm_action(bfm_model, observation: Mapping[str, torch.Tensor], projected_z: torch.Tensor) -> torch.Tensor:
    normalized = bfm_model._normalize(dict(observation))
    distribution = bfm_model._actor(normalized, projected_z, bfm_model.cfg.actor_std)
    return distribution.mean.float()


def jepa_mpc_command_mask(
    commands: torch.Tensor,
    *,
    command_deadzone: float = 0.1,
) -> torch.Tensor:
    """Keep zero-command standing direct while allowing forward/backward MPC.

    The velocity objective comes from a learned proprioceptive estimator. Its
    simulator-only ``base_lin_vel`` label is not required at deployment.
    """
    if commands.ndim != 2 or commands.shape[-1] < 3:
        raise ValueError(f"Expected commands [B, >=3], got {tuple(commands.shape)}")
    if command_deadzone < 0.0:
        raise ValueError(f"command_deadzone must be non-negative, got {command_deadzone}")
    active = (commands[:, :2].norm(dim=-1) >= float(command_deadzone)) | (
        commands[:, 2].abs() >= float(command_deadzone)
    )
    return active


@torch.no_grad()
def plan_jepa_latent_residuals(
    *,
    world_model: FrozenJEPAWorldModel,
    context_observations: torch.Tensor,
    action_history: torch.Tensor,
    base_projected_z: torch.Tensor,
    basis: torch.Tensor,
    bfm_model,
    bfm_observation: Mapping[str, torch.Tensor],
    commands: torch.Tensor,
    last_action: torch.Tensor,
    horizon: int,
    num_candidates: int,
    residual_scale: float,
    velocity_cost_weight: float = 2.0,
    yaw_cost_weight: float = 1.0,
    upright_cost_weight: float = 1.0,
    action_rate_cost_weight: float = 0.02,
    residual_cost_weight: float = 0.01,
) -> torch.Tensor:
    """Select the first latent of the best JEPA candidate sequence.

    The decoder makes the latent prediction observable to the MPC objective.
    The motion head predicts planar velocity and yaw rate only from deployable
    proprioceptive history plus candidate actions. The five-step objective also
    penalizes every consecutive action change, not only the first action.
    """
    batch_size, latent_dim = base_projected_z.shape
    if context_observations.shape != (batch_size, world_model.context_len, world_model.obs_dim):
        raise ValueError(
            f"Expected JEPA context [{batch_size}, {world_model.context_len}, {world_model.obs_dim}], "
            f"got {tuple(context_observations.shape)}"
        )
    if action_history.shape != (batch_size, world_model.context_len - 1, world_model.action_dim):
        raise ValueError("JEPA action history must contain context_len - 1 actions")
    if latent_dim != world_model.embedding_dim:
        raise ValueError(
            f"JEPA embedding dimension {world_model.embedding_dim} must match BFM latent dimension {latent_dim}"
        )
    if basis.ndim != 2 or basis.shape[1] != latent_dim:
        raise ValueError(f"Expected latent basis [P, {latent_dim}], got {tuple(basis.shape)}")
    if horizon <= 0 or num_candidates < 2 or residual_scale < 0.0:
        raise ValueError("Invalid frozen-JEPA MPC horizon, candidate count, or residual scale")
    if world_model.obs_dim != 631 or world_model.action_dim != int(last_action.shape[-1]):
        raise ValueError(
            f"Frozen JEPA expects actor_obs=631 and action={last_action.shape[-1]}, "
            f"got obs={world_model.obs_dim}, action={world_model.action_dim}"
        )

    residuals = torch.randn(
        batch_size,
        num_candidates,
        horizon,
        basis.shape[0],
        device=base_projected_z.device,
        dtype=base_projected_z.dtype,
    )
    residuals[:, 0] = 0.0  # Candidate zero is always the direct policy latent.
    latent_norm = base_projected_z.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    candidate_z = F.normalize(
        base_projected_z[:, None, None, :] + float(residual_scale) * (residuals @ basis), dim=-1
    ) * latent_norm[:, None, None, :]

    candidate_actions = []
    repeated_observation = _repeat_observation(bfm_observation, num_candidates)
    for step in range(horizon):
        step_z = candidate_z[:, :, step].reshape(batch_size * num_candidates, latent_dim)
        candidate_actions.append(_bfm_action(bfm_model, repeated_observation, step_z).reshape(batch_size, num_candidates, -1))
    candidate_actions_tensor = torch.stack(candidate_actions, dim=2)

    history = world_model.normalize_actions(action_history)[:, None].expand(-1, num_candidates, -1, -1)
    future = world_model.normalize_actions(candidate_actions_tensor)
    action_sequence = torch.cat((history, future), dim=2).reshape(
        batch_size * num_candidates,
        world_model.context_len + horizon - 1,
        world_model.action_dim,
    )
    context = world_model.normalize_observations(context_observations)[:, None].expand(
        -1, num_candidates, -1, -1
    ).reshape(batch_size * num_candidates, world_model.context_len, world_model.obs_dim)
    predicted_latent = world_model.model(context, action_sequence)
    predicted_motion = world_model.model.predict_motion(predicted_latent).reshape(batch_size, num_candidates, 3)
    predicted_observation = world_model.model.decode(predicted_latent)
    predicted_observation = predicted_observation.reshape(batch_size, num_candidates, world_model.obs_dim)
    predicted_observation = predicted_observation * world_model.obs_std + world_model.obs_mean

    # ROSBAG actor_obs layout is state[52], last_action[23],
    # history_actor[300], z[256]. Within state: dof_pos[0:23],
    # dof_vel[23:46], projected_gravity[46:49], base_ang_vel[49:52].
    predicted_gravity_xy = predicted_observation[:, :, 46:48]
    if world_model.initialization == "legacy_checkpoint_no_motion_head":
        velocity_cost = torch.zeros_like(predicted_observation[:, :, 0])
        yaw_cost = (predicted_observation[:, :, 51] - commands[:, None, 2]).square()
    else:
        velocity_cost = (predicted_motion[:, :, :2] - commands[:, None, :2]).square().sum(dim=-1)
        yaw_cost = (predicted_motion[:, :, 2] - commands[:, None, 2]).square()
    upright_cost = predicted_gravity_xy.square().sum(dim=-1)
    previous_and_candidates = torch.cat((last_action[:, None, None].expand(-1, num_candidates, -1, -1), candidate_actions_tensor), dim=2)
    action_rate_cost = (previous_and_candidates[:, :, 1:] - previous_and_candidates[:, :, :-1]).square().mean(dim=(2, 3))
    residual_cost = residuals.square().mean(dim=(2, 3))
    score = (
        -float(velocity_cost_weight) * velocity_cost
        - float(yaw_cost_weight) * yaw_cost
        - float(upright_cost_weight) * upright_cost
        - float(action_rate_cost_weight) * action_rate_cost
        - float(residual_cost_weight) * residual_cost
    )
    winner = score.argmax(dim=1)
    winner = torch.where(jepa_mpc_command_mask(commands), winner, torch.zeros_like(winner))
    return candidate_z[torch.arange(batch_size, device=score.device), winner, 0]
