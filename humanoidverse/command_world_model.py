"""World-model utilities for command-conditioned BFM latent planning.

The models in this module deliberately model the transition caused by a
*frozen* BFM actor.  Their action is therefore the projected BFM latent, not a
PiPlus joint command.  Keeping this boundary explicit makes the planner an
optional Stage2 controller and keeps first-stage checkpoints unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommandWorldModel(nn.Module):
    """Predict compact Stage2 state, reward, and termination from BFM latent."""

    def __init__(self, state_dim: int, command_dim: int, latent_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.command_dim = int(command_dim)
        self.latent_dim = int(latent_dim)
        if min(self.state_dim, self.command_dim, self.latent_dim, int(hidden_dim)) <= 0:
            raise ValueError("World-model dimensions must be positive.")
        input_dim = self.state_dim + self.command_dim + self.latent_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_state = nn.Linear(hidden_dim, self.state_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.termination = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        commands: torch.Tensor,
        projected_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state [N, {self.state_dim}], got {tuple(state.shape)}")
        if commands.shape != (state.shape[0], self.command_dim):
            raise ValueError(f"Expected commands [N, {self.command_dim}], got {tuple(commands.shape)}")
        if projected_z.shape != (state.shape[0], self.latent_dim):
            raise ValueError(f"Expected projected_z [N, {self.latent_dim}], got {tuple(projected_z.shape)}")
        hidden = self.trunk(torch.cat((state, commands, projected_z), dim=-1))
        return state + self.delta_state(hidden), self.reward(hidden).squeeze(-1), self.termination(hidden).squeeze(-1)


@dataclass
class WorldModelLoss:
    loss: torch.Tensor
    dynamics_mse: torch.Tensor
    reward_loss: torch.Tensor
    termination_loss: torch.Tensor


def world_model_loss(
    model: CommandWorldModel,
    state: torch.Tensor,
    commands: torch.Tensor,
    projected_z: torch.Tensor,
    next_state: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    *,
    reward_coef: float,
    termination_coef: float,
) -> WorldModelLoss:
    """Return a finite, dimension-normalized model-fitting objective."""
    predicted_state, predicted_reward, termination_logits = model(state, commands, projected_z)
    dynamics_mse = F.mse_loss(predicted_state, next_state)
    reward_loss = F.smooth_l1_loss(predicted_reward, rewards)
    termination_loss = F.binary_cross_entropy_with_logits(termination_logits, terminated.float())
    loss = dynamics_mse + float(reward_coef) * reward_loss + float(termination_coef) * termination_loss
    return WorldModelLoss(loss, dynamics_mse, reward_loss, termination_loss)


class WorldModelReplay:
    """Fixed-size GPU replay for online Stage2 transition fitting."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        command_dim: int,
        latent_dim: int,
        *,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.command_dim = int(command_dim)
        self.latent_dim = int(latent_dim)
        if self.capacity <= 0:
            raise ValueError("World-model replay capacity must be positive.")
        self.device = device
        self.states = torch.empty(self.capacity, self.state_dim, device=device)
        self.commands = torch.empty(self.capacity, self.command_dim, device=device)
        self.projected_z = torch.empty(self.capacity, self.latent_dim, device=device)
        self.next_states = torch.empty(self.capacity, self.state_dim, device=device)
        self.rewards = torch.empty(self.capacity, device=device)
        self.terminated = torch.empty(self.capacity, dtype=torch.bool, device=device)
        self.size = 0
        self.cursor = 0

    def add(
        self,
        states: torch.Tensor,
        commands: torch.Tensor,
        projected_z: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
    ) -> None:
        count = int(states.shape[0])
        expected = (count, self.state_dim)
        if states.shape != expected or next_states.shape != expected:
            raise ValueError("World-model replay state shapes do not match its contract.")
        if commands.shape != (count, self.command_dim) or projected_z.shape != (count, self.latent_dim):
            raise ValueError("World-model replay action shapes do not match its contract.")
        if rewards.shape != (count,) or terminated.shape != (count,):
            raise ValueError("World-model replay reward/termination shapes do not match its contract.")
        if count == 0:
            return
        if not all(torch.isfinite(value).all() for value in (states, commands, projected_z, next_states, rewards)):
            raise ValueError("Refusing to store non-finite world-model transitions.")
        if count >= self.capacity:
            states = states[-self.capacity :]
            commands = commands[-self.capacity :]
            projected_z = projected_z[-self.capacity :]
            next_states = next_states[-self.capacity :]
            rewards = rewards[-self.capacity :]
            terminated = terminated[-self.capacity :]
            count = self.capacity
        indices = (torch.arange(count, device=self.device) + self.cursor) % self.capacity
        self.states[indices] = states.detach()
        self.commands[indices] = commands.detach()
        self.projected_z[indices] = projected_z.detach()
        self.next_states[indices] = next_states.detach()
        self.rewards[indices] = rewards.detach()
        self.terminated[indices] = terminated.detach()
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample an empty world-model replay.")
        indices = torch.randint(self.size, (min(int(batch_size), self.size),), device=self.device)
        return (
            self.states[indices],
            self.commands[indices],
            self.projected_z[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.terminated[indices],
        )


class DynamicsErrorWindow:
    """A serializable sliding validation gate for enabling MPC."""

    def __init__(self, window_size: int, threshold: float, warmup_updates: int) -> None:
        if window_size <= 0 or threshold <= 0.0 or warmup_updates < 0:
            raise ValueError("Invalid world-model error gate configuration.")
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.warmup_updates = int(warmup_updates)
        self.values: deque[float] = deque(maxlen=self.window_size)
        self.update_count = 0

    @property
    def mean(self) -> float:
        return float(sum(self.values) / len(self.values)) if self.values else float("inf")

    @property
    def ready(self) -> bool:
        return self.update_count >= self.warmup_updates and len(self.values) == self.window_size and self.mean <= self.threshold

    def update(self, dynamics_mse: float) -> None:
        if not torch.isfinite(torch.tensor(dynamics_mse)):
            raise ValueError("World-model dynamics error must be finite.")
        self.values.append(float(dynamics_mse))
        self.update_count += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "threshold": self.threshold,
            "warmup_updates": self.warmup_updates,
            "values": list(self.values),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["window_size"]) != self.window_size or float(state["threshold"]) != self.threshold:
            raise ValueError("World-model gate checkpoint does not match the requested window or threshold.")
        if int(state["warmup_updates"]) != self.warmup_updates:
            raise ValueError("World-model gate checkpoint does not match the requested warmup updates.")
        values = [float(value) for value in state["values"]]
        if len(values) > self.window_size:
            raise ValueError("World-model gate checkpoint has an invalid error window.")
        self.values = deque(values, maxlen=self.window_size)
        self.update_count = int(state["update_count"])


@torch.no_grad()
def expert_latent_pca_basis(projected_latents: torch.Tensor, dimension: int) -> torch.Tensor:
    """Return orthonormal directions that keep MPC near expert BFM latents."""
    if projected_latents.ndim != 2:
        raise ValueError("Projected expert latents must be a rank-two tensor.")
    dimension = int(dimension)
    maximum = min(projected_latents.shape[0], projected_latents.shape[1])
    if dimension <= 0 or dimension > maximum:
        raise ValueError(f"Planner dimension must be in [1, {maximum}], got {dimension}.")
    centered = projected_latents.float() - projected_latents.float().mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(centered, q=dimension, center=False)
    return vectors[:, :dimension].T.contiguous()


@torch.no_grad()
def plan_latent_residuals(
    *,
    model: CommandWorldModel,
    states: torch.Tensor,
    commands: torch.Tensor,
    base_projected_z: torch.Tensor,
    basis: torch.Tensor,
    horizon: int,
    num_candidates: int,
    residual_scale: float,
    discount: float,
    termination_penalty: float,
) -> torch.Tensor:
    """Plan local latent residual sequences and return the first projected latent.

    Candidate zero is always the command encoder's direct BFM latent.  All
    other candidates stay on the BFM latent-radius sphere, while their motion
    is restricted to principal directions observed in expert trajectories.
    """
    if horizon <= 0 or num_candidates < 2 or residual_scale < 0.0 or not 0.0 < discount <= 1.0:
        raise ValueError("Invalid latent-residual MPC parameters.")
    batch_size, latent_dim = base_projected_z.shape
    if states.shape != (batch_size, model.state_dim) or commands.shape != (batch_size, model.command_dim):
        raise ValueError("Planner state/command shapes do not match the world model.")
    if latent_dim != model.latent_dim or basis.ndim != 2 or basis.shape[1] != latent_dim:
        raise ValueError("Planner latent basis does not match the world model.")
    if not torch.isfinite(base_projected_z).all():
        raise ValueError("Planner received non-finite projected BFM latents.")

    candidate_shape = (int(horizon), batch_size, int(num_candidates), basis.shape[0])
    residuals = torch.randn(candidate_shape, device=states.device, dtype=states.dtype)
    residuals[:, :, 0] = 0.0
    base = base_projected_z.unsqueeze(1).expand(-1, num_candidates, -1)
    planned_states = states.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, model.state_dim)
    repeated_commands = commands.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, model.command_dim)
    alive = torch.ones(batch_size * num_candidates, device=states.device, dtype=states.dtype)
    returns = torch.zeros_like(alive)
    first_action: torch.Tensor | None = None
    latent_norm = base_projected_z.norm(dim=-1, keepdim=True).clamp_min(1.0e-6).unsqueeze(1)

    for step in range(horizon):
        offset = residuals[step] @ basis
        candidates = F.normalize(base + float(residual_scale) * offset, dim=-1) * latent_norm
        flat_candidates = candidates.reshape(-1, latent_dim)
        if step == 0:
            first_action = flat_candidates.reshape(batch_size, num_candidates, latent_dim)
        planned_states, predicted_reward, termination_logits = model(planned_states, repeated_commands, flat_candidates)
        termination_probability = torch.sigmoid(termination_logits)
        step_discount = float(discount) ** step
        returns += step_discount * alive * (predicted_reward - float(termination_penalty) * termination_probability)
        alive = alive * (1.0 - termination_probability)

    if first_action is None:
        raise RuntimeError("Latent planner did not produce an action.")
    winner = returns.reshape(batch_size, num_candidates).argmax(dim=-1)
    return first_action[torch.arange(batch_size, device=states.device), winner]
