"""Gaussian command-to-z actor, with the target framework's privileged PPO critic."""

import math

from torch import nn, no_grad
from torch.distributions import Normal

from .actor_critic import ActorCritic


class BFMCommandActorCritic(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        hidden_dim=256,
        hidden_layers=1,
        project_log_std=False,
        **kwargs,
    ):
        kwargs.update(
            noise_std_type="log",
            init_noise_std=math.exp(-1.5),
        )
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        layers = [
            nn.Linear(num_actor_obs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        ]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.actor = nn.Sequential(*layers)
        self.project_log_std = bool(project_log_std)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.project_log_std:
            # The historical dual-teacher task projects the trainable parameter,
            # avoiding zero clamp gradients after std leaves the allowed range.
            with no_grad():
                self.log_std.clamp_(-5.0, 1.0)
        log_std = self.log_std if self.project_log_std else self.log_std.clamp(-5.0, 1.0)
        self.distribution = Normal(mean, log_std.exp().expand_as(mean))
