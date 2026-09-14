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
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


class ActorCriticWaq(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_obs_history,
        num_actions,
        encoder_hidden_dims=[128, 64, 19],
        decoder_hidden_dims=[64, 128, 48],
        latent_dim=19,
        short_lent_history=5,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticWaq.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.num_actions = num_actions
        self.short_obs = short_lent_history * num_actor_obs
        self.num_actor_obs = num_actor_obs
        mlp_input_dim_a = num_actor_obs + latent_dim
        mlp_input_dim_c = num_critic_obs
        mlp_input_dim_e = num_obs_history
        mlp_output_dim_d = num_actor_obs
        self.enc_input_dim = mlp_input_dim_e
        encoder_layers = []
        encoder_layers.append(nn.Linear(self.enc_input_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                self.cv_mu = nn.Linear(encoder_hidden_dims[l], latent_dim - 3)
                self.cv_var = nn.Linear(encoder_hidden_dims[l], latent_dim - 3)
                self.cv_vel = nn.Linear(encoder_hidden_dims[l], 3)
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder_module = nn.Sequential(*encoder_layers)
        self.mlp_output_dim = mlp_output_dim_d
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, decoder_hidden_dims[0]))
        decoder_layers.append(activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], self.mlp_output_dim))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                decoder_layers.append(activation)
        self.decoder_module = nn.Sequential(*decoder_layers)
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Encoder MLP: {self.encoder_module}")
        print(f"Decoder MLP: {self.decoder_module}")
        print(f"mu: {self.cv_mu}")
        print(f"var: {self.cv_var}")
        print(f"vel: {self.cv_vel}")
        print("DreamWaq Network Init !!!!")
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate((mod for mod in sequential if isinstance(mod, nn.Linear)))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def encode(self, observation_history):
        h = self.encoder_module(observation_history)
        vel, mu, log_var = (self.cv_vel(h), self.cv_mu(h), self.cv_var(h))
        return (vel, mu, log_var)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent):
        return self.decoder_module(latent)

    def vae_forward(self, observation_history):
        vel, mu, log_var = self.encode(observation_history)
        latent = self.reparameterize(mu, log_var)
        recons = self.decode(torch.cat([vel.detach(), latent], dim=-1))
        return (latent, recons, vel, mu, log_var)

    def net_l2_norm(self, network, mean=False):
        weights = 0
        param_num = 0
        for item in list(network.parameters()):
            if item.requires_grad:
                weights += item.pow(2).sum()
                param_num += np.prod(list(item.data.shape))
        if mean:
            param_num = max(param_num, 1)
            weights = weights / param_num
        return weights

    def update_distribution(self, observations, observation_history):
        latent, recons, vae_vel, latent_mu, latent_log_var = self.vae_forward(observation_history)
        mean = self.actor(torch.cat([vae_vel.detach(), latent, observations], dim=-1))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)
        return (latent, recons, vae_vel, latent_mu, latent_log_var)

    def act(self, observations, observation_history, **kwargs):
        latent, recons, vae_vel, latent_mu, latent_log_var = self.update_distribution(observations, observation_history)
        action = self.distribution.sample()
        return (action, latent, recons, vae_vel, latent_mu, latent_log_var)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, observation_history):
        h = self.encoder_module(observation_history)
        vel, mu = (self.cv_vel(h), self.cv_mu(h))
        latent = mu
        action = self.actor(torch.cat((vel, latent, observations), dim=-1))
        return (action, vel)

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  the runner to determine whether optimizer state can be restored.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
