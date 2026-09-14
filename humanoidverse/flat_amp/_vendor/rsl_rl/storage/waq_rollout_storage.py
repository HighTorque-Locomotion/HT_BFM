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
import torch
from rsl_rl.utils import split_and_pad_trajectories


class WaqRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None
            self.observation_histories = None
            self.next_observations = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        lin_vel_index,
        obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        self.training_type = "rl"
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape
        self.obs_history_shape = obs_history_shape
        self.lin_vel_index = lin_vel_index
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.observation_histories = torch.zeros(
            num_transitions_per_env, num_envs, *obs_history_shape, device=self.device
        )
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-08)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations
        base_lin_vel = privileged_observations[:, self.lin_vel_index : self.lin_vel_index + 3]
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        obs_history = self.observation_histories.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                base_lin_vel_batch = base_lin_vel[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None
                yield (
                    obs_batch,
                    next_obs_batch,
                    privileged_observations_batch,
                    base_lin_vel_batch,
                    obs_history_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),
                    None,
                    rnd_state_batch,
                )
