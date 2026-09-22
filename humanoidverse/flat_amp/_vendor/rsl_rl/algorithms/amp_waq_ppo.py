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
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.modules import ActorCriticWaq
from rsl_rl.storage import ReplayBuffer, WaqRolloutStorage
from rsl_rl.utils import string_to_callable
import torch.nn.functional as F


class AMPWAQPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCriticWaq
    "The actor critic module."

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        beta=0.0,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        self.rnd = None
        self.rnd_optimizer = None
        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            if symmetry_cfg["use_data_augmentation"] and (not callable(symmetry_cfg["data_augmentation_func"])):
                raise ValueError(
                    f"Data augmentation enabled but the function is not callable: {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None
        self.amploss_coef = 1.0
        self.min_std = min_std
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = WaqRolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.policy = policy
        self.policy.to(self.device)
        params = [
            {"params": self.policy.parameters(), "name": "policy"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 0.001, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 0.1, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.storage: WaqRolloutStorage = None
        self.transition = WaqRolloutStorage.Transition()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.beta = beta
        print(f"Beta for VAE loss: {self.beta}")
        self.num_vae_module_substeps = 1

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        lin_vel_index,
        actor_obs_shape,
        critic_obs_shape,
        actor_obs_history_shape,
        actions_shape,
    ):
        rnd_state_shape = None
        self.storage = WaqRolloutStorage(
            "rl",
            num_envs,
            num_transitions_per_env,
            lin_vel_index,
            actor_obs_shape,
            critic_obs_shape,
            actor_obs_history_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, obs_history, amp_obs):
        self.transition.actions = self.policy.act(obs, obs_history)[0].detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.amp_transition.observations = amp_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, next_obs_no_noise):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs_no_noise
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_body_vel_loss = 0
        mean_recons_loss = 0
        mean_kld_loss = 0
        mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                next_obs_batch,
                critic_obs_batch,
                base_lin_vel_batch,
                obs_history_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample
            num_aug = 1
            original_batch_size = obs_batch.shape[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-08)
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                obs_history_batch, _ = data_augmentation_func(
                    obs=obs_history_batch, actions=None, env=self.symmetry["_env"], obs_type="policy_history"
                )
                base_lin_vel_batch, _ = data_augmentation_func(
                    obs=base_lin_vel_batch, actions=None, env=self.symmetry["_env"], obs_type="base_lin_vel"
                )
                next_obs_batch, _ = data_augmentation_func(
                    obs=next_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
            latent, recons, vae_vel, latent_mu, latent_log_var = self.policy.update_distribution(
                obs_batch, obs_history_batch
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-05)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-05, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(0.01, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            body_vel_loss = F.mse_loss(vae_vel, base_lin_vel_batch)
            recons_loss = F.mse_loss(
                torch.cat((recons[:, 0:6], recons[:, 9 : 9 + 2 * self.policy.num_actions]), dim=1),
                torch.cat((next_obs_batch[:, 0:6], next_obs_batch[:, 9 : 9 + 2 * self.policy.num_actions]), dim=1),
            )
            kld_loss = torch.mean(-0.5 * torch.sum(1 + latent_log_var - latent_mu.pow(2) - latent_log_var.exp(), 1), 0)
            vae_loss = body_vel_loss + recons_loss + self.beta * kld_loss
            loss = (
                vae_loss
                + surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + 5e-06 * self.policy.net_l2_norm(self.policy.encoder_module)
                + 5e-06 * self.policy.net_l2_norm(self.policy.actor)
                + 5e-06 * self.policy.net_l2_norm(self.policy.critic)
                + 5e-06 * self.policy.net_l2_norm(self.policy.cv_vel)
                + 5e-06 * self.policy.net_l2_norm(self.policy.cv_mu)
                + 5e-06 * self.policy.net_l2_norm(self.policy.cv_var)
            )
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    obs_history_batch, _ = data_augmentation_func(
                        obs=obs_history_batch, actions=None, env=self.symmetry["_env"], obs_type="policy_history"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(
                    obs_batch.detach().clone(), obs_history_batch.detach().clone()
                )[0]
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
            loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            mean_body_vel_loss += body_vel_loss.item()
            mean_recons_loss += recons_loss.item()
            mean_kld_loss += kld_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_body_vel_loss /= num_updates * self.num_vae_module_substeps
        mean_recons_loss /= num_updates * self.num_vae_module_substeps
        mean_kld_loss /= num_updates * self.num_vae_module_substeps
        self.storage.clear()
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
            "vae_body_vel": mean_body_vel_loss,
            "vae_recons": mean_recons_loss,
            "vae_kld": mean_kld_loss,
        }
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    "\n    Helper functions\n    "

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        all_params = self.policy.parameters()
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
