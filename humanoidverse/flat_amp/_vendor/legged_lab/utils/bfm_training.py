"""Configuration and runner for frozen-BFM control using the existing Pi AMP objective."""

from __future__ import annotations

import copy

import torch

from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


def make_bfm_train_cfg(base_cfg, z_dim=256, use_phase=True):
    cfg = copy.deepcopy(base_cfg)
    base_policy = cfg["policy"]
    cfg["policy"] = {
        "class_name": "BFMCommandActorCritic",
        "hidden_dim": 256,
        "hidden_layers": 1,
        "actor_hidden_dims": [256],
        "critic_hidden_dims": base_policy["critic_hidden_dims"],
        "activation": base_policy["activation"],
        "num_critic_heads": 1,
    }
    cfg["algorithm"]["class_name"] = "AMPPPO"
    if (
        not cfg["algorithm"].get("use_amp", True)
        or cfg["algorithm"].get("freeze_amp", False)
        or cfg["algorithm"].get("dagger_only", False)
        or cfg["algorithm"].get("dagger_loss_coef", 0.0) != 0
    ):
        raise ValueError("Expected walk_pi_waq_22 online AMP/PPO, without frozen AMP or DAgger")
    # AMPWAQDEPTHPPO adds depth/DAgger switches absent from the common AMPPPO core.
    for key in (
        "use_amp",
        "freeze_amp",
        "dagger_loss_coef",
        "dagger_loss_min_coef",
        "dagger_loss_decay",
        "dagger_only",
    ):
        cfg["algorithm"].pop(key, None)
    # Mirror permutation for 22 motor actions is undefined for 256 latent coordinates.
    cfg["algorithm"]["symmetry_cfg"] = None
    cfg["min_normalized_std"] = [0.0] * z_dim
    cfg["runner_class_name"] = "AmpOnPolicyRunner"
    cfg["resume"] = False
    cfg["experiment_name"] += "_bfm"
    cfg["run_name"] = "frozen_bfm_ht_phase" if use_phase else "frozen_bfm_ht"
    cfg["bfm_phase_observation"] = bool(use_phase)
    # These auxiliary models assume joint-action dimensions; they are not part of walk_pi_waq_22.
    for key in ("foothold_imagination_cfg", "visual_pretrain_cfg"):
        if (cfg["algorithm"].get(key) or {}).get("enable", False):
            raise ValueError(f"{key} is not supported by the frozen-BFM command policy")
    if cfg.get("empirical_normalization", False):
        raise ValueError("BFM bridge requires empirical_normalization=False, as in walk_pi_waq_22")
    return cfg


class BFMOnPolicyRunner(AmpOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.bfm_metadata = {
            "schema": 1,
            "contract": env.bridge.contract.metadata(),
            "actor_sha256": env.actor.digest,
            "target_joint_names": list(env.bridge.target_joint_names),
            "target_action_scale": torch.as_tensor(env.action_scale).detach().cpu().tolist(),
            "target_default_pos": env.robot.data.default_joint_pos[0].detach().cpu().tolist(),
            "actor_observation": "command,state,last_action,history_actor",
            "latent_projection": "sqrt(z_dim)*normalize(raw_z)",
            "actuator_profile": train_cfg.get("bfm_actuator_profile", "ht"),
        }
        if env.use_phase:
            gait = env.cfg.gait
            self.bfm_metadata.update(
                schema=2,
                actor_observation="command,state,last_action,history_actor,phase_sin_cos",
                phase_observation={
                    "encoding": "sin(2*pi*phase),cos(2*pi*phase)",
                    "source": "current_actor_obs_add_nosie[:,75:77]",
                    "encoder_indices": [363, 364],
                    "gait_cycle": float(gait.gait_cycle),
                    "lin_command_threshold": float(gait.lin_command_threshold),
                    "yaw_command_threshold": float(gait.yaw_command_threshold),
                    "standing_phase": 0.0,
                    "reset_phase_offsets": [0.0, 0.5],
                    "right_leg_phase_offset": 0.5,
                },
            )
        if train_cfg.get("bfm_phase_observation", env.use_phase) != env.use_phase:
            raise ValueError("Runner and environment disagree on phase observation")
        super().__init__(env, train_cfg, log_dir, device)
        print(
            f"[BFM] encoder_input={self.alg.policy.actor[0].in_features}, "
            f"explicit_phase={env.use_phase}, frozen_actor_input=unchanged, "
            f"checkpoint_schema={self.bfm_metadata['schema']}",
            flush=True,
        )
        self.logger_type = self.cfg.get("logger", "tensorboard")
        if env.actor.model is not None:
            frozen_ids = {id(p) for p in env.actor.model.parameters()}
            trainable_ids = {id(p) for group in self.alg.optimizer.param_groups for p in group["params"]}
            if frozen_ids & trainable_ids or any(p.requires_grad for p in env.actor.model.parameters()):
                raise RuntimeError("BFM parameters must be frozen and excluded from the PPO/AMP optimizer")

    def save(self, path, infos=None):
        super().save(path, infos={"bfm": self.bfm_metadata, "user_infos": infos})

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        info = checkpoint.get("infos") or {}
        if info.get("bfm") != self.bfm_metadata:
            raise ValueError(
                "Checkpoint BFM identity/control contract differs; cannot resume as the same latent policy"
            )
        expected = self.alg.policy.state_dict()
        actual = checkpoint["model_state_dict"]
        if set(actual) != set(expected) or any(actual[k].shape != expected[k].shape for k in expected):
            raise ValueError("Command policy architecture differs; partial checkpoint loading is not allowed")
        result = super().load(path, load_optimizer)
        if load_optimizer:
            self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]["lr"])
        return result
