"""Configuration and runner for frozen-BFM control using the existing Pi AMP objective."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

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


def configure_backward_fast_gait(env_cfg, train_cfg, profile):
    """Use the historical command clock without changing rewards or motion data."""
    if not train_cfg["bfm_phase_observation"]:
        raise ValueError("Backward-fast gait requires explicit phase input")
    env_cfg.gait.gait_cycle = profile.default_period
    train_cfg["policy"]["project_log_std"] = True
    train_cfg["bfm_gait_profile"] = profile.metadata()
    train_cfg["run_name"] = "phase_backward_fast_scratch"


def snapshot_bfm_run(run_dir, train_cfg):
    """Keep required reproduction records inside this run's launch directory."""
    root = Path(__file__).resolve().parents[2]
    launch_dir = Path(run_dir) / "launch"
    source_dir = launch_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    relatives = [
        "legged_lab/scripts/train_bfm_amp.py",
        "legged_lab/utils/bfm_training.py",
        "legged_lab/utils/bfm_bridge.py",
        "legged_lab/utils/bfm_gait.py",
        "legged_lab/utils/bfm_distillation.py",
        "legged_lab/envs/pi/pi_22dof_cfg.py",
        "legged_lab/envs/pi/pi_cfg.py",
        "legged_lab/envs/pi/pi_env.py",
        "legged_lab/mdp/rewards.py",
        "rsl_rl/modules/bfm_command_actor_critic.py",
        "rsl_rl/runners/amp_on_policy_runner.py",
        "rsl_rl/algorithms/amp_ppo.py",
    ]
    sources = {}
    for relative in relatives:
        path = root / relative
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    motions = []
    for filename in train_cfg["amp_motion_files"]:
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        motions.append(
            {
                "file": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "weight": train_cfg["amp_motion_weights"][path.stem],
            }
        )
    with (launch_dir / "bfm_run_manifest.json").open("x") as output:
        json.dump(
            {
                "task": "walk_pi_waq_22",
                "use_phase": train_cfg["bfm_phase_observation"],
                "gait_profile": train_cfg.get("bfm_gait_profile"),
                "distillation": train_cfg.get("bfm_distillation"),
                "source_sha256": sources,
                "amp_motions": motions,
                "std_mode": "trainable; parameter projected into log_std [-5,1] before each distribution",
            },
            output,
            indent=2,
        )


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
        if train_cfg["policy"].get("project_log_std", False):
            self.bfm_metadata["latent_std_mode"] = "trainable_projected_log_std_-5_1"
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
        if train_cfg.get("bfm_gait_profile"):
            if not env.use_phase:
                raise ValueError("Variable gait requires the phase-enabled encoder")
            self.bfm_metadata["schema"] = 4
            self.bfm_metadata["gait_profile"] = copy.deepcopy(train_cfg["bfm_gait_profile"])
            self.bfm_metadata["phase_observation"]["scheduler"] = copy.deepcopy(train_cfg["bfm_gait_profile"])
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
        self.distiller = None
        self.distillation_initialization = None
        if train_cfg.get("bfm_distillation"):
            from legged_lab.utils.bfm_distillation import DualBFMTeachers

            self.distiller = DualBFMTeachers(
                self.alg.policy,
                self.bfm_metadata,
                train_cfg["bfm_distillation"],
                command_scale=getattr(env.obs_scales, "commands", 1.0),
            )
            self.distiller.assert_frozen(self.alg.optimizer)
            self.alg.policy_auxiliary_loss = self.distiller
        if env.actor.model is not None:
            frozen_ids = {id(p) for p in env.actor.model.parameters()}
            trainable_ids = {id(p) for group in self.alg.optimizer.param_groups for p in group["params"]}
            if frozen_ids & trainable_ids or any(p.requires_grad for p in env.actor.model.parameters()):
                raise RuntimeError("BFM parameters must be frozen and excluded from the PPO/AMP optimizer")

    def save(self, path, infos=None):
        checkpoint_info = {"bfm": self.bfm_metadata, "user_infos": infos}
        if self.distiller is not None:
            checkpoint_info.update(distillation=self.distiller.metadata, initialization=self.distillation_initialization)
        super().save(path, infos=checkpoint_info)

    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        if hasattr(self.env, "pop_gait_metrics"):
            for key, value in self.env.pop_gait_metrics().items():
                self.writer.add_scalar(key, value, locs["it"])
        if self.cfg.get("bfm_gait_profile") and locs["it"] % 50 == 0:
            with torch.inference_mode():
                obs, _ = self.env.get_observations()
                mean = self.alg.policy.act_inference(obs[:64])
                generator = torch.Generator(device=self.device).manual_seed(71039 + locs["it"])
                noise = torch.randn(mean.shape, device=self.device, generator=generator)
                sampled = mean + noise * self.alg.policy.log_std.clamp(-5, 1).exp()
                direction = torch.nn.functional.normalize(mean, dim=-1)
                sampled_direction = torch.nn.functional.normalize(sampled, dim=-1)
                angle = torch.rad2deg(torch.acos((direction * sampled_direction).sum(-1).clamp(-1, 1)))
                self.writer.add_scalar("Exploration/projected_z_angle_deg_mean_64", angle.mean().item(), locs["it"])
                self.writer.add_scalar("Exploration/raw_mean_norm_64", mean.norm(dim=-1).mean().item(), locs["it"])

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        info = checkpoint.get("infos") or {}
        expected_distillation = self.distiller.metadata if self.distiller is not None else None
        if load_optimizer and info.get("distillation") != expected_distillation:
            raise ValueError("Distillation resume requires identical teacher identities and schedule; inference needs no teachers")
        if info.get("bfm") != self.bfm_metadata:
            raise ValueError("Checkpoint BFM identity/control contract differs; cannot resume as the same latent policy")
        expected = self.alg.policy.state_dict()
        actual = checkpoint["model_state_dict"]
        if set(actual) != set(expected) or any(actual[k].shape != expected[k].shape for k in expected):
            raise ValueError("Command policy architecture differs; partial checkpoint loading is not allowed")
        result = super().load(path, load_optimizer)
        if load_optimizer:
            self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]["lr"])
        self.distillation_initialization = info.get("initialization")
        return result

    def initialize_from_no_phase_teacher(self):
        if self.distiller is None or self.current_learning_iteration != 0:
            raise ValueError("Teacher warm start requires a new distillation runner")
        self.distillation_initialization = self.distiller.initialize_student(self)
        return self.distillation_initialization
