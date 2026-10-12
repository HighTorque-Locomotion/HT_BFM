"""Frozen dual-teacher latent distillation on student PPO observations, not teacher rollouts."""

from __future__ import annotations

import copy
import hashlib
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from legged_lab.utils.bfm_gait import BackwardFastGait, command_masks


@dataclass(frozen=True)
class DistillationSchedule:
    coefficient: float = 5.0
    final_coefficient: float = 0.5
    hold_iterations: int = 1000
    decay_iterations: int = 10000
    initial_raw_norm: float = 16.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Distillation settings must be finite and nonnegative")
        if self.decay_iterations < 1 or self.initial_raw_norm <= 0 or self.coefficient <= 0:
            raise ValueError("Distillation coefficient, raw norm and decay duration must be positive")
        if self.final_coefficient > self.coefficient:
            raise ValueError("Distillation final coefficient must not exceed its initial value")

    def weight(self, iteration):
        progress = min(max((iteration - self.hold_iterations) / self.decay_iterations, 0), 1)
        return self.coefficient + (self.final_coefficient - self.coefficient) * progress


def _read_teacher(path, expected_metadata, schema):
    raw = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    metadata = (checkpoint.get("infos") or {}).get("bfm") or {}
    if metadata.get("schema") != schema:
        raise ValueError(f"Teacher requires original schema {schema}: {path}")
    for key in ("contract", "actor_sha256", "target_joint_names", "target_action_scale", "target_default_pos",
                "actuator_profile", "latent_projection"):
        if metadata.get(key) != expected_metadata.get(key):
            raise ValueError(f"Teacher {key} differs from student: {path}")
    expected_observation = "command,state,last_action,history_actor" + (",phase_sin_cos" if schema == 2 else "")
    if metadata.get("actor_observation") != expected_observation or "obs_norm_state_dict" in checkpoint:
        raise ValueError(f"Teacher observation encoding/normalization differs: {path}")
    if schema == 2:
        phase = metadata.get("phase_observation") or {}
        student_phase = expected_metadata.get("phase_observation") or {}
        for key, value in phase.items():
            if student_phase.get(key) != value:
                raise ValueError(f"Teacher fixed-phase contract differs at {key}: {path}")
        if phase.get("gait_cycle") != 0.8:
            raise ValueError("Phase teacher must use the original 0.8s gait")
    state = checkpoint["model_state_dict"]
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError(f"Non-finite teacher policy: {path}")
    return checkpoint, {"checkpoint_sha256": hashlib.sha256(raw).hexdigest(), "iteration": int(checkpoint["iter"]), "schema": schema}


def _frozen_encoder(student_actor, checkpoint, input_dim):
    # Copying/slicing consumes no random numbers and does not perturb the student's seed.
    actor = copy.deepcopy(student_actor)
    actor[0].in_features = input_dim
    actor[0].weight = nn.Parameter(actor[0].weight[:, :input_dim].detach().clone(), requires_grad=False)
    state = {key.removeprefix("actor."): value for key, value in checkpoint["model_state_dict"].items() if key.startswith("actor.")}
    actor.load_state_dict(state, strict=True)
    return actor.eval().requires_grad_(False)


class DualBFMTeachers:
    """No-phase teacher for backward/standing; phase teacher for all other moving commands."""

    def __init__(self, student, expected_metadata, settings, command_scale=1.0):
        self.schedule = DistillationSchedule(**settings.get("schedule", {}))
        self.command_scale = float(command_scale)
        if not math.isfinite(self.command_scale) or self.command_scale <= 0:
            raise ValueError("Expected a positive scalar command observation scale")
        profile = expected_metadata.get("gait_profile")
        if expected_metadata.get("schema") != 4 or profile is None:
            raise ValueError("Dual-teacher student requires the explicit variable-phase schema 4")
        self.gait = BackwardFastGait(**{key: profile[key] for key in asdict(BackwardFastGait())})
        self.no_phase_checkpoint, no_phase_info = _read_teacher(settings["no_phase_teacher"], expected_metadata, 1)
        phase_checkpoint, phase_info = _read_teacher(settings["phase_teacher"], expected_metadata, 2)
        self.no_phase_actor = _frozen_encoder(student.actor, self.no_phase_checkpoint, 363)
        self.phase_actor = _frozen_encoder(student.actor, phase_checkpoint, 365)
        self.metadata = {
            "version": 1, "no_phase_teacher": no_phase_info, "phase_teacher": phase_info,
            "schedule": asdict(self.schedule), "command_scale": self.command_scale,
            "selector": "no-phase for vx<=-0.1 or standing; phase for every other moving command",
            "loss": "mean of available backward/standing/other group mean cosine losses on normalized z",
            "rollouts": "student Gaussian only; teachers queried on student PPO minibatch observations",
            "initialization": "no-phase encoder/critic/AMP; two added input columns zero; positive final-layer rescale; fresh optimizer/std",
        }

    def groups(self, obs):
        if obs.ndim != 2 or obs.shape[1] != 365:
            raise ValueError("Distillation needs [batch,365] student observations")
        masks = command_masks(obs[:, :3] / self.command_scale, self.gait)
        return {"backward": masks["backward"], "standing": masks["standing"],
                "phase_motion": ~(masks["backward"] | masks["standing"])}

    @torch.no_grad()
    def targets(self, obs):
        groups = self.groups(obs)
        phase_mask = groups["phase_motion"]
        # Each target is one teacher's direction, never an average of raw latents.
        no_phase = F.normalize(self.no_phase_actor(obs[:, :363]), dim=-1)
        phase = F.normalize(self.phase_actor(obs), dim=-1)
        return torch.where(phase_mask[:, None], phase, no_phase), groups

    def __call__(self, observations, student_mean, iteration):
        target, groups = self.targets(observations.detach())
        error = (1 - (F.normalize(student_mean, dim=-1) * target).sum(-1)).clamp_min(0)
        group_losses, active, stats = [], [], {}
        for name, mask in groups.items():
            count = mask.sum()
            value = (error * mask).sum() / count.clamp_min(1)
            group_losses.append(value)
            active.append((count > 0).to(error.dtype))
            stats[f"cosine_{name}"] = value.detach()
            stats[f"fraction_{name}"] = mask.float().mean()
        cosine = torch.stack(group_losses).sum() / torch.stack(active).sum().clamp_min(1)
        coefficient = self.schedule.weight(iteration)
        loss = coefficient * cosine
        stats.update(cosine=cosine.detach(), weighted=loss.detach(), coefficient=error.new_tensor(coefficient))
        return loss, stats

    @torch.no_grad()
    def initialize_student(self, runner):
        """Warm start, not resume: preserve deterministic projected actions, reset optimizer/std."""
        if runner.alg.optimizer.state:
            raise ValueError("Teacher initialization requires a fresh optimizer, not an already-trained runner")
        student = runner.alg.policy
        source = self.no_phase_checkpoint["model_state_dict"]
        target = student.state_dict()
        if set(source) != set(target):
            raise ValueError("Unexpected teacher/student policy key mismatch")
        for key, value in source.items():
            if key == "log_std":
                target[key].fill_(-1.5)
            elif key in ("actor.0.weight", "critic.0.weight"):
                if target[key].shape != (value.shape[0], value.shape[1] + 2):
                    raise ValueError(f"Only two appended phase columns may differ: {key}")
                target[key].zero_()
                target[key][:, :value.shape[1]].copy_(value)
            else:
                if target[key].shape != value.shape:
                    raise ValueError(f"Unexpected warm-start shape mismatch: {key}")
                target[key].copy_(value)
        student.load_state_dict(target, strict=True)
        runner.alg.discriminator.load_state_dict(self.no_phase_checkpoint["discriminator_state_dict"], strict=True)
        normalizer = self.no_phase_checkpoint["amp_normalizer"]
        if normalizer.mean.shape != runner.alg.amp_normalizer.mean.shape:
            raise ValueError("Teacher and student AMP observation dimensions differ")
        runner.alg.amp_normalizer = copy.deepcopy(normalizer)
        obs, _ = runner.env.get_observations()
        before = student.act_inference(obs[:256])
        norm = float(before.norm(dim=-1).median())
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError("Cannot calibrate raw latent scale from a zero/non-finite teacher")
        gain = self.schedule.initial_raw_norm / norm
        student.actor[-1].weight.mul_(gain)
        student.actor[-1].bias.mul_(gain)
        after = student.act_inference(obs[:256])
        error = float((F.normalize(before, dim=-1) - F.normalize(after, dim=-1)).abs().max())
        if error > 1e-5:
            raise RuntimeError("Positive raw-z rescaling changed projected action directions")
        return {"mode": "teacher_warm_start_not_resume", "source": self.metadata["no_phase_teacher"],
                "initial_source_raw_norm_median": norm, "positive_output_gain": gain,
                "initial_raw_norm_target": self.schedule.initial_raw_norm, "projected_direction_max_abs_error": error,
                "fresh_optimizer": True, "initial_log_std": -1.5, "new_run_iteration": 0}

    def assert_frozen(self, optimizer):
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for teacher in (self.no_phase_actor, self.phase_actor):
            if any(p.requires_grad or id(p) in optimized for p in teacher.parameters()):
                raise RuntimeError("Teacher encoders must remain frozen and outside PPO/AMP optimizer")
