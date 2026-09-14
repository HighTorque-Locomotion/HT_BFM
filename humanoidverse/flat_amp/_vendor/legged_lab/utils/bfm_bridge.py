"""Frozen H0W BFM interface. No Isaac Sim imports; usable in CPU contract tests."""

from __future__ import annotations

import hashlib
import json
import math
import sys
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
        if len(self.joint_names) != 22 or len(set(self.joint_names)) != 22:
            raise ValueError("BFM requires 22 unique H0W joint names")
        for field in (self.default_pos, self.action_scale):
            if len(field) != 22 or not np.isfinite(field).all():
                raise ValueError("BFM pose/scales must contain 22 finite values")
        if min(self.action_scale) <= 0 or min(self.action_multiplier, self.action_clip, self.control_dt) <= 0:
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
            raise ValueError("Expected H0W position control with isaac_pd_scale=False")
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
        return cls(
            joint_names=names,
            default_pos=tuple(float(robot["init_state"]["default_joint_angles"][name]) for name in names),
            action_scale=tuple(scales),
            action_multiplier=multiplier,
            action_clip=float(control["action_clip_value"]),
        )

    @property
    def state_dim(self):
        return 50

    @property
    def observation_dim(self):
        return self.state_dim + 22 + 72 * self.history_length

    def metadata(self):
        # JSON round-trip gives stable tuples/lists when comparing saved checkpoints.
        import json

        return json.loads(json.dumps(asdict(self)))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_state(model, weights_path):
    """The source loader is non-strict; never accept random actor/normalizer tensors."""
    from safetensors import safe_open

    required = {
        key: tensor for key, tensor in model.state_dict().items() if key.startswith(("_actor.", "_obs_normalizer."))
    }
    with safe_open(str(weights_path), framework="pt", device="cpu") as saved:
        missing = set(required) - set(saved.keys())
        if missing:
            raise ValueError(f"BFM checkpoint is missing frozen tensors: {sorted(missing)}")
        for key, expected in required.items():
            value = saved.get_tensor(key)
            if value.shape != expected.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid BFM checkpoint tensor: {key}")


def load_actor_bundle(model_dir, device):
    """Build only the exact actor and normalizers from a lossless FP32 export."""
    from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
    from safetensors.torch import load_file

    manifest = json.loads((model_dir / "actor_bundle.json").read_text())
    if manifest["format"] != "ht_bfm_frozen_actor_v1":
        raise ValueError("Unknown frozen BFM actor bundle format")
    for name, digest in manifest["files"].items():
        if name not in ("config.json", "init_kwargs.json", "model.safetensors"):
            raise ValueError("Unexpected actor bundle file")
        if file_digest(model_dir / name) != digest:
            raise ValueError(f"BFM actor bundle checksum mismatch: {name}")
    if set(manifest["files"]) != {
        "config.json",
        "init_kwargs.json",
        "model.safetensors",
    }:
        raise ValueError("Incomplete BFM actor bundle manifest")
    config_data = json.loads((model_dir / "config.json").read_text())
    config_data["device"] = "cpu"
    cfg = FBcprAuxModelConfig(**config_data)
    init = json.loads((model_dir / "init_kwargs.json").read_text())
    model = torch.nn.Module()
    model.cfg = cfg
    model.action_dim = int(init["action_dim"])
    model.obs_space = json_to_space(init["obs_space"])
    model._actor = cfg.archi.actor.build(model.obs_space, cfg.archi.z_dim, model.action_dim)
    model._obs_normalizer = cfg.obs_normalizer.build(model.obs_space)
    model.load_state_dict(load_file(model_dir / "model.safetensors", device="cpu"), strict=True)
    model.eval().requires_grad_(False)
    return model.to(device), manifest["source_model_sha256"]


class BFMBridge:
    """Maintain source-order observations and convert BFM actions to target PD coordinates."""

    def __init__(self, contract, target_joint_names, num_envs, device):
        self.contract = contract
        self.device = torch.device(device)
        self.target_joint_names = tuple(target_joint_names)
        if len(set(target_joint_names)) != 22 or set(target_joint_names) != set(contract.joint_names):
            raise ValueError("Target articulation and BFM must have exactly the same 22 joint names")
        self.to_bfm = torch.tensor([target_joint_names.index(n) for n in contract.joint_names], device=device)
        self.to_target = torch.argsort(self.to_bfm)
        self.default_pos = torch.tensor(contract.default_pos, device=device, dtype=torch.float32)
        self.action_scale = torch.tensor(contract.action_scale, device=device, dtype=torch.float32)
        self.last_action = torch.zeros(num_envs, 22, device=device)
        # Source history is grouped by field, newest first, excluding the current frame.
        self.history = {
            key: torch.zeros(num_envs, contract.history_length, dim, device=device)
            for key, dim in (
                ("actions", 22),
                ("base_ang_vel", 3),
                ("dof_pos", 22),
                ("dof_vel", 22),
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
        state[:, 22:44] *= 0.05
        history = obs["history_actor"].clone()
        offset = self.contract.history_length * (22 + 3 + 22)
        history[:, offset : offset + self.contract.history_length * 22] *= 0.05
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
            raise ValueError("BFM actor must return finite [num_envs, 22] actions")
        sim_action = (actor_output * self.contract.action_multiplier).clamp(
            -self.contract.action_clip, self.contract.action_clip
        )
        desired_pos = (self.default_pos + sim_action * self.action_scale)[:, self.to_target]
        if torch.any(torch.as_tensor(target_action_scale, device=self.device) <= 0):
            raise ValueError("Target action scales must be positive")
        # The target environment still applies its PD, encoder bias, delay and head mask.
        return (desired_pos - target_default_pos) / target_action_scale, sim_action

    def advance(self, previous_obs, sim_action, done_ids):
        state = previous_obs["state"]
        fields = {
            "actions": previous_obs["last_action"],
            "base_ang_vel": state[:, -3:],
            "dof_pos": state[:, :22],
            "dof_vel": state[:, 22:44],
            "projected_gravity": state[:, 44:47],
        }
        for key, history in self.history.items():
            history[:, 1:] = history[:, :-1].clone()
            history[:, 0] = fields[key]
            history[done_ids] = 0
        self.last_action.copy_(sim_action)
        self.last_action[done_ids] = 0


class FrozenBFM:
    """Load a native HT_BFM checkpoint or an exported actor including its normalizers."""

    def __init__(self, path, contract, device, source_repo=None):
        self.path = Path(path).expanduser().resolve()
        self.contract = contract
        self.device = torch.device(device)
        self.model = None
        self.session = None
        if not self.path.exists():
            raise FileNotFoundError(f"BFM weights do not exist: {self.path}")
        if self.path.suffix == ".onnx":
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda":
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("CUDA BFM rollouts require onnxruntime-gpu or a native BFM checkpoint")
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": self.device.index or 0}))
            self.session = ort.InferenceSession(str(self.path), providers=providers)
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            expected = contract.observation_dim + contract.z_dim
            if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape[-1] != expected or outputs[0].shape[-1] != 22:
                raise ValueError(
                    f"Expected BFM ONNX [batch,{expected}] -> [batch,22]; a locomotion policy.onnx cannot be used"
                )
            if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
                raise ValueError("BFM ONNX must use float32 inputs/outputs")
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
            self.fixed_batch = inputs[0].shape[0] if isinstance(inputs[0].shape[0], int) else None
            self.digest = file_digest(self.path)
        else:
            if source_repo is not None:
                source_repo = Path(source_repo).expanduser().resolve()
                if not (source_repo / "humanoidverse/agents/fb_cpr_aux/model.py").is_file():
                    raise ValueError("--bfm-source must point at the HT_BFM repository")
                sys.path.insert(0, str(source_repo))
            from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModel

            model_dir = self.path
            for candidate in (
                self.path / "checkpoint/model",
                self.path / "model",
                self.path,
            ):
                if (candidate / "model.safetensors").is_file():
                    model_dir = candidate
                    break
            for directory in (model_dir, model_dir.parent, model_dir.parent.parent):
                saved_config = directory / "config.yaml"
                if saved_config.is_file():
                    recorded_contract = BFMContract.from_robot_yaml(saved_config)
                    if recorded_contract.metadata() != contract.metadata():
                        raise ValueError(f"Selected BFM control contract differs from checkpoint {saved_config}")
                    break
            # Target critics are training-only; do not allocate their checkpoint copies.
            if (model_dir / "actor_bundle.json").is_file():
                self.model, self.digest = load_actor_bundle(model_dir, self.device)
            else:
                self.model = FBcprAuxModel.load(model_dir, device=self.device.type, strict=False).to(self.device)
                self.digest = file_digest(model_dir / "model.safetensors")
            validate_frozen_state(self.model, model_dir / "model.safetensors")
            if self.model.action_dim != 22 or self.model.cfg.archi.z_dim != contract.z_dim:
                raise ValueError("BFM checkpoint action/latent dimensions do not match the H0W contract")
            if bool(self.model.cfg.archi.norm_z) != contract.norm_z:
                raise ValueError("BFM checkpoint norm_z does not match the contract")
            self.model.eval().requires_grad_(False)
            # Verify actor filters and observation widths before the first simulation step.
            required = {
                "state": 50,
                "last_action": 22,
                "history_actor": 72 * contract.history_length,
            }
            for key, dim in required.items():
                if self.model.obs_space[key].shape != (dim,):
                    raise ValueError(f"BFM checkpoint observation {key} does not match {dim}")

    @torch.no_grad()
    def __call__(self, observations, z):
        if self.model is not None:
            # Some source models also store privileged normalizers used only by critics.
            normalizers = self.model._obs_normalizer._normalizers
            normalized = {key: normalizers[key](value) for key, value in observations.items()}
            return self.model._actor(normalized, z, self.model.cfg.actor_std).mean.float()
        batch = z.shape[0]
        if self.fixed_batch is not None and batch != self.fixed_batch:
            raise ValueError(
                f"BFM ONNX has fixed batch {self.fixed_batch}; re-export with dynamic batch for {batch} environments"
            )
        x = torch.cat((*observations.values(), z), dim=-1).contiguous()
        if self.device.type == "cpu":
            return torch.from_numpy(self.session.run([self.output_name], {self.input_name: x.numpy()})[0])
        # Bind GPU tensors directly; do not transfer a 4096-environment rollout through NumPy each step.
        output = torch.empty(batch, 22, device=self.device)
        binding = self.session.io_binding()
        index = self.device.index or 0
        binding.bind_input(self.input_name, "cuda", index, np.float32, tuple(x.shape), x.data_ptr())
        binding.bind_output(
            self.output_name,
            "cuda",
            index,
            np.float32,
            tuple(output.shape),
            output.data_ptr(),
        )
        torch.cuda.synchronize(self.device)
        self.session.run_with_iobinding(binding)
        torch.cuda.synchronize(self.device)
        return output


class BFMLatentEnv:
    """Expose 256 PPO actions while the wrapped reward manager still sees 22 joint actions."""

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
        if frame.shape != (self.env.num_envs, 77):
            raise ValueError("Expected Pi 22DoF frame [N,77], ending in phase sin/cos")
        scales = self.env.obs_scales
        q = frame[:, 9:31] / scales.joint_pos + data.default_joint_pos
        dq = frame[:, 31:53] / scales.joint_vel + data.default_joint_vel
        self._observations = self.bridge.observe(
            q,
            dq,
            frame[:, 3:6] / scales.projected_gravity,
            frame[:, :3] / scales.ang_vel,
        )
        self._actor_obs = self.bridge.encoder_observation(
            self._observations,
            frame[:, 6:9],
            phase_sin_cos=frame[:, 75:77] if self.use_phase else None,
        )
        # Append BFM memory to the target's privileged critic so its value is conditioned on latent-control history.
        extras["observations"]["critic"] = torch.cat((extras["observations"]["critic"], self._actor_obs), dim=-1)
        self._extras = extras
        return self._actor_obs, extras

    def get_observations(self):
        if self._observations is None:
            _, extras = self.env.get_observations()
            return self._refresh(extras)
        return self._actor_obs, self._extras

    @torch.no_grad()
    def step(self, raw_z):
        self.get_observations()
        output = self.actor(self._observations, self.bridge.project_z(raw_z))
        action, sim_action = self.bridge.target_actions(
            output, self.env.robot.data.default_joint_pos, self.env.action_scale
        )
        _, reward, done, extras = self.env.step(action)
        # Raw source action (32*x), before target delay/head masking, is the BFM feedback convention.
        self.bridge.advance(self._observations, sim_action, self.env.reset_env_ids)
        obs, extras = self._refresh(extras)
        return obs, reward, done, extras


def with_terminal_amp_capture(env_class):
    """Capture expert-feature transitions before PiDof22Env.reset mutates physical state."""

    class TerminalAMPEnv(env_class):
        def step(self, actions):
            self._capture_terminal_amp = True
            try:
                return super().step(actions)
            finally:
                self._capture_terminal_amp = False

        def reset(self, env_ids):
            if getattr(self, "_capture_terminal_amp", False):
                self.extras["terminal_amp_states"] = self.get_amp_obs_for_expert_trans()[env_ids].clone()
            return super().reset(env_ids)

    return TerminalAMPEnv
