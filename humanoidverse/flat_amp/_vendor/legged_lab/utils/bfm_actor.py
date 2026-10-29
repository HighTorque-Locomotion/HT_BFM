# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived from humanoidverse/agents/nn_models.py at 0c234a2.
# Licensed under CC BY-NC 4.0, in the repository root LICENSE.
"""Load only a frozen Pi12 residual actor and its observation normalizers.

No source checkout, critic networks, training buffers or Isaac runtime is needed.
The network names and operations match the native BFM checkpoint exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .bfm_bridge import BFMContract, file_digest

ACTOR_KEYS = ("state", "last_action", "history_actor")


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), *([nn.Mish()] if activation else [])
        )

    def forward(self, x):
        return self.mlp(x)


class ResidualBlock(Block):
    def __init__(self, dim):
        super().__init__(dim, dim)

    def forward(self, x):
        return x + self.mlp(x)


def residual_embedding(input_dim, hidden_dim, layers):
    if layers < 2 or hidden_dim % 2:
        raise ValueError("Residual BFM requires >=2 embedding layers and an even hidden dimension")
    return nn.Sequential(
        Block(input_dim, hidden_dim),
        *(ResidualBlock(hidden_dim) for _ in range(layers - 2)),
        Block(hidden_dim, hidden_dim // 2),
    )


class ResidualActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, cfg):
        super().__init__()
        hidden, layers = cfg["hidden_dim"], cfg["embedding_layers"]
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden, layers)
        self.embed_s = residual_embedding(obs_dim, hidden, layers)
        self.policy = nn.Sequential(
            *(ResidualBlock(hidden) for _ in range(cfg["hidden_layers"])), Block(hidden, action_dim, False)
        )

    def forward(self, obs, z):
        embedding = torch.cat((self.embed_s(obs), self.embed_z(torch.cat((obs, z), dim=-1))), dim=-1)
        return torch.tanh(self.policy(embedding))


class BatchNormNormalizer(nn.Module):
    def __init__(self, dim, momentum):
        super().__init__()
        self._normalizer = nn.BatchNorm1d(dim, affine=False, momentum=momentum)

    def forward(self, x):
        return self._normalizer(x)


class FrozenActor(nn.Module):
    def __init__(self, config, dimensions, action_dim):
        super().__init__()
        cfg = config["archi"]["actor"]
        if cfg.get("name") not in ("actor", "residual") or cfg["model"] != "residual":
            raise ValueError("Expected a native residual BFM actor")
        if cfg["input_filter"] != {"name": "DictInputFilterConfig", "key": list(ACTOR_KEYS)}:
            raise ValueError("BFM actor input order must be state,last_action,history_actor")
        self._actor = ResidualActor(sum(dimensions.values()), config["archi"]["z_dim"], action_dim, cfg)
        self._obs_normalizer = nn.Module()
        normalizers = {}
        for key, dim in dimensions.items():
            norm_cfg = config["obs_normalizer"]["normalizers"][key]
            if norm_cfg["name"] == "BatchNormNormalizerConfig":
                normalizers[key] = BatchNormNormalizer(dim, norm_cfg["momentum"])
            elif norm_cfg["name"] == "IdentityNormalizerConfig":
                normalizers[key] = nn.Identity()
            else:
                raise ValueError(f"Unsupported BFM normalizer: {norm_cfg['name']}")
        self._obs_normalizer._normalizers = nn.ModuleDict(normalizers)

    def forward(self, observations, z):
        obs = torch.cat([self._obs_normalizer._normalizers[key](observations[key]) for key in ACTOR_KEYS], dim=-1)
        return self._actor(obs, z)


def resolve_model_dir(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path / "checkpoint/model", path / "model", path):
        if (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"No native BFM model.safetensors under {path}")


def resolve_robot_config(path, explicit=None):
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    model_dir = resolve_model_dir(path)
    for directory in (model_dir, model_dir.parent, model_dir.parent.parent):
        if (directory / "config.yaml").is_file():
            return directory / "config.yaml"
    raise FileNotFoundError("BFM config.yaml was not found; supply --bfm-robot-config")


class FrozenBFM:
    def __init__(self, path, contract, device):
        from safetensors import safe_open

        model_dir = resolve_model_dir(path)
        self.path = model_dir
        self.device = torch.device(device)
        self.contract = contract
        for directory in (model_dir, model_dir.parent, model_dir.parent.parent):
            recorded = directory / "config.yaml"
            if recorded.is_file():
                if BFMContract.from_robot_yaml(recorded).metadata() != contract.metadata():
                    raise ValueError("Selected BFM control contract differs from the checkpoint config.yaml")
                break
        config = json.loads((model_dir / "config.json").read_text())
        init = json.loads((model_dir / "init_kwargs.json").read_text())
        if (init["action_dim"] != 12 or config["archi"]["z_dim"] != contract.z_dim
                or bool(config["archi"]["norm_z"]) != contract.norm_z):
            raise ValueError("BFM checkpoint action/latent dimensions or norm_z do not match the Pi12 contract")
        dimensions = {"state": 30, "last_action": 12, "history_actor": 42 * contract.history_length}
        for key, dim in dimensions.items():
            if init["obs_space"]["spaces"][key]["shape"] != [dim]:
                raise ValueError(f"BFM checkpoint observation {key} does not match {dim}")
        self.encoder_hidden_dim = int(config["archi"]["b"]["hidden_dim"])
        self.encoder_hidden_layers = int(config["archi"]["b"]["hidden_layers"])
        self.model = FrozenActor(config, dimensions, 12)
        required = self.model.state_dict()
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as saved:
            # Ignore training-only critics and privileged normalizers, never actor tensors.
            actor_keys = {key for key in saved.keys() if key.startswith("_actor.")}
            if actor_keys != {key for key in required if key.startswith("_actor.")}:
                raise ValueError("BFM checkpoint actor tensors differ from the configured architecture")
            missing = required.keys() - set(saved.keys())
            if missing:
                raise ValueError(f"BFM checkpoint is missing frozen tensors: {sorted(missing)}")
            state = {}
            for key, expected in required.items():
                value = saved.get_tensor(key)
                if value.shape != expected.shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid BFM checkpoint tensor: {key}")
                if key.endswith("running_var") and torch.any(value < 0):
                    raise ValueError(f"Negative BFM normalization variance: {key}")
                state[key] = value
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        # Hash the full native weights plus architecture and input contract for resume validation.
        digest = hashlib.sha256()
        for name in ("model.safetensors", "config.json", "init_kwargs.json"):
            digest.update(file_digest(model_dir / name).encode())
        self.digest = digest.hexdigest()

    @torch.no_grad()
    def __call__(self, observations, z):
        return self.model(observations, z).float()
