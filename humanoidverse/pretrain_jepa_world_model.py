"""Pre-train an action-conditioned JEPA world model from ROS2 MCAP data.

The default dataset contains ``/actor_obs`` (631 floats) and ``/raw_actions``
(23 floats).  The two topics are timestamp-aligned with zero-order hold and
then converted into temporal context/target pairs.  The model is a JEPA in the
sense of LeCun et al.: an online context encoder and predictor match the
latent produced by a slowly-moving (EMA) target encoder.  It does not decode
or reconstruct the observation as its primary objective; an auxiliary
observation decoder is trained so frozen JEPA predictions can be scored by MPC.

Example:

    uv run python -m humanoidverse.pretrain_jepa_world_model \
        --data-dir dataset/world \
        --output logs/world_model/jepa.pt \
        --epochs 50 --batch-size 256 --device cuda

The loader also accepts an extracted ``.mcap`` file or an ``.npz``/``.pt``
file containing observation and action arrays.  This keeps the training loop
usable after ROS bag conversion without changing the model code.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from humanoidverse.record_actor_obs_compare import _read_mcap_actor_obs


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    timestamps_ns: np.ndarray
    name: str


def _pick_array(mapping: Any, names: tuple[str, ...], label: str) -> np.ndarray:
    if isinstance(mapping, dict) or hasattr(mapping, "files"):
        for name in names:
            if (name in mapping) if isinstance(mapping, dict) else (name in mapping.files):
                value = mapping[name]
                if isinstance(value, dict) and "data" in value:
                    value = value["data"]
                return np.asarray(value, dtype=np.float32)
    raise KeyError(f"Could not find {label}; tried keys {names}")


def _load_array_file(path: Path) -> Trajectory:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            observations = _pick_array(data, ("obs", "observation", "observations", "actor_obs"), "observations")
            actions = _pick_array(data, ("action", "actions", "raw_actions"), "actions")
            timestamps = data["timestamps_ns"] if "timestamps_ns" in data else data.get("timestamps")
    elif path.suffix in {".pt", ".pth", ".pkl", ".pickle"}:
        if path.suffix in {".pt", ".pth"}:
            mapping = torch.load(path, map_location="cpu", weights_only=False)
        else:
            with path.open("rb") as file:
                mapping = pickle.load(file)
        observations = _pick_array(mapping, ("obs", "observation", "observations", "actor_obs"), "observations")
        actions = _pick_array(mapping, ("action", "actions", "raw_actions"), "actions")
        timestamps = mapping.get("timestamps_ns", mapping.get("timestamps")) if isinstance(mapping, dict) else None
    else:
        raise ValueError(f"Unsupported array dataset format: {path}")

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected 2D observations/actions in {path}, got {observations.shape}/{actions.shape}")
    if timestamps is None:
        timestamps = np.arange(len(observations), dtype=np.int64)
    return _align_streams(observations, actions, np.asarray(timestamps, dtype=np.int64), None, path.name)


def _align_streams(
    observations: np.ndarray,
    actions: np.ndarray,
    observation_times: np.ndarray,
    action_times: np.ndarray | None,
    name: str,
) -> Trajectory:
    """Align asynchronous action messages using the latest preceding action."""
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    observation_times = np.asarray(observation_times, dtype=np.int64)
    if action_times is not None:
        action_times = np.asarray(action_times, dtype=np.int64)
        if len(action_times) != len(actions):
            raise ValueError(f"Action timestamps have length {len(action_times)}, actions have length {len(actions)}")
        order = np.argsort(action_times)
        action_times, actions = action_times[order], actions[order]
        indices = np.searchsorted(action_times, observation_times, side="right") - 1
        indices = np.clip(indices, 0, len(actions) - 1)
        actions = actions[indices]
    elif len(actions) != len(observations):
        raise ValueError("Array data with different observation/action lengths must provide timestamps_ns")

    if len(observations) != len(actions) or len(observations) < 2:
        raise ValueError(f"Trajectory {name} has incompatible or too few rows: {len(observations)}/{len(actions)}")
    finite = np.isfinite(observations).all(axis=1) & np.isfinite(actions).all(axis=1)
    if not finite.all():
        observations, actions, observation_times = observations[finite], actions[finite], observation_times[finite]
    return Trajectory(observations, actions, observation_times, name)


def _load_mcap(path: Path, obs_topic: str, action_topic: str) -> Trajectory:
    observations, observation_times = _read_mcap_actor_obs(path, obs_topic, None)
    actions, action_times = _read_mcap_actor_obs(path, action_topic, None)
    return _align_streams(observations, actions, observation_times, action_times, path.name)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in archive {archive}: {member.name}")
        tar.extractall(destination)


def load_trajectories(data_path: Path, obs_topic: str, action_topic: str) -> list[Trajectory]:
    """Load all trajectories from a file or directory, including tar.gz MCAP bags."""
    paths = sorted(path for path in data_path.rglob("*") if path.is_file()) if data_path.is_dir() else [data_path]
    trajectories: list[Trajectory] = []
    for path in paths:
        if path.name.startswith("."):
            continue
        if path.suffixes[-2:] in [[".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]]:
            with tempfile.TemporaryDirectory(prefix="jepa_bag_") as temp_dir:
                extracted = Path(temp_dir)
                _safe_extract_tar(path, extracted)
                trajectories.extend(load_trajectories(extracted, obs_topic, action_topic))
        elif path.suffix == ".mcap":
            trajectories.append(_load_mcap(path, obs_topic, action_topic))
        elif path.suffix in {".npz", ".pt", ".pth", ".pkl", ".pickle"}:
            trajectories.append(_load_array_file(path))
    if not trajectories:
        raise FileNotFoundError(f"No .mcap, .npz, .pt, or pickle datasets found under {data_path}")
    return trajectories


class TemporalJEPADataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Temporal samples with either a held-out suffix or a purged time-series fold."""

    def __init__(
        self,
        trajectories: list[Trajectory],
        context_len: int,
        prediction_horizon: int,
        split: str,
        val_fraction: float,
        max_gap_seconds: float,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        fold_index: int | None = None,
        num_folds: int = 1,
    ) -> None:
        self.context_len = int(context_len)
        self.prediction_horizon = int(prediction_horizon)
        if self.context_len <= 0 or self.prediction_horizon <= 0:
            raise ValueError("context_len and prediction_horizon must be positive")
        self.samples: list[tuple[Trajectory, int]] = []
        self._starts: list[int] = []
        self.trajectories = trajectories
        self.val_fraction = float(val_fraction)
        self.max_gap_ns = int(float(max_gap_seconds) * 1.0e9)
        self.fold_index = fold_index
        self.num_folds = int(num_folds)
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split {split!r}")
        if self.fold_index is not None and not 0 <= self.fold_index < self.num_folds:
            raise ValueError("fold_index must be in [0, num_folds)")
        required = self.context_len + self.prediction_horizon
        for trajectory in trajectories:
            count = len(trajectory.observations)
            last_start = count - required
            if last_start < 0:
                continue
            split_start = int(count * (1.0 - self.val_fraction))
            if self.fold_index is not None:
                fold_start, fold_end = _fold_bounds(count, self.fold_index, self.num_folds)
            for start in range(last_start + 1):
                target = start + required - 1
                contiguous = np.diff(trajectory.timestamps_ns[start : target + 1])
                if len(contiguous) and (contiguous < 0).any():
                    continue
                if len(contiguous) and int(contiguous.max()) > self.max_gap_ns:
                    continue
                if self.fold_index is None:
                    include = (split == "val") == (start >= split_start)
                else:
                    # A validation window must be fully contained in its time
                    # block.  A training window must be fully outside it.  The
                    # windows crossing either boundary are purged so heavily
                    # overlapping temporal samples cannot leak across splits.
                    is_val = start >= fold_start and target < fold_end
                    is_train = target < fold_start or start >= fold_end
                    include = is_val if split == "val" else is_train
                if include:
                    self.samples.append((trajectory, start))
        if not self.samples:
            raise ValueError(f"No {split} samples available; reduce context/horizon or val_fraction")
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory, start = self.samples[index]
        target = start + self.context_len + self.prediction_horizon - 1
        obs = trajectory.observations[start : start + self.context_len]
        actions = trajectory.actions[start:target]
        target_obs = trajectory.observations[target]
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
            target_obs = (target_obs - self.obs_mean) / self.obs_std
        if self.action_mean is not None:
            actions = (actions - self.action_mean) / self.action_std
        return torch.from_numpy(obs), torch.from_numpy(actions), torch.from_numpy(target_obs)


def _fold_bounds(row_count: int, fold_index: int, num_folds: int) -> tuple[int, int]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 for cross-validation")
    return row_count * fold_index // num_folds, row_count * (fold_index + 1) // num_folds


def _normalization_stats(
    trajectories: list[Trajectory],
    train_fraction: float,
    eps: float,
    fold_index: int | None = None,
    num_folds: int = 1,
) -> tuple[np.ndarray, ...]:
    if fold_index is None:
        observation_parts = [t.observations[: max(1, int(len(t.observations) * train_fraction))] for t in trajectories]
        action_parts = [t.actions[: max(1, int(len(t.actions) * train_fraction))] for t in trajectories]
    else:
        observation_parts, action_parts = [], []
        for trajectory in trajectories:
            fold_start, fold_end = _fold_bounds(len(trajectory.observations), fold_index, num_folds)
            if fold_start:
                observation_parts.append(trajectory.observations[:fold_start])
                action_parts.append(trajectory.actions[:fold_start])
            if fold_end < len(trajectory.observations):
                observation_parts.append(trajectory.observations[fold_end:])
                action_parts.append(trajectory.actions[fold_end:])
    observations = np.concatenate(observation_parts)
    actions = np.concatenate(action_parts)
    obs_mean, obs_std = observations.mean(axis=0), observations.std(axis=0)
    action_mean, action_std = actions.mean(axis=0), actions.std(axis=0)
    return obs_mean.astype(np.float32), np.maximum(obs_std, eps).astype(np.float32), action_mean.astype(np.float32), np.maximum(action_std, eps).astype(np.float32)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class JEPAWorldModel(nn.Module):
    """Action-conditioned latent predictor with an EMA target encoder."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, embedding_dim: int, context_len: int) -> None:
        super().__init__()
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.hidden_dim, self.embedding_dim, self.context_len = int(hidden_dim), int(embedding_dim), int(context_len)
        self.context_encoder = MLPEncoder(obs_dim, hidden_dim, embedding_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.action_encoder = MLPEncoder(action_dim, hidden_dim, embedding_dim)
        self.position = nn.Parameter(torch.zeros(1, context_len, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=max(1, min(8, embedding_dim // 32)),
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # JEPA is trained in latent space.  This lightweight decoder is an
        # auxiliary readout for downstream MPC scoring; it is not used by the
        # JEPA objective to reconstruct pixels or raw sensor packets.
        self.observation_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        # Privileged simulator supervision teaches this readout to infer planar
        # velocity and yaw rate from proprioceptive history and future actions.
        # base_lin_vel is a training label only; it is never part of JEPA input.
        self.motion_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[1] != self.context_len:
            raise ValueError(f"Expected observations [B, {self.context_len}, {self.obs_dim}], got {tuple(observations.shape)}")
        if actions.ndim != 3 or actions.shape[1] < self.context_len:
            raise ValueError("Action sequence must include at least one action per context observation")
        context_obs = self.context_encoder(observations)
        context_actions = self.action_encoder(actions[:, : self.context_len])
        tokens = self.temporal_encoder(context_obs + context_actions + self.position)
        context = tokens.mean(dim=1)
        future_actions = actions[:, self.context_len :]
        if future_actions.shape[1]:
            action_summary = self.action_encoder(future_actions).mean(dim=1)
        else:
            action_summary = context_actions[:, -1]
        return self.predictor(torch.cat((context, action_summary), dim=-1))

    @torch.no_grad()
    def target(self, observations: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(observations)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent to normalized observation coordinates for MPC."""
        return self.observation_decoder(latent)

    def predict_motion(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict future ``[base_vx, base_vy, yaw_rate]`` from a JEPA latent."""
        return self.motion_head(latent)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 < momentum <= 1.0:
            raise ValueError("target momentum must be in (0, 1]")
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def _jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return (2.0 - 2.0 * (prediction * target).sum(dim=-1)).mean()


def _training_loss(
    model: JEPAWorldModel,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_observation: torch.Tensor,
    observation_reconstruction_coef: float,
) -> torch.Tensor:
    latent_loss = _jepa_loss(prediction, target)
    if observation_reconstruction_coef <= 0.0:
        return latent_loss
    predicted_observation = model.decode(prediction)
    target_observation_prediction = model.decode(target.detach())
    reconstruction_loss = F.smooth_l1_loss(predicted_observation, target_observation)
    reconstruction_loss = reconstruction_loss + 0.5 * F.smooth_l1_loss(target_observation_prediction, target_observation)
    return latent_loss + float(observation_reconstruction_coef) * reconstruction_loss


def _run_epoch(
    model: JEPAWorldModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    target_momentum: float,
    grad_clip: float,
    observation_reconstruction_coef: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for observations, actions, target_obs in loader:
        observations, actions, target_obs = observations.to(device), actions.to(device), target_obs.to(device)
        with torch.set_grad_enabled(training):
            prediction = model(observations, actions)
            target = model.target(target_obs)
            loss = _training_loss(
                model,
                prediction,
                target,
                target_obs,
                observation_reconstruction_coef,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                model.update_target(target_momentum)
        batch_size = observations.shape[0]
        total += float(loss.detach()) * batch_size
        count += batch_size
    return total / max(count, 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/world"), help="MCAP/tar.gz directory or converted array file")
    parser.add_argument("--output", type=Path, default=Path("logs/world_model/jepa.pt"))
    parser.add_argument("--obs-topic", default="/actor_obs")
    parser.add_argument("--action-topic", default="/raw_actions")
    parser.add_argument("--context-len", type=int, default=8, help="Number of past observation/action tokens")
    parser.add_argument("--prediction-horizon", type=int, default=4, help="Future steps represented by the target observation")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--num-folds",
        type=int,
        default=1,
        help="Number of purged contiguous time-series folds; 1 keeps the legacy suffix validation split.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1, help="Overwrite each fold's resumable latest checkpoint every N epochs.")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--target-momentum", type=float, default=0.996)
    parser.add_argument(
        "--observation-reconstruction-coef",
        type=float,
        default=1.0,
        help="Auxiliary decoder loss coefficient required for frozen JEPA MPC observation scoring.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fold_checkpoint_paths(output: Path, fold_index: int) -> tuple[Path, Path]:
    fold_output = output.with_name(f"{output.stem}_fold{fold_index + 1:02d}{output.suffix}")
    fold_best = fold_output.with_name(f"{fold_output.stem}_best{fold_output.suffix}")
    return fold_output, fold_best


def _write_metadata(
    path: Path,
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
    folds: list[dict[str, Any]],
) -> None:
    best = min(folds, key=lambda item: item["best_val_loss"]) if folds else None
    payload = {
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "num_folds": int(args.num_folds),
        "epochs_per_fold": int(args.epochs),
        "best_fold": None if best is None else int(best["fold"]),
        "best_epoch": None if best is None else int(best["best_epoch"]),
        "best_val_loss": None if best is None else float(best["best_val_loss"]),
        "mean_fold_best_val_loss": None if not folds else float(np.mean([item["best_val_loss"] for item in folds])),
        "std_fold_best_val_loss": None if not folds else float(np.std([item["best_val_loss"] for item in folds])),
        "folds": folds,
        "config": vars(args),
        "checkpoint_components": [
            "context_encoder",
            "action_encoder",
            "temporal_encoder",
            "predictor",
            "target_encoder",
            "observation_decoder",
        ],
    }
    path.write_text(json.dumps(payload, default=str, indent=2))


def _train_fold(
    *,
    args: argparse.Namespace,
    trajectories: list[Trajectory],
    device: torch.device,
    fold_index: int | None,
) -> dict[str, Any]:
    seed = args.seed if fold_index is None else args.seed + fold_index
    _seed_everything(seed)
    obs_mean, obs_std, action_mean, action_std = _normalization_stats(
        trajectories,
        1.0 - args.val_fraction,
        1.0e-6,
        fold_index=fold_index,
        num_folds=args.num_folds,
    )
    common = dict(
        context_len=args.context_len,
        prediction_horizon=args.prediction_horizon,
        val_fraction=args.val_fraction,
        max_gap_seconds=args.max_gap_seconds,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        fold_index=fold_index,
        num_folds=args.num_folds,
    )
    train_data = TemporalJEPADataset(trajectories, split="train", **common)
    val_data = TemporalJEPADataset(trajectories, split="val", **common)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = JEPAWorldModel(obs_mean.size, action_mean.size, args.hidden_dim, args.embedding_dim, args.context_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    fold_number = 1 if fold_index is None else fold_index + 1
    fold_count = 1 if fold_index is None else args.num_folds
    print(
        f"Fold {fold_number}/{fold_count}: train={len(train_data)}, val={len(val_data)}, "
        f"obs_dim={obs_mean.size}, action_dim={action_mean.size}, device={device}"
    )
    if fold_index is None:
        fold_output = args.output
        fold_best_output = args.output.with_name(args.output.stem + "_best" + args.output.suffix)
    else:
        fold_output, fold_best_output = _fold_checkpoint_paths(args.output, fold_index)
    best_val = float("inf") if args.resume is None else float(checkpoint.get("val_loss", float("inf")))
    best_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.target_momentum,
            args.grad_clip,
            args.observation_reconstruction_coef,
        )
        with torch.no_grad():
            val_loss = _run_epoch(
                model,
                val_loader,
                device,
                None,
                args.target_momentum,
                args.grad_clip,
                args.observation_reconstruction_coef,
            )
        print(
            f"fold={fold_number:02d}/{fold_count:02d} epoch={epoch + 1:04d}/{args.epochs:04d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(args),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "trajectories": [trajectory.name for trajectory in trajectories],
            "fold_index": fold_index,
            "num_folds": args.num_folds,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "checkpoint_components": [
                "context_encoder",
                "action_encoder",
                "temporal_encoder",
                "predictor",
                "target_encoder",
                "observation_decoder",
            ],
        }
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(checkpoint, fold_output)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(checkpoint, fold_best_output)

    print(
        f"Completed fold {fold_number}/{fold_count}: best_epoch={best_epoch} "
        f"best_val_loss={best_val:.6f} checkpoint={fold_best_output}"
    )
    return {
        "fold": fold_number,
        "seed": seed,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "latest_checkpoint": str(fold_output),
        "best_checkpoint": str(fold_best_output),
        "obs_dim": int(obs_mean.size),
        "action_dim": int(action_mean.size),
    }


def main() -> None:
    args = _parse_args()
    if args.num_folds < 1:
        raise ValueError("--num-folds must be positive")
    if args.num_folds == 1 and not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be in (0, 0.5)")
    if args.num_folds > 1 and args.resume is not None:
        raise ValueError("--resume is only supported with --num-folds 1; cross-validation saves resumable checkpoints per fold")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    device = torch.device(args.device)
    trajectories = load_trajectories(args.data_dir, args.obs_topic, args.action_topic)
    print(f"Loaded {len(trajectories)} trajectories for {args.num_folds}-fold training")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, Any]] = []
    fold_indices: list[int | None] = [None] if args.num_folds == 1 else list(range(args.num_folds))
    global_best = float("inf")
    for fold_index in fold_indices:
        result = _train_fold(args=args, trajectories=trajectories, device=device, fold_index=fold_index)
        fold_results.append(result)
        if args.num_folds > 1 and result["best_val_loss"] < global_best:
            global_best = float(result["best_val_loss"])
            shutil.copy2(result["best_checkpoint"], args.output)
            print(
                f"Updated global best: fold={result['fold']} epoch={result['best_epoch']} "
                f"val_loss={global_best:.6f} checkpoint={args.output}"
            )
        _write_metadata(
            args.output.with_suffix(".json"),
            args,
            result["obs_dim"],
            result["action_dim"],
            fold_results,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved selected checkpoint: {args.output}")
    print(f"Saved cross-validation summary: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
