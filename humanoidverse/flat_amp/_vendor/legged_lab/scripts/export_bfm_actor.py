"""Losslessly extract frozen BFM inference tensors and verify against the full checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from legged_lab.utils.bfm_bridge import BFMBridge, BFMContract, FrozenBFM, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-weights", required=True)
    parser.add_argument("--bfm-source", required=True)
    parser.add_argument("--bfm-robot-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.bfm_weights).expanduser().resolve()
    model_dir = next(
        p for p in (source / "checkpoint/model", source / "model", source) if (p / "model.safetensors").is_file()
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as saved:
        state = {key: saved.get_tensor(key) for key in saved.keys() if key.startswith(("_actor.", "_obs_normalizer."))}
    # Keep the original tensor dtype and values; no quantization or casting.
    save_file(state, str(output / "model.safetensors"))
    for name in ("config.json", "init_kwargs.json"):
        shutil.copy2(model_dir / name, output / name)
    shutil.copy2(args.bfm_robot_config, output / "config.yaml")
    manifest = {
        "format": "ht_bfm_frozen_actor_v1",
        "source_model_sha256": file_digest(model_dir / "model.safetensors"),
        "files": {
            name: file_digest(output / name) for name in ("model.safetensors", "config.json", "init_kwargs.json")
        },
    }
    (output / "actor_bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    contract = BFMContract.from_robot_yaml(args.bfm_robot_config)
    full = FrozenBFM(model_dir, contract, "cpu", args.bfm_source)
    compact = FrozenBFM(output, contract, "cpu", args.bfm_source)
    bridge = BFMBridge(contract, list(contract.joint_names), 16, "cpu")
    torch.manual_seed(42)
    max_error = 0.0
    for _ in range(8):
        observations = {
            "state": torch.randn(16, 50),
            "last_action": torch.randn(16, 22),
            "history_actor": torch.randn(16, 288),
        }
        z = bridge.project_z(torch.randn(16, contract.z_dim))
        expected, actual = full(observations, z), compact(observations, z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        max_error = max(max_error, (actual - expected).abs().max().item())
    assert full.digest == compact.digest
    print(
        json.dumps(
            {
                "output": str(output),
                "max_absolute_error": max_error,
                "tensor_count": len(state),
                "bytes": (output / "model.safetensors").stat().st_size,
                "source_sha256": full.digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
