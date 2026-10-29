"""Compare exported ONNX against actual checkpoint outputs recorded in Isaac playback.

python -m humanoidverse.flat_amp.compare_bfm_onnx --trace TRACE --export-dir EXPORTED
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_z(raw):
    return np.float32(np.sqrt(raw.shape[-1])) * (
        raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), np.float32(1e-12))
    )


def action_targets(action, meta):
    contract = meta["contract"]
    order = [contract["joint_names"].index(name) for name in meta["target_joint_names"]]
    sim_action = np.clip(action * np.float32(contract["action_multiplier"]), -contract["action_clip"], contract["action_clip"])
    desired = (np.asarray(contract["default_pos"], dtype=np.float32)
               + sim_action * np.asarray(contract["action_scale"], dtype=np.float32))[:, order]
    default = np.asarray(meta["target_default_pos"], dtype=np.float32)
    scale = np.asarray(meta["target_action_scale"], dtype=np.float32)
    # Playback flattens samples in step, env order; broadcast each environment's pose.
    if default.ndim == 2:
        default = np.tile(default, (len(action) // len(default), 1))
    target_action = (desired - default) / scale
    return target_action, np.clip(target_action, -meta["clip_actions"], meta["clip_actions"]) * scale + default


def error_stats(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite inference result")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "reference_max_abs": float(np.abs(expected).max()),
        "per_dimension_max_abs": absolute.max(axis=0).tolist(),
        "worst_sample_dimension": list(map(int, np.unravel_index(absolute.argmax(), absolute.shape))),
    }


def compare(trace, export_dir, batch_size=1):
    meta = json.loads((trace / "metadata.json").read_text())
    interface = json.loads((export_dir / "interface.json").read_text())
    if interface["source_checkpoint_sha256"] != meta["checkpoint_sha256"]:
        raise ValueError("ONNX export and simulation trace came from different checkpoints")
    if interface["bfm_metadata"]["actor_sha256"] != meta["actor_sha256"]:
        raise ValueError("ONNX export and simulation used different frozen BFM weights")
    if (interface["bfm_metadata"]["contract"] != meta["contract"]
            or interface["bfm_metadata"]["target_joint_names"] != meta["target_joint_names"]):
        raise ValueError("ONNX and trace joint/control contracts differ")
    for key in ("target_default_pos", "target_action_scale"):
        if not np.allclose(meta[key], interface["bfm_metadata"][key], atol=1e-7, rtol=0):
            raise ValueError(f"ONNX and trace differ: {key}")
    if meta["clip_actions"] != interface["control"]["clip_actions"]:
        raise ValueError("ONNX and trace action clipping differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = {}
    for name in ("encoder", "bfm"):
        model = interface["models"][name]
        path = export_dir / model["file"]
        if sha256(path) != model["sha256"]:
            raise ValueError(f"{name} ONNX hash differs from interface.json")
        sessions[name] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    chunks = sorted(trace.glob("chunk_*.npz"))
    if not chunks:
        raise ValueError("No completed simulation samples")
    arrays = {}
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                value = data[key]
                arrays.setdefault(key, []).append(value.reshape((-1, *value.shape[2:])))
    data = {key: np.concatenate(values) for key, values in arrays.items()}
    samples = len(data["raw_z"])

    def run(name, values):
        session = sessions[name]
        input_name = session.get_inputs()[0].name
        results = []
        for start in range(0, len(values), batch_size):
            results.append(session.run(None, {input_name: values[start:start + batch_size]})[0])
        return np.concatenate(results)

    raw = run("encoder", data["encoder_obs"])
    z = project_z(raw)
    isolated = run("bfm", data["bfm_input"])
    chained_input = data["bfm_input"].copy()
    chained_input[:, 210:] = z
    chained = run("bfm", chained_input)
    isolated_action, isolated_target = action_targets(isolated, meta)
    chained_action, chained_target = action_targets(chained, meta)
    # Verify the independent deployment conversion against actual simulator conversion.
    _, reconstructed = action_targets(data["bfm_action"], meta)
    checks = {
        "encoder_raw_z": (raw, data["raw_z"]),
        "normalized_z": (z, data["bfm_input"][:, 210:]),
        "bfm_same_input": (isolated, data["bfm_action"]),
        "bfm_chained": (chained, data["bfm_action"]),
        "target_action_same_input": (isolated_action, data["target_action"]),
        "target_action_chained": (chained_action, data["target_action"]),
        "joint_target_same_input_rad": (isolated_target, data["joint_target"]),
        "joint_target_chained_rad": (chained_target, data["joint_target"]),
        "target_conversion_only_rad": (reconstructed, data["joint_target"]),
    }
    report = {
        "trace": str(trace.resolve()), "export_dir": str(export_dir.resolve()),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "onnx_sha256": {name: interface["models"][name]["sha256"] for name in sessions},
        "provider": "CPUExecutionProvider", "onnxruntime_version": ort.__version__, "batch_size": batch_size,
        "samples": samples, "num_envs": meta["num_envs"], "done_count": int(np.count_nonzero(data["done"])),
        "command_min": data["encoder_obs"][:, :3].min(axis=0).tolist(),
        "command_max": data["encoder_obs"][:, :3].max(axis=0).tolist(),
        "target_joint_names": meta["target_joint_names"],
        "joint_target_note": meta["joint_target_note"],
        "errors": {name: error_stats(*pair) for name, pair in checks.items()},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Default: <trace>/onnx_comparison.json")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    report = compare(args.trace, args.export_dir, args.batch_size)
    output = args.output or args.trace / "onnx_comparison.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"samples={report['samples']} done={report['done_count']} provider={report['provider']} batch={args.batch_size}")
    for name, stats in report["errors"].items():
        print(f"{name:32s} max={stats['max_abs']:.8g} mean={stats['mean_abs']:.8g} rmse={stats['rmse']:.8g}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
