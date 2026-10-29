"""Export a matched HT_BFM Pi12 command encoder + native frozen BFM to ONNX.

Run on the training workstation in the HT_BFM environment. No Isaac Sim, ROS,
TensorRT engine build or robot control is involved. The output must be new.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from legged_lab.utils.bfm_deployment import (
    file_sha256,
    validate_interface,
    vector,
)


class ConfigLoader(yaml.SafeLoader):
    """Accept Isaac config tuples/slice data without importing or executing objects."""


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", lambda loader, node: loader.construct_sequence(node)
)


def resolve_values(value, names, label, default=None):
    if not isinstance(value, dict):
        return vector(value, len(names), label)
    result = []
    for name in names:
        matches = [number for pattern, number in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1 or (not matches and default is None):
            raise ValueError(f"{label}: expected one matching value for {name}, got {matches}")
        result.append(matches[0] if matches else default)
    return vector(result, len(names), label)


def control_config(env, meta, urdf_path):
    names = meta["target_joint_names"]
    robot = env["scene"]["robot"]
    default = resolve_values(robot["init_state"]["joint_pos"], names, "default pose", default=0.0)
    scale = resolve_values(env["robot"]["action_scale"], names, "target action scale")
    np.testing.assert_allclose(default, meta["target_default_pos"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(scale, meta["target_action_scale"], atol=1e-6, rtol=0)
    gains = {"p_gains": [], "d_gains": [], "torque_limits": []}
    for name in names:
        actuators = [
            cfg
            for cfg in robot["actuators"].values()
            if any(re.fullmatch(pattern, name) for pattern in cfg["joint_names_expr"])
        ]
        if len(actuators) != 1:
            raise ValueError(f"Expected one actuator for {name}")
        cfg = actuators[0]
        for field, source in (("p_gains", "stiffness"), ("d_gains", "damping"), ("torque_limits", "max_torque")):
            gains[field].append(float(resolve_values(cfg[source], [name], source)[0]))
    joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
    limits = []
    for name in names:
        joint = joints[name]
        if joint.attrib["type"] != "revolute":
            raise ValueError(f"Expected a bounded revolute joint: {name}")
        limit = joint.find("limit")
        limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    ranges = env["commands"]["ranges"]
    gait = env["gait"]
    return {
        **gains,
        "joint_limits_low": [pair[0] for pair in limits],
        "joint_limits_high": [pair[1] for pair in limits],
        "step_dt": float(env["sim"]["dt"] * env["sim"]["decimation"]),
        "clip_actions": float(env["normalization"]["clip_actions"]),
        "command_scale": vector(env["normalization"]["obs_scales"]["commands"], 3, "command scale").tolist(),
        "command_ranges": [list(ranges[key]) for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
        "gait_cycle": float(gait["gait_cycle"]),
        "lin_command_threshold": float(gait["lin_command_threshold"]),
        "yaw_command_threshold": float(gait["yaw_command_threshold"]),
    }


def build_encoder(checkpoint, frozen):
    from rsl_rl.modules.bfm_command_actor_critic import BFMCommandActorCritic

    meta = checkpoint["infos"]["bfm"]
    state = checkpoint["model_state_dict"]

    def linear_shapes(prefix):
        keys = [key for key, value in state.items() if re.fullmatch(rf"{prefix}\.\d+\.weight", key) and value.ndim == 2]
        return [state[key].shape for key in sorted(keys, key=lambda key: int(key.split(".")[1]))]

    actors, critics = linear_shapes("actor"), linear_shapes("critic")
    obs_dim = 215 if meta["use_phase"] else 213
    if not actors or not critics or actors[0][1] != obs_dim or actors[-1][0] != 256:
        raise ValueError("Checkpoint does not contain the expected Pi12 command-to-z encoder")
    if critics[0][1] != obs_dim + 52 or critics[-1][0] != 1:
        raise ValueError("Unexpected Pi12 critic structure")
    if actors[0][0] != frozen.encoder_hidden_dim or len(actors) - 1 != frozen.encoder_hidden_layers:
        raise ValueError("Encoder architecture differs from its native BFM contract")
    policy = BFMCommandActorCritic(
        obs_dim,
        obs_dim + 52,
        256,
        hidden_dim=frozen.encoder_hidden_dim,
        hidden_layers=frozen.encoder_hidden_layers,
        critic_hidden_dims=[shape[0] for shape in critics[:-1]],
        activation="elu",
    )
    policy.load_state_dict(state, strict=True)
    policy.eval().requires_grad_(False)
    return policy.actor, obs_dim


def export_models(checkpoint, frozen, interface, output):
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn

    encoder, obs_dim = build_encoder(checkpoint, frozen)

    class BFMForExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, actor_obs):
            return self.model(
                {"state": actor_obs[:, :30], "last_action": actor_obs[:, 30:42], "history_actor": actor_obs[:, 42:210]},
                actor_obs[:, 210:],
            )

    bfm = BFMForExport(frozen.model).eval().requires_grad_(False)
    interface["models"] = {}
    rng = np.random.default_rng(42)
    for key, model, input_dim, output_dim, input_name in (
        ("encoder", encoder, obs_dim, 256, "encoder_obs"),
        ("bfm", bfm, 466, 12, "actor_obs"),
    ):
        filename = key + ".onnx"
        path = output / filename
        example = torch.zeros(1, input_dim, dtype=torch.float32)
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=[input_name],
            output_names=["raw_z" if key == "encoder" else "action"],
            dynamic_axes={input_name: {0: "batch"}, ("raw_z" if key == "encoder" else "action"): {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        max_error = 0.0
        for batch in (1, 2, 4):
            sample = rng.normal(0, 0.25, (batch, input_dim)).astype(np.float32)
            if key == "bfm":
                z = sample[:, 210:]
                sample[:, 210:] = z * 16 / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            with torch.no_grad():
                expected = model(torch.from_numpy(sample)).numpy()
            actual = session.run(None, {input_name: sample})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        interface["models"][key] = {
            "file": filename,
            "sha256": file_sha256(path),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "native_onnx_max_abs_error": max_error,
        }
        print(f"{filename}: {input_dim} -> {output_dim}, native/ONNX max error {max_error:.3g}")


def export_checkpoint(args, checkpoint):
    """Export a BFM checkpoint selected by the common flat_amp export entry point."""
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output is not None else None
    import torch

    from legged_lab.utils.bfm_actor import FrozenBFM, resolve_robot_config
    from legged_lab.utils.bfm_bridge import BFMContract

    torch.set_num_threads(2)
    meta = (checkpoint.get("infos") or {}).get("bfm")
    if not meta:
        raise ValueError("Expected a BFM+AMP checkpoint, not a direct-joint AMP checkpoint")
    contract = BFMContract.from_robot_yaml(resolve_robot_config(args.bfm_weights, args.bfm_robot_config))
    frozen = FrozenBFM(args.bfm_weights, contract, "cpu")
    if meta["actor_sha256"] != frozen.digest or meta["contract"] != contract.metadata():
        raise ValueError("Native BFM weights/config do not match the encoder checkpoint")
    env_path = args.env_config or checkpoint_path.parent / "params/env.yaml"
    with env_path.expanduser().open() as stream:
        env = yaml.load(stream, Loader=ConfigLoader)
    urdf_path = (args.urdf or Path(env["scene"]["robot"]["spawn"]["asset_path"])).expanduser().resolve()
    interface = {
        "schema": 1,
        "controller": "pi12_bfm_amp",
        "bfm_metadata": meta,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_env_sha256": file_sha256(env_path.expanduser()),
        "source_urdf_sha256": file_sha256(urdf_path),
        "control": control_config(env, meta, urdf_path),
    }
    validate_interface(interface)
    if not math.isclose(interface["control"]["step_dt"], 0.02, abs_tol=1e-9):
        raise ValueError("This deployment targets the 50 Hz Pi12 policy")
    if args.check_only:
        encoder, obs_dim = build_encoder(checkpoint, frozen)
        with torch.no_grad():
            result = encoder(torch.zeros(1, obs_dim))
        if result.shape != (1, 256) or not torch.isfinite(result).all():
            raise ValueError("Invalid command-to-z inference")
        print(
            f"Validated {checkpoint_path.name}: encoder {obs_dim}->256, BFM 466->12, "
            f"native SHA256={frozen.digest}; no files exported"
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete pair only after both models pass numerical validation.
    with tempfile.TemporaryDirectory(prefix=".pi_bfm_export_", dir=output.parent) as temporary:
        staging = Path(temporary) / "exported"
        staging.mkdir()
        export_models(checkpoint, frozen, interface, staging)
        (staging / "interface.json").write_text(json.dumps(interface, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    print(f"Exported paired Pi12 models to {output}")

