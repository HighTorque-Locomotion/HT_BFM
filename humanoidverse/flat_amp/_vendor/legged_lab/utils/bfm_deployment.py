"""Pi12 BFM deployment schema validation, independent of the onboard repository.

Schema 1 matches instinct_onboard's PiBFMPolicy reader. Keep the JSON contract
compatible; the robot does not import HT_BFM or its training dependencies.
"""

import hashlib
import math
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector(value, size, name, positive=False):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(size, array, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise ValueError(f"{name} must contain {size} finite{' positive' if positive else ''} values")
    return array.copy()


def validate_interface(data):
    if data.get("schema") != 1 or data.get("controller") != "pi12_bfm_amp":
        raise ValueError("Expected a Pi12 BFM+AMP deployment interface (schema 1)")
    meta, control = data["bfm_metadata"], data["control"]
    if meta.get("schema") != 3 or meta.get("task") != "walk_pi_waq_12" or meta.get("actuator_profile") != "ht":
        raise ValueError("Unsupported training checkpoint metadata or actuator profile")
    contract = meta["contract"]
    names = meta["target_joint_names"]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(contract["joint_names"]):
        raise ValueError("Target and BFM must contain the same 12 unique joints")
    if len(contract["joint_names"]) != 12:
        raise ValueError("Duplicate BFM joints")
    if contract["history_length"] != 4 or contract["z_dim"] != 256 or contract["norm_z"] is not True:
        raise ValueError("Only the Pi12 four-frame / normalized z256 contract is supported")
    if not isinstance(meta["use_phase"], bool):
        raise ValueError("use_phase must be boolean")
    expected_obs = "command,state,last_action,history_actor"
    if meta["use_phase"]:
        expected_obs += ",phase_sin_cos"
    if meta["actor_observation"] != expected_obs or meta["latent_projection"] != "sqrt(z_dim)*normalize(raw_z)":
        raise ValueError("Unsupported encoder observation or latent projection")
    for key in ("action_multiplier", "action_clip", "angular_velocity_scale", "control_dt"):
        vector(contract[key], 1, key, positive=True)
    vector(contract["default_pos"], 12, "BFM default_pos")
    vector(contract["action_scale"], 12, "BFM action_scale", positive=True)
    vector(meta["target_default_pos"], 12, "target_default_pos")
    vector(meta["target_action_scale"], 12, "target_action_scale", positive=True)
    for key in ("p_gains", "d_gains", "torque_limits"):
        vector(control[key], 12, key, positive=True)
    for key in ("step_dt", "clip_actions", "gait_cycle", "lin_command_threshold", "yaw_command_threshold"):
        vector(control[key], 1, key, positive=True)
    if not math.isclose(control["step_dt"], contract["control_dt"], abs_tol=1e-9):
        raise ValueError("BFM and target control periods differ")
    vector(control["command_scale"], 3, "command_scale", positive=True)
    limits = np.asarray(control["command_ranges"], dtype=np.float32)
    if limits.shape != (3, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Invalid command ranges")
    if np.any(limits[:, 0] > 0) or np.any(limits[:, 1] < 0):
        raise ValueError("Command ranges must include standing (zero)")
    low = vector(control["joint_limits_low"], 12, "joint_limits_low")
    high = vector(control["joint_limits_high"], 12, "joint_limits_high")
    default = vector(meta["target_default_pos"], 12, "target_default_pos")
    if np.any(low >= high) or np.any(default < low) or np.any(default > high):
        raise ValueError("Invalid joint limits or default pose outside limits")
    if meta["use_phase"]:
        phase = meta["phase_observation"]
        for key in ("gait_cycle", "lin_command_threshold", "yaw_command_threshold"):
            if not math.isclose(phase[key], control[key], abs_tol=1e-9):
                raise ValueError(f"Checkpoint phase contract differs: {key}")
        if phase["standing_phase"] != 0 or phase["reset_phase_offsets"] != [0.0, 0.5]:
            raise ValueError("Unsupported phase reset convention")
    return data
