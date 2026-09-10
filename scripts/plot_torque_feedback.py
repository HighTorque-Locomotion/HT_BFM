#!/usr/bin/env python3
"""Plot commanded and measured motor torques from a ROS 2 MCAP bag zip."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

mpl.rcParams["font.family"] = "Noto Sans CJK JP"
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
mpl.rcParams["axes.unicode_minus"] = False


MOTOR_COMMAND_MSG = """string uuid
int32[] motor_ids
float32[] positions
float32[] velocities
float32[] kp
float32[] kd
float32[] torques
"""


MOTOR_ID_TO_JOINT = {
    11: "l_ankle_roll_joint", 10: "l_ankle_pitch_joint", 9: "l_calf_joint", 8: "l_thigh_joint",
    7: "l_hip_roll_joint", 6: "l_hip_pitch_joint", 5: "r_ankle_roll_joint", 4: "r_ankle_pitch_joint",
    3: "r_calf_joint", 2: "r_thigh_joint", 1: "r_hip_roll_joint", 0: "r_hip_pitch_joint",
    22: "waist_yaw_joint", 16: "l_shoulder_pitch_joint", 17: "l_shoulder_roll_joint",
    18: "l_upper_arm_joint", 19: "l_elbow_joint", 12: "r_shoulder_pitch_joint",
    13: "r_shoulder_roll_joint", 14: "r_upper_arm_joint", 15: "r_elbow_joint",
    20: "head_yaw_joint", 21: "head_pitch_joint",
}


def load_data(bag_dir: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(MOTOR_COMMAND_MSG, "hightorque_msgs/msg/MotorControlCommand"))

    command_t: list[float] = []
    command_pos: list[np.ndarray] = []
    command_vel: list[np.ndarray] = []
    command_kp: list[np.ndarray] = []
    command_kd: list[np.ndarray] = []
    joint_t: list[float] = []
    joint_pos: list[np.ndarray] = []
    joint_vel: list[np.ndarray] = []
    joint_tau: list[np.ndarray] = []
    joint_names: list[str] | None = None

    with Reader(bag_dir) as reader:
        for connection in reader.connections:
            if connection.topic not in {"/control_command", "/joint_states"}:
                continue
            for _, timestamp_ns, raw in reader.messages(connections=[connection]):
                msg = typestore.deserialize_cdr(raw, connection.msgtype)
                timestamp = timestamp_ns * 1e-9
                if connection.topic == "/control_command":
                    ids = np.asarray(msg.motor_ids, dtype=np.int64)
                    positions = np.asarray(msg.positions, dtype=np.float64)
                    velocities = np.asarray(msg.velocities, dtype=np.float64)
                    kp = np.asarray(msg.kp, dtype=np.float64)
                    kd = np.asarray(msg.kd, dtype=np.float64)
                    if ids.size != 23 or positions.size != 23 or velocities.size != 23 or kp.size != 23 or kd.size != 23:
                        continue
                    values = [0.0] * 23
                    for i, motor_id in enumerate(ids):
                        values[list(MOTOR_ID_TO_JOINT).index(int(motor_id))] = positions[i]
                    def by_id(values_in):
                        return np.asarray([values_in[list(ids).index(motor_id)] for motor_id in MOTOR_ID_TO_JOINT], dtype=np.float64)
                    command_t.append(timestamp)
                    command_pos.append(by_id(positions))
                    command_vel.append(by_id(velocities))
                    command_kp.append(by_id(kp))
                    command_kd.append(by_id(kd))
                else:
                    names = list(msg.name)
                    effort = np.asarray(msg.effort, dtype=np.float64)
                    if len(names) != 23 or effort.size != 23:
                        continue
                    if joint_names is None:
                        joint_names = names
                    elif names != joint_names:
                        raise RuntimeError("/joint_states joint order changes during the bag")
                    joint_t.append(timestamp)
                    joint_pos.append(np.asarray(msg.position, dtype=np.float64))
                    joint_vel.append(np.asarray(msg.velocity, dtype=np.float64))
                    joint_tau.append(effort)

    if not command_t or not joint_t or joint_names is None:
        raise RuntimeError("The bag does not contain usable /control_command and /joint_states torque data")

    return (
        np.asarray(command_t),
        np.asarray(command_pos),
        np.asarray(command_vel),
        np.asarray(command_kp),
        np.asarray(command_kd),
        np.asarray(joint_t),
        np.asarray(joint_pos),
        np.asarray(joint_vel),
        np.asarray(joint_tau),
        joint_names,
    )


def make_plot(command_t, command_pos, command_vel, command_kp, command_kd, joint_t, joint_pos, joint_vel, measured_tau, names, output: Path) -> None:
    order = np.argsort(joint_t)
    joint_t = joint_t[order]
    measured_tau = measured_tau[order]
    command_order = np.argsort(command_t)
    command_t = command_t[command_order]
    command_pos = command_pos[command_order]
    command_vel = command_vel[command_order]
    command_kp = command_kp[command_order]
    command_kd = command_kd[command_order]

    # Remove duplicate timestamps before interpolation and hold the last command.
    unique_t, unique_idx = np.unique(command_t, return_index=True)
    command_pos = command_pos[unique_idx]
    command_vel = command_vel[unique_idx]
    command_kp = command_kp[unique_idx]
    command_kd = command_kd[unique_idx]
    command_on_joint = np.empty_like(measured_tau)
    command_velocity_on_joint = np.empty_like(measured_tau)
    kp_on_joint = np.empty_like(measured_tau)
    kd_on_joint = np.empty_like(measured_tau)
    for motor in range(23):
        command_on_joint[:, motor] = np.interp(
            joint_t,
            unique_t,
            command_pos[:, motor],
            left=command_pos[0, motor],
            right=command_pos[-1, motor],
        )
        command_velocity_on_joint[:, motor] = np.interp(joint_t, unique_t, command_vel[:, motor], left=command_vel[0, motor], right=command_vel[-1, motor])
        kp_on_joint[:, motor] = np.interp(joint_t, unique_t, command_kp[:, motor], left=command_kp[0, motor], right=command_kp[-1, motor])
        kd_on_joint[:, motor] = np.interp(joint_t, unique_t, command_kd[:, motor], left=command_kd[0, motor], right=command_kd[-1, motor])
    target_tau = kp_on_joint * (command_on_joint - joint_pos) + kd_on_joint * (command_velocity_on_joint - joint_vel)

    relative_t = joint_t - joint_t[0]
    fig, axes = plt.subplots(8, 3, figsize=(18, 24), sharex=True)
    axes = axes.ravel()
    for motor, (axis, name) in enumerate(zip(axes, names)):
        axis.plot(relative_t, target_tau[:, motor], color="#d62728", linewidth=0.9, label="目标力矩（PD计算）")
        axis.plot(relative_t, measured_tau[:, motor], color="#1f77b4", linewidth=0.8, alpha=0.9, label="实际力矩反馈")
        axis.set_title(f"电机 {motor}: {name}", fontsize=10)
        axis.set_ylabel("力矩 (N·m)")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
    for axis in axes[23:]:
        axis.axis("off")
    axes[21].set_xlabel("时间 (s)")
    axes[22].set_xlabel("时间 (s)")
    fig.suptitle("目标电机力矩 vs 实际电机力矩反馈值", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="torque_bag_") as temp_dir:
        with zipfile.ZipFile(args.zip_path) as archive:
            archive.extractall(temp_dir)
        bag_dirs = list(Path(temp_dir).glob("**/*.mcap"))
        if len(bag_dirs) != 1:
            raise RuntimeError(f"Expected one MCAP file, found {len(bag_dirs)}")
        data = load_data(bag_dirs[0].parent)
    make_plot(*data, args.output)
    print(f"saved {args.output}")
    print(f"joint samples: {data[5].shape[0]}, command samples: {data[0].shape[0]}, duration: {data[5][-1] - data[5][0]:.3f}s")


if __name__ == "__main__":
    main()
