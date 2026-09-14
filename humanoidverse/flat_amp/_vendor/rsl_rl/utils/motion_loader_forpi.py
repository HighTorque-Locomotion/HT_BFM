# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import glob
import json

import numpy as np
import torch


def _quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_apply(q, v):
    q_w = q[..., :1]
    q_vec = q[..., 1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


class AMPLoaderPi:
    FULL_JOINT_POS_SIZE = 23

    FULL_JOINT_VEL_SIZE = 23

    WAIST_JOINT_NAMES = ("waist_yaw_joint",)

    END_EFFECTOR_POS_SIZE_WITHOUT_WAIST = 15

    ROOT_VEL_SIZE = 6

    START = 0

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
        motion_weights=None,
        amp_exclude_ankle_roll=True,
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.motion_weights_override = motion_weights
        self.amp_exclude_ankle_roll = amp_exclude_ankle_roll
        self.has_waist_joints = len(self.WAIST_JOINT_NAMES) > 0
        excluded_ankle_roll_joints = 2 if self.amp_exclude_ankle_roll else 0
        self.amp_joint_pos_size = self.FULL_JOINT_POS_SIZE - excluded_ankle_roll_joints
        self.amp_joint_vel_size = self.FULL_JOINT_VEL_SIZE - excluded_ankle_roll_joints
        print(f"AMP joint pos size: {self.amp_joint_pos_size}, AMP joint vel size: {self.amp_joint_vel_size}")
        self.joint_pose_start_idx = AMPLoaderPi.START + AMPLoaderPi.ROOT_VEL_SIZE
        self.joint_pose_end_idx = self.joint_pose_start_idx + self.amp_joint_pos_size
        self.joint_vel_start_idx = self.joint_pose_end_idx
        self.joint_vel_end_idx = self.joint_vel_start_idx + self.amp_joint_vel_size
        self.end_pos_start_idx = self.joint_vel_end_idx
        end_effector_pos_size = self.END_EFFECTOR_POS_SIZE_WITHOUT_WAIST
        if self.has_waist_joints:
            end_effector_pos_size += 3
        self.end_pos_end_idx = self.end_pos_start_idx + end_effector_pos_size
        print(f"AMP obs size: {self.end_pos_end_idx - AMPLoaderPi.START}")

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            if motion_file.endswith(".npz"):
                motion_data, frame_duration, motion_weight = self._load_npz_motion(motion_file)
            else:
                with open(motion_file) as f:
                    motion_json = json.load(f)
                    motion_data = np.array(motion_json["Frames"], dtype=np.float32)
                    motion_weight = float(motion_json["MotionWeight"])
                    frame_duration = float(motion_json["FrameDuration"])

            motion_tensor = torch.as_tensor(motion_data[:, : self.end_pos_end_idx], dtype=torch.float32, device=device)
            self.trajectories.append(motion_tensor)
            self.trajectories_full.append(motion_tensor)
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(self._resolve_motion_weight(motion_file, i, motion_weight))
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            print(f"traj_len:{traj_len}")
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        print(f"Trajectory weights: {self.trajectory_weights}")
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def _resolve_motion_weight(self, motion_file, motion_idx, default_weight):
        if self.motion_weights_override is None:
            return default_weight
        if isinstance(self.motion_weights_override, dict):
            if motion_file in self.motion_weights_override:
                return float(self.motion_weights_override[motion_file])
            motion_name = motion_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if motion_name in self.motion_weights_override:
                return float(self.motion_weights_override[motion_name])
            return default_weight
        return float(self.motion_weights_override[motion_idx])

    def _load_npz_motion(self, motion_file):
        data = np.load(motion_file, allow_pickle=True)
        required_keys = [
            "fps",
            "root_pos",
            "root_rot",
            "body_names",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"{motion_file} is missing required key '{key}'")

        joint_pos_key = "joint_pos" if "joint_pos" in data else "dof_pos"
        if joint_pos_key not in data:
            raise KeyError(f"{motion_file} is missing required key 'dof_pos' or 'joint_pos'")
        if "joint_pos" in data and "dof_pos" in data:
            pos_diff = float(np.max(np.abs(data["joint_pos"] - data["dof_pos"])))
            if pos_diff > 1.0e-5:
                print(
                    f"[AMPLoaderPi] {motion_file}: joint_pos and dof_pos use different column orders; "
                    "using joint_pos with joint_names."
                )

        if "joint_vel" in data:
            joint_vel_key = "joint_vel"
        elif joint_pos_key == "dof_pos" and "dof_vel" in data:
            joint_vel_key = "dof_vel"
        else:
            joint_vel_key = None

        fps = int(np.asarray(data["fps"]).item())
        frame_duration = 1.0 / fps
        root_pos = torch.tensor(data["root_pos"].astype(np.float32), device=self.device)
        root_quat = torch.tensor(data["root_rot"].astype(np.float32), device=self.device)
        joint_pos = torch.tensor(data[joint_pos_key].astype(np.float32), device=self.device)
        if joint_vel_key is not None:
            joint_vel = torch.tensor(data[joint_vel_key].astype(np.float32), device=self.device)
        else:
            joint_vel = self._finite_difference_joint_vel(joint_pos, frame_duration)
        body_pos_w = torch.tensor(data["body_pos_w"].astype(np.float32), device=self.device)
        body_quat_w = torch.tensor(data["body_quat_w"].astype(np.float32), device=self.device)
        body_lin_vel_w = torch.tensor(data["body_lin_vel_w"].astype(np.float32), device=self.device)
        body_ang_vel_w = torch.tensor(data["body_ang_vel_w"].astype(np.float32), device=self.device)
        body_names = [str(name) for name in data["body_names"].tolist()]

        if joint_pos.shape[0] < 2:
            raise ValueError(f"{motion_file} has {joint_pos.shape[0]} frames, expected at least 2")
        if joint_pos.shape[1] < self.amp_joint_pos_size:
            raise ValueError(
                f"{motion_file} has {joint_pos.shape[1]} joints, expected at least {self.amp_joint_pos_size}"
            )
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"{motion_file} joint_vel shape {joint_vel.shape} does not match joint_pos shape {joint_pos.shape}"
            )

        joint_groups = self._get_joint_groups(motion_file, data, joint_pos.shape[1])

        body_indices = self._get_body_indices(
            motion_file,
            body_names,
            [
                "l_ankle_roll_link",
                "r_ankle_roll_link",
                "torso_link",
                "l_wrist_link",
                "r_wrist_link",
                "camera_link",
            ],
        )
        left_foot_id, right_foot_id, waist_id, left_wrist_id, right_wrist_id, head_end_id = body_indices

        root_lin_vel = _quat_apply(_quat_conjugate(root_quat), body_lin_vel_w[:, 0, :])
        root_ang_vel = _quat_apply(_quat_conjugate(root_quat), body_ang_vel_w[:, 0, :])

        torso_pos = body_pos_w[:, waist_id, :]
        torso_quat = body_quat_w[:, waist_id, :]
        left_foot_pos = _quat_apply(_quat_conjugate(root_quat), body_pos_w[:, left_foot_id, :] - root_pos)
        right_foot_pos = _quat_apply(_quat_conjugate(root_quat), body_pos_w[:, right_foot_id, :] - root_pos)
        left_hand_pos = _quat_apply(_quat_conjugate(torso_quat), body_pos_w[:, left_wrist_id, :] - torso_pos)
        right_hand_pos = _quat_apply(_quat_conjugate(torso_quat), body_pos_w[:, right_wrist_id, :] - torso_pos)
        head_end_pos = _quat_apply(_quat_conjugate(torso_quat), body_pos_w[:, head_end_id, :] - torso_pos)

        joint_pos_parts = [
            joint_pos[:, joint_groups["left_leg"]],
            joint_pos[:, joint_groups["right_leg"]],
        ]
        joint_vel_parts = [
            joint_vel[:, joint_groups["left_leg"]],
            joint_vel[:, joint_groups["right_leg"]],
        ]
        end_pos_parts = [left_foot_pos, right_foot_pos]
        if self.has_waist_joints:
            joint_pos_parts.append(joint_pos[:, joint_groups["waist"]])
            joint_vel_parts.append(joint_vel[:, joint_groups["waist"]])
            waist_pos = _quat_apply(_quat_conjugate(root_quat), torso_pos - root_pos)
            end_pos_parts.append(waist_pos)
        joint_pos_parts.extend(
            [
                joint_pos[:, joint_groups["left_arm"]],
                joint_pos[:, joint_groups["right_arm"]],
                joint_pos[:, joint_groups["head"]],
            ]
        )
        joint_vel_parts.extend(
            [
                joint_vel[:, joint_groups["left_arm"]],
                joint_vel[:, joint_groups["right_arm"]],
                joint_vel[:, joint_groups["head"]],
            ]
        )
        end_pos_parts.extend([left_hand_pos, right_hand_pos, head_end_pos])

        motion_data = torch.cat(
            (
                root_lin_vel,
                root_ang_vel,
                *joint_pos_parts,
                *joint_vel_parts,
                *end_pos_parts,
            ),
            dim=-1,
        )
        if motion_data.shape[1] != self.end_pos_end_idx:
            raise ValueError(
                f"{motion_file} produced AMP obs dim {motion_data.shape[1]}, expected {self.end_pos_end_idx}"
            )
        return motion_data, frame_duration, 1.0

    def _get_joint_groups(self, motion_file, data, num_joints):
        group_names = {
            "left_leg": [
                "l_hip_pitch_joint",
                "l_hip_roll_joint",
                "l_thigh_joint",
                "l_calf_joint",
                "l_ankle_pitch_joint",
            ],
            "right_leg": [
                "r_hip_pitch_joint",
                "r_hip_roll_joint",
                "r_thigh_joint",
                "r_calf_joint",
                "r_ankle_pitch_joint",
            ],
            "waist": list(self.WAIST_JOINT_NAMES),
            "left_arm": [
                "l_shoulder_pitch_joint",
                "l_shoulder_roll_joint",
                "l_upper_arm_joint",
                "l_elbow_joint",
            ],
            "right_arm": [
                "r_shoulder_pitch_joint",
                "r_shoulder_roll_joint",
                "r_upper_arm_joint",
                "r_elbow_joint",
            ],
            "head": ["head_yaw_joint", "head_pitch_joint"],
        }
        if not self.amp_exclude_ankle_roll:
            group_names["left_leg"].append("l_ankle_roll_joint")
            group_names["right_leg"].append("r_ankle_roll_joint")
        print(f"Expected joint groups: {group_names}")
        if "joint_names" in data:
            joint_names = [str(name) for name in data["joint_names"].tolist()]
            return {
                group: [
                    AMPLoaderPi._joint_index(motion_file, joint_names, joint_name) for joint_name in joint_names_group
                ]
                for group, joint_names_group in group_names.items()
            }
        raise ValueError(f"{motion_file} is missing joint_names; pi AMP npz requires named joints")

    @staticmethod
    def _joint_index(motion_file, joint_names, joint_name):
        if joint_name not in joint_names:
            raise ValueError(f"{motion_file} is missing joint '{joint_name}' in joint_names")
        return joint_names.index(joint_name)

    @staticmethod
    def _get_body_indices(motion_file, body_names, required_body_names):
        body_indices = []
        for body_name in required_body_names:
            if body_name not in body_names:
                raise ValueError(f"{motion_file} is missing body '{body_name}' in body_names")
            body_indices.append(body_names.index(body_name))
        return body_indices

    @staticmethod
    def _finite_difference_joint_vel(joint_pos, dt):
        joint_vel = torch.zeros_like(joint_pos)
        if joint_pos.shape[0] >= 3:
            joint_vel[1:-1] = (joint_pos[2:] - joint_pos[:-2]) / (2.0 * dt)
            joint_vel[0] = joint_vel[1]
            joint_vel[-1] = joint_vel[-2]
        else:
            joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / dt
            joint_vel[-1] = joint_vel[-2]
        return joint_vel

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        frame = min(max(p * (n - 1), 0.0), n - 1)
        idx_low, idx_high = int(np.floor(frame)), int(np.ceil(frame))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = frame - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        frame = np.clip(p * (n - 1), 0.0, n - 1)
        idx_low, idx_high = np.floor(frame).astype(np.int64), np.ceil(frame).astype(np.int64)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(frame - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        frame = min(max(p * (n - 1), 0.0), n - 1)
        idx_low, idx_high = int(np.floor(frame)), int(np.ceil(frame))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = frame - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        frame = np.clip(p * (n - 1), 0.0, n - 1)
        idx_low, idx_high = np.floor(frame).astype(np.int64), np.ceil(frame).astype(np.int64)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), self.end_pos_end_idx - AMPLoaderPi.START, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), self.end_pos_end_idx - AMPLoaderPi.START, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoaderPi.START : self.end_pos_end_idx
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoaderPi.START : self.end_pos_end_idx]
        blend = torch.tensor(frame - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoaderPi.START : self.end_pos_end_idx]
                s_next = self.preloaded_s_next[idxs, AMPLoaderPi.START : self.end_pos_end_idx]
                # print(AMPLoaderPi.END_POS_END_IDX-AMPLoaderPi.START)
                # print(s.shape, s_next.shape)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(self, pose):
        return pose[AMPLoaderPi.START : self.joint_pose_end_idx]

    def get_joint_pose_batch(self, poses):
        return poses[:, AMPLoaderPi.START : self.joint_pose_end_idx]

    def get_joint_vel(self, pose):
        return pose[self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_end_pos(self, pose):
        return pose[self.end_pos_start_idx : self.end_pos_end_idx]

    def get_end_pos_batch(self, poses):
        return poses[:, self.end_pos_start_idx : self.end_pos_end_idx]
