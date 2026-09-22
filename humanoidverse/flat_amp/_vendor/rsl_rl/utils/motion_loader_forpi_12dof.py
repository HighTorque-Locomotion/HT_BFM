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

"""Load named Pi12 leg trajectories from the HT_lab motion-selection YAML.

The source NPZs contain joint_pos/joint_names, base_pos_w, base_quat_w (wxyz)
and framerate. Upper-body columns are ignored by name. Foot positions are
recomputed with the target URDF, in the same base frame as the online AMP
features. MuJoCo is used only for offline forward kinematics, not simulation.
"""

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from humanoidverse.flat_amp.pi12_paths import motion_yaml as default_motion_yaml
from humanoidverse.flat_amp.pi12_paths import robot_urdf

LEG_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint" for side in ("l", "r") for joint in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")
)


def selected_motion_files(selection_path):
    selection_path = Path(selection_path).expanduser().resolve()
    with selection_path.open() as stream:
        selection = yaml.safe_load(stream)
    entries = selection.get("selected_files") if isinstance(selection, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{selection_path}: expected a nonempty selected_files list")
    result = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(f"{selection_path}: motion entries must be paths")
        path = Path(entry).expanduser()
        candidates = [path] if path.is_absolute() else [selection_path.parent / path, selection_path.parent.parent / path]
        resolved = next((p.resolve() for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"{selection_path}: selected motion does not exist: {entry}")
        if resolved.suffix != ".npz":
            raise ValueError(f"Pi12 requires named NPZ motions, got {resolved}")
        result.append(str(resolved))
    return result


class AMPLoaderPi12:
    """Standalone Pi12 expert loading, interpolation and transition sampling."""

    START = 0
    ROOT_VEL_SIZE = 6
    FULL_JOINT_POS_SIZE = 12
    FULL_JOINT_VEL_SIZE = 12
    WAIST_JOINT_NAMES = ()
    END_EFFECTOR_POS_SIZE_WITHOUT_WAIST = 6

    def __init__(
        self,
        device,
        time_between_frames,
        motion_files=None,
        motion_yaml=None,
        urdf_path=None,
        amp_exclude_ankle_roll=False,
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_weights=None,
    ):
        if amp_exclude_ankle_roll:
            raise ValueError("Pi12 AMP includes all 12 joints, including ankle roll")
        if not motion_files:
            motion_files = selected_motion_files(motion_yaml or default_motion_yaml())
        if any(Path(p).suffix != ".npz" for p in motion_files):
            raise ValueError("Pi12 requires named NPZ motions")
        self._fk_model = mujoco.MjModel.from_xml_path(str(urdf_path or robot_urdf()))
        self._fk_data = mujoco.MjData(self._fk_model)
        model_names = [self._fk_model.joint(i).name for i in range(self._fk_model.njnt)]
        if self._fk_model.nq != 12 or set(model_names) != set(LEG_JOINT_NAMES):
            raise ValueError("Pi12 AMP requires the 12 named leg joints in the target URDF")
        self._qpos_ids = [int(self._fk_model.joint(n).qposadr[0]) for n in LEG_JOINT_NAMES]
        self._feet_ids = [self._fk_model.body(f"{side}_ankle_roll_link").id for side in ("l", "r")]
        self._root_id = self._fk_model.body("base_link").id
        self.device = device
        self.time_between_frames = time_between_frames
        self.motion_weights_override = motion_weights
        self.amp_exclude_ankle_roll = False
        self.has_waist_joints = len(self.WAIST_JOINT_NAMES) > 0
        excluded_ankle_roll_joints = 2 if self.amp_exclude_ankle_roll else 0
        self.amp_joint_pos_size = self.FULL_JOINT_POS_SIZE - excluded_ankle_roll_joints
        self.amp_joint_vel_size = self.FULL_JOINT_VEL_SIZE - excluded_ankle_roll_joints
        print(f"AMP joint pos size: {self.amp_joint_pos_size}, AMP joint vel size: {self.amp_joint_vel_size}")
        self.joint_pose_start_idx = AMPLoaderPi12.START + AMPLoaderPi12.ROOT_VEL_SIZE
        self.joint_pose_end_idx = self.joint_pose_start_idx + self.amp_joint_pos_size
        self.joint_vel_start_idx = self.joint_pose_end_idx
        self.joint_vel_end_idx = self.joint_vel_start_idx + self.amp_joint_vel_size
        self.end_pos_start_idx = self.joint_vel_end_idx
        end_effector_pos_size = self.END_EFFECTOR_POS_SIZE_WITHOUT_WAIST
        if self.has_waist_joints:
            end_effector_pos_size += 3
        self.end_pos_end_idx = self.end_pos_start_idx + end_effector_pos_size
        print(f"AMP obs size: {self.end_pos_end_idx - AMPLoaderPi12.START}")

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
            motion_data, frame_duration, motion_weight = self._load_npz_motion(motion_file)

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

    def foot_positions(self, joint_positions):
        """Input is left leg then right leg; output is base-local left/right foot XYZ."""
        result = np.empty((len(joint_positions), 6), dtype=np.float64)
        for index, pose in enumerate(joint_positions):
            self._fk_data.qpos[self._qpos_ids] = pose
            mujoco.mj_kinematics(self._fk_model, self._fk_data)
            root_rotation = self._fk_data.xmat[self._root_id].reshape(3, 3)
            offsets = self._fk_data.xpos[self._feet_ids] - self._fk_data.xpos[self._root_id]
            result[index] = (offsets @ root_rotation).reshape(-1)
        return result

    def _load_npz_motion(self, motion_file):
        # HT_lab's selected local NPZs include object-dtype joint-name arrays.
        with np.load(motion_file, allow_pickle=True) as data:
            required = ("joint_names", "joint_pos", "base_pos_w", "base_quat_w", "framerate")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{motion_file}: missing fields {missing}")
            names = [str(name) for name in data["joint_names"].tolist()]
            if len(names) != len(set(names)):
                raise ValueError(f"{motion_file}: duplicate joint names")
            indices = [self._joint_index(motion_file, names, name) for name in LEG_JOINT_NAMES]
            all_positions = np.asarray(data["joint_pos"], dtype=np.float64)
            if all_positions.ndim != 2 or all_positions.shape[1] != len(names):
                raise ValueError(f"{motion_file}: joint_pos does not match joint_names")
            positions = all_positions[:, indices]
            root_pos = np.asarray(data["base_pos_w"], dtype=np.float64)
            root_quat = np.asarray(data["base_quat_w"], dtype=np.float64)
            fps = float(np.asarray(data["framerate"]).item())
        count = len(positions)
        if count < 2 or root_pos.shape != (count, 3) or root_quat.shape != (count, 4):
            raise ValueError(f"{motion_file}: expected at least 2 aligned frames of joint/root poses")
        if not np.isfinite(fps) or fps <= 0 or not all(np.isfinite(x).all() for x in (positions, root_pos, root_quat)):
            raise ValueError(f"{motion_file}: nonfinite poses or invalid framerate")
        if np.any(np.linalg.norm(root_quat, axis=1) < 1e-6):
            raise ValueError(f"{motion_file}: invalid zero quaternion")
        dt = 1.0 / fps
        rotation = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])
        # Match HT_lab's forward velocity estimation. Rotation differences use
        # the shortest arc, so a q -> -q sign flip cannot create an angular spike.
        joint_vel = np.diff(positions, axis=0) / dt
        joint_vel = np.concatenate((joint_vel, joint_vel[-1:]))
        linear_world = np.diff(root_pos, axis=0) / dt
        linear_world = np.concatenate((linear_world, linear_world[-1:]))
        angular_world = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
        angular_world = np.concatenate((angular_world, angular_world[-1:]))
        features = np.concatenate(
            (rotation.inv().apply(linear_world), rotation.inv().apply(angular_world), positions, joint_vel, self.foot_positions(positions)),
            axis=1,
        ).astype(np.float32)
        if features.shape != (count, 36) or not np.isfinite(features).all():
            raise ValueError(f"{motion_file}: invalid Pi12 AMP features")
        return features, dt, 1.0


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

    @staticmethod
    def _joint_index(motion_file, joint_names, joint_name):
        if joint_name not in joint_names:
            raise ValueError(f"{motion_file} is missing joint '{joint_name}' in joint_names")
        return joint_names.index(joint_name)

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
        all_frame_amp_starts = torch.zeros(len(traj_idxs), self.end_pos_end_idx - AMPLoaderPi12.START, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), self.end_pos_end_idx - AMPLoaderPi12.START, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoaderPi12.START : self.end_pos_end_idx
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoaderPi12.START : self.end_pos_end_idx]
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
                s = self.preloaded_s[idxs, AMPLoaderPi12.START : self.end_pos_end_idx]
                s_next = self.preloaded_s_next[idxs, AMPLoaderPi12.START : self.end_pos_end_idx]
                # print(AMPLoaderPi12.END_POS_END_IDX-AMPLoaderPi12.START)
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
        return pose[AMPLoaderPi12.START : self.joint_pose_end_idx]

    def get_joint_pose_batch(self, poses):
        return poses[:, AMPLoaderPi12.START : self.joint_pose_end_idx]

    def get_joint_vel(self, pose):
        return pose[self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_end_pos(self, pose):
        return pose[self.end_pos_start_idx : self.end_pos_end_idx]

    def get_end_pos_batch(self, poses):
        return poses[:, self.end_pos_start_idx : self.end_pos_end_idx]
