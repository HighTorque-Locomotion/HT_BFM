"""Load named Pi12 leg trajectories from the HT_lab motion-selection YAML.

The source NPZs contain joint_pos/joint_names, base_pos_w, base_quat_w (wxyz)
and framerate. Upper-body columns are ignored by name. Foot positions are
recomputed with the target URDF, in the same base frame as the online AMP
features. MuJoCo is used only for offline forward kinematics, not simulation.
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from humanoidverse.flat_amp.pi12_paths import motion_yaml as default_motion_yaml
from humanoidverse.flat_amp.pi12_paths import robot_urdf

from .motion_loader_forpi import AMPLoaderPi

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


class AMPLoaderPi12(AMPLoaderPi):
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
        **kwargs,
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
        super().__init__(
            device,
            time_between_frames,
            motion_files=motion_files,
            amp_exclude_ankle_roll=False,
            **kwargs,
        )

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
