import os
import json
import subprocess
import threading
import time

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
import torch
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation
import joblib
import numpy as np
from torch.utils._pytree import tree_map

try:
    import mediapy as media
except ImportError:
    # Video export is optional; inference without --save-mp4 should not need
    # the mediapy package.
    media = None

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


ROBOT_CONFIG_OVERRIDES = {
    "g1": "robot=g1/g1_29dof_hard_waist",
    "PiPlus_S_12L8A0G2H1W_LSE": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "piplus_lse": "robot=piplus/PiPlus_S_12L8A0G2H1W_LSE",
    "PiPlus_S_12L8A0G2H0W": "robot=piplus/PiPlus_S_12L8A0G2H0W",
}


def _find_motion_msgs_python_path() -> str | None:
    """Locate the Python bindings for the custom motion_target_msgs package.

    The ROS2 system Python (3.10) only finds them when the instinct_onboard
    colcon workspace is sourced; probe the known workspace layouts instead so
    the bridge can import ``motion_target_msgs`` without manual sourcing.
    """
    import glob

    base_dirs = [
        "~/project/test/ws_instinct_onboard",
        "~/project/test/instinct_onboard",
        "~/WorkSpace/instinct_onboard",
        "~/桌面/instinct_onboard-b",
        "~/WorkSpace/instinct_onboard-ht_jump1",
    ]
    patterns = []
    for base in base_dirs:
        expanded = os.path.expanduser(base)
        patterns.append(os.path.join(expanded, "install", "*", "local", "lib", "python3*", "dist-packages"))
        patterns.append(os.path.join(expanded, "install", "*", "lib", "python3*", "site-packages"))
    for pattern in patterns:
        for candidate in glob.glob(pattern):
            if os.path.isdir(os.path.join(candidate, "motion_target_msgs")):
                return candidate
    return None


def _build_bridge_env() -> dict:
    """Environment for the stdout ROS bridge: source the ROS Humble setup so
    the system Python can load rclpy (LD_LIBRARY_PATH etc.) and add the
    motion_target_msgs bindings to PYTHONPATH."""
    env = os.environ.copy()
    setup_script = "/opt/ros/humble/setup.bash"
    if os.path.exists(setup_script):
        try:
            proc = subprocess.run(
                ["bash", "-c", f"source {setup_script} >/dev/null 2>&1 && env"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                ros_env = {}
                for line in proc.stdout.splitlines():
                    if "=" in line:
                        key, _, value = line.partition("=")
                        ros_env[key] = value
                for key in (
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "PYTHONPATH",
                    "AMENT_PREFIX_PATH",
                    "CMAKE_PREFIX_PATH",
                    "PKG_CONFIG_PATH",
                    "RMW_IMPLEMENTATION",
                    "COLCON_PREFIX_PATH",
                    "ROS_DISTRO",
                    "ROS_VERSION",
                    "CYCLONEDDS_URI",
                ):
                    if key in ros_env:
                        env[key] = ros_env[key]
        except Exception:
            pass  # fall through to the manual PYTHONPATH handling below
    else:
        # Manual fallback without setup.bash: the two known ROS Python dirs.
        for ros_site in (
            "/opt/ros/humble/lib/python3.10/site-packages",
            "/opt/ros/humble/local/lib/python3.10/dist-packages",
        ):
            if os.path.isdir(ros_site) and ros_site not in env.get("PYTHONPATH", ""):
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = ros_site + (os.pathsep + existing if existing else "")
    motion_msgs_path = _find_motion_msgs_python_path()
    if motion_msgs_path is not None:
        if motion_msgs_path not in env.get("PYTHONPATH", ""):
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = motion_msgs_path + (os.pathsep + existing if existing else "")
        # The rosidl type-support shared libraries live in .../motion_target_msgs/lib.
        lib_dir = os.path.abspath(
            os.path.join(motion_msgs_path, "..", "..", "..", "..", "lib")
        )
        if os.path.isdir(lib_dir):
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            if lib_dir not in existing_ld:
                env["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + existing_ld if existing_ld else "")
    return env


class OnnxTrackingPolicy:
    """Run the deployed actor and realtime z encoder with ONNX Runtime."""

    def __init__(
        self,
        model_folder: Path,
        policy_path: Path | None = None,
        z_encoder_path: Path | None = None,
        smoothing_horizon: int = 8,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX mode requires onnxruntime. Install onnxruntime or onnxruntime-gpu."
            ) from exc

        exported_dir = model_folder / "exported"
        self.policy_path = policy_path or exported_dir / "FBcprAuxModel.onnx"
        self.z_encoder_path = z_encoder_path or exported_dir / "FBcprAuxModel_z_encoder.onnx"
        for path in (self.policy_path, self.z_encoder_path):
            if not path.exists():
                raise FileNotFoundError(f"ONNX model not found: {path}")

        available_providers = ort.get_available_providers()
        # Prefer CUDA whenever the GPU build of ONNX Runtime is installed;
        # CPU remains a fallback for environments without a CUDA provider.
        # TensorRT is not required for this ONNX path and may be installed
        # without its shared libraries.  Avoid probing it so a broken TRT
        # plugin cannot obscure CUDA provider initialization.
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [name for name in preferred if name in available_providers]
        if not providers:
            raise RuntimeError("ONNX Runtime has no available execution provider")
        self.policy_session = ort.InferenceSession(str(self.policy_path), providers=providers)
        self.z_session = ort.InferenceSession(str(self.z_encoder_path), providers=providers)
        self.policy_input = self.policy_session.get_inputs()[0]
        self.policy_output = self.policy_session.get_outputs()[0]
        self.z_input = self.z_session.get_inputs()[0]
        self.z_output = self.z_session.get_outputs()[0]
        self.actor_obs_keys = self._read_actor_obs_keys(model_folder)
        self.z_dim = self._static_last_dim(self.z_output.shape)
        if self.z_dim is None:
            raise ValueError(f"ONNX z encoder output shape is dynamic: {self.z_output.shape}")
        self.smoothing_horizon = max(1, int(smoothing_horizon))
        self.z_history: list[np.ndarray] = []

    @staticmethod
    def _static_last_dim(shape) -> int | None:
        if not shape:
            return None
        value = shape[-1]
        return int(value) if isinstance(value, (int, np.integer)) else None

    @staticmethod
    def _read_actor_obs_keys(model_folder: Path) -> list[str]:
        config_path = model_folder / "config.json"
        with config_path.open("r") as f:
            config = json.load(f)
        model_cfg = config.get("agent", {}).get("model", config.get("model", {}))
        keys = model_cfg.get("archi", {}).get("actor", {}).get("input_filter", {}).get("key")
        if isinstance(keys, str):
            return [keys]
        if not keys:
            raise KeyError(f"Actor observation keys are missing from {config_path}")
        return list(keys)

    def reset(self) -> None:
        self.z_history.clear()

    def encode_z(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            encoder_obs = torch.cat([obs["state"], obs["privileged_state"]], dim=-1)
        except KeyError as exc:
            raise KeyError("ONNX z encoder requires state and privileged_state observations") from exc
        encoder_np = np.ascontiguousarray(encoder_obs.detach().cpu().numpy(), dtype=np.float32)
        expected_dim = self._static_last_dim(self.z_input.shape)
        if expected_dim is not None and encoder_np.shape[-1] != expected_dim:
            raise ValueError(
                f"ONNX z encoder expects {expected_dim} features, got {encoder_np.shape[-1]}"
            )
        # The deployed z encoder is exported with a fixed batch dimension of
        # one, matching the realtime Orin invocation.  Encode an offline
        # reference trajectory frame by frame rather than passing its full
        # time axis as an invalid ONNX batch.
        z_batches = []
        total_frames = encoder_np.shape[0]
        for frame_index, frame in enumerate(encoder_np):
            z_batches.append(
                self.z_session.run(
                    [self.z_output.name], {self.z_input.name: frame[None]}
                )[0]
            )
            if (frame_index + 1) % 500 == 0 or frame_index + 1 == total_frames:
                print(f"Encoding BFM z: {frame_index + 1}/{total_frames}")
        z = np.concatenate(z_batches, axis=0).astype(np.float32, copy=False)
        encoded = []
        for z_t in z:
            self.z_history.append(z_t.copy())
            del self.z_history[:-self.smoothing_horizon]
            smoothed = np.mean(np.stack(self.z_history, axis=0), axis=0)
            norm = np.linalg.norm(smoothed)
            if norm <= 1e-8:
                raise ValueError("ONNX z encoder produced a zero latent vector")
            encoded.append(smoothed * (np.sqrt(self.z_dim) / norm))
        return torch.from_numpy(np.asarray(encoded, dtype=np.float32)).to(encoder_obs.device)

    def _flatten_actor_obs(self, obs: dict[str, torch.Tensor]) -> np.ndarray:
        missing = [key for key in self.actor_obs_keys if key not in obs]
        if missing:
            raise KeyError(f"Actor observation is missing ONNX keys: {missing}; available={list(obs)}")
        return np.ascontiguousarray(
            torch.cat([obs[key] for key in self.actor_obs_keys], dim=-1)
            .detach()
            .cpu()
            .numpy(),
            dtype=np.float32,
        )

    def act(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        actor_obs = self._flatten_actor_obs(obs)
        z_np = np.ascontiguousarray(z.detach().cpu().numpy(), dtype=np.float32)
        model_input = np.concatenate([actor_obs, z_np], axis=-1)
        expected_dim = self._static_last_dim(self.policy_input.shape)
        if expected_dim is not None and model_input.shape[-1] != expected_dim:
            raise ValueError(
                f"ONNX actor expects {expected_dim} features, got {model_input.shape[-1]}"
            )
        action = self.policy_session.run(
            [self.policy_output.name], {self.policy_input.name: model_input}
        )[0]
        return torch.from_numpy(np.asarray(action, dtype=np.float32)).to(z.device)


class RosZSource:
    """Keep the newest 256-D z received from a ROS2 Float32MultiArray topic."""

    def __init__(self, topic: str, expected_dim: int, timeout: float = 0.5, msg_type: str = "bfm_z"):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from std_msgs.msg import Float32MultiArray
        except ImportError:
            # ROS Humble ships rclpy for system Python 3.10 while this project
            # commonly runs in Conda Python 3.11. Use the system interpreter
            # through a line-oriented stdout bridge in that case.
            bridge = Path(__file__).with_name("ros_z_stdout_bridge.py")
            system_python = "/usr/bin/python3"
            if not Path(system_python).exists():
                raise ImportError("ROS2 Python is unavailable in this environment") from None
            self._process = subprocess.Popen(
                [system_python, str(bridge), topic, msg_type],
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                # Inherit the sourced ROS setup environment plus the located
                # motion_target_msgs bindings.
                env=_build_bridge_env(),
            )
            self._bridge_mode = True
            self._start_bridge_reader()
            self.topic = topic
            self.expected_dim = int(expected_dim)
            self.timeout = float(timeout)
            self._lock = threading.Lock()
            self._latest = None
            self._received_at = 0.0
            self._received_count = 0
            self._last_warning_at = 0.0
            return

        self.topic = topic
        self.expected_dim = int(expected_dim)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._received_at = 0.0
        self._received_count = 0
        self._last_warning_at = 0.0
        self._rclpy = rclpy

        if not rclpy.ok():
            rclpy.init(args=None)
        source = self

        class _Node(Node):
            def __init__(self):
                super().__init__("humanoidverse_realtime_z_source")
                qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                )
                self.subscription = self.create_subscription(
                    Float32MultiArray, topic, self._callback, qos
                )

            def _callback(self, msg):
                value = np.asarray(msg.data, dtype=np.float32).reshape(-1)
                if value.shape != (source.expected_dim,) or not np.isfinite(value).all():
                    self.get_logger().error(
                        f"Ignoring invalid z on {source.topic}: shape={value.shape}",
                        throttle_duration_sec=1.0,
                    )
                    return
                with source._lock:
                    source._latest = value.copy()
                    source._received_at = time.monotonic()
                    source._received_count += 1

        self.node = _Node()
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True,
        )
        self._spin_thread.start()

    def _start_bridge_reader(self) -> None:
        def reader() -> None:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                try:
                    value = np.asarray(json.loads(line), dtype=np.float32).reshape(-1)
                    if value.shape != (self.expected_dim,) or not np.isfinite(value).all():
                        continue
                    with self._lock:
                        self._latest = value.copy()
                        self._received_at = time.monotonic()
                        self._received_count += 1
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

    def wait_for_first(self) -> torch.Tensor:
        started_at = time.monotonic()
        while True:
            with self._lock:
                if self._latest is not None:
                    return torch.from_numpy(self._latest.copy())
            if getattr(self, "_bridge_mode", False) and self._process.poll() is not None:
                raise RuntimeError(
                    f"ROS z bridge exited with code {self._process.returncode}; "
                    "check that /opt/ros/humble/setup.bash was sourced before starting HT_BFM"
                )
            if time.monotonic() - started_at > max(5.0, self.timeout * 4.0):
                raise TimeoutError(
                    f"No z received on ROS topic {self.topic!r} within 5 seconds. "
                    "Verify `ros2 topic echo /bfm_z_realtime --once` in this shell."
                )
            time.sleep(0.005)

    def latest(self) -> tuple[torch.Tensor, int]:
        with self._lock:
            if self._latest is None:
                raise RuntimeError(f"No z received yet on ROS topic {self.topic!r}")
            age = time.monotonic() - self._received_at
            max_stale = max(5.0, self.timeout * 4.0)
            if age > max_stale:
                raise RuntimeError(
                    f"ROS z topic {self.topic!r} has been silent for {age:.2f}s "
                    f"(>{max_stale:.2f}s); giving up"
                )
            if age > self.timeout:
                # Tolerate short gaps by reusing the latest z; only give up if
                # the topic has been silent for max_stale seconds.
                now = time.monotonic()
                if now - self._last_warning_at >= 1.0:
                    self._last_warning_at = now
                    print(
                        f"WARNING: ROS z topic {self.topic!r} stale for {age:.2f}s "
                        f"(timeout={self.timeout:.2f}s); reusing last z"
                    )
            return torch.from_numpy(self._latest.copy()), self._received_count

    def close(self) -> None:
        if getattr(self, "_bridge_mode", False):
            self._process.terminate()
        else:
            self.node.destroy_node()


class RosJointTargetSource:
    """Latest 23-D target joint angles from a ROS String-JSON topic.

    The publisher (pub_bfm_joint_realtime_*.py) emits one JSON frame per line
    on a String topic.  The frame carries the target ``dof`` (absolute joint
    angles, rad, in the same order as ``robot.dof_names``).
    """

    def __init__(self, topic: str, expected_dim: int, timeout: float = 0.5):
        bridge = Path(__file__).with_name("ros_z_stdout_bridge.py")
        system_python = "/usr/bin/python3"
        if not Path(system_python).exists():
            raise ImportError("ROS2 Python is unavailable in this environment")
        self._process = subprocess.Popen(
            [system_python, str(bridge), topic, "string"],
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            # Inherit the sourced ROS setup environment plus the located
            # motion_target_msgs bindings.
            env=_build_bridge_env(),
        )
        self.topic = topic
        self.expected_dim = int(expected_dim)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._received_at = 0.0
        self._received_count = 0
        self._last_warning_at = 0.0
        self._start_reader()

    def _start_reader(self) -> None:
        def reader() -> None:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                try:
                    frame = json.loads(line)
                    dof = np.asarray(frame["dof"], dtype=np.float32).reshape(-1)
                    if dof.shape != (self.expected_dim,) or not np.isfinite(dof).all():
                        continue
                    with self._lock:
                        self._latest = dof.copy()
                        self._received_at = time.monotonic()
                        self._received_count += 1
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

    def latest(self) -> tuple[torch.Tensor, int]:
        with self._lock:
            if self._latest is None:
                raise RuntimeError(f"No joint target received yet on {self.topic!r}")
            age = time.monotonic() - self._received_at
            max_stale = max(5.0, self.timeout * 4.0)
            if age > max_stale:
                raise RuntimeError(
                    f"Joint target topic {self.topic!r} has been silent for {age:.2f}s "
                    f"(>{max_stale:.2f}s); giving up"
                )
            if age > self.timeout:
                # Tolerate short gaps by reusing the latest target.
                now = time.monotonic()
                if now - self._last_warning_at >= 1.0:
                    self._last_warning_at = now
                    print(
                        f"WARNING: joint target topic {self.topic!r} stale for {age:.2f}s "
                        f"(timeout={self.timeout:.2f}s); reusing last target"
                    )
            return torch.from_numpy(self._latest.copy()), self._received_count

    def close(self) -> None:
        self._process.terminate()


def _plot_joint_tracking(
    target: np.ndarray, sim: np.ndarray, dof_names: list[str], out_path: Path
) -> None:
    """Plot per-joint target (red dashed) vs sim (blue solid) angle curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(target.shape[0])
    n = target.shape[1]
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), squeeze=False)
    for j in range(n):
        ax = axes[j // cols][j % cols]
        ax.plot(steps, target[:, j], color="red", linestyle="--", linewidth=1.0, label="target")
        ax.plot(steps, sim[:, j], color="blue", linewidth=1.2, label="sim")
        ax.set_title(dof_names[j], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Joint angle (rad): target (red dashed) vs sim (blue)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_joint_error(error: np.ndarray, dof_names: list[str], out_path: Path) -> None:
    """Plot mean abs error / RMSE curves over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(error.shape[0])
    mean_err = error.mean(axis=1)
    rms = np.sqrt((error**2).mean(axis=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, mean_err, label="mean abs error", linewidth=1.2)
    ax.plot(steps, rms, label="RMSE", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("rad")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _flush_joint_comparison(
    joint_errs: list[np.ndarray],
    joint_target_log: list[np.ndarray],
    joint_sim_log: list[np.ndarray],
    dof_names: list[str],
    output_dir: Path,
    motion_id: int,
) -> None:
    """Print accuracy summary and save data/plots for the joint comparison.

    Called from a finally block so that a Ctrl+C interruption still flushes
    the samples accumulated so far.
    """
    if not joint_errs:
        return
    arr = np.stack(joint_errs)
    rms = np.sqrt((arr**2).mean())
    print("\n" + "=" * 70)
    print(f"=== Joint tracking accuracy (motion {motion_id}) ===")
    print(f"samples       : {arr.shape[0]}")
    print(f"mean abs error: {arr.mean():.4f} rad ({np.degrees(arr.mean()):.2f} deg)")
    print(f"RMSE          : {rms:.4f} rad ({np.degrees(rms):.2f} deg)")
    print(f"max abs error : {arr.max():.4f} rad ({np.degrees(arr.max()):.2f} deg)")
    print("per-joint mean abs error (rad, deg):")
    for name, mae in zip(dof_names, arr.mean(axis=0)):
        print(f"  {name:24s} {mae:8.4f}  {np.degrees(mae):7.2f}")
    print("=" * 70)

    np.savez(
        output_dir / f"joint_tracking_{motion_id}.npz",
        target_dof=np.stack(joint_target_log),
        sim_dof=np.stack(joint_sim_log),
        error=arr,
        dof_names=np.array(dof_names),
    )
    print(f"Saved joint tracking data: {output_dir / f'joint_tracking_{motion_id}.npz'}")
    _plot_joint_tracking(
        np.stack(joint_target_log), np.stack(joint_sim_log), dof_names,
        output_dir / f"joint_tracking_{motion_id}.png",
    )
    _plot_joint_error(
        arr, dof_names, output_dir / f"joint_tracking_error_{motion_id}.png",
    )
    print(
        f"Saved joint tracking plots: "
        f"{output_dir / f'joint_tracking_{motion_id}.png'}, "
        f"{output_dir / f'joint_tracking_error_{motion_id}.png'}"
    )


def _resolve_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.exists():
        return expanded.resolve()

    path_text = str(expanded)
    marker = "humanoidverse/"
    if marker in path_text:
        repo_relative = HUMANOIDVERSE_DIR / path_text.split(marker, 1)[1]
        if repo_relative.exists():
            return repo_relative.resolve()

    return expanded


def _append_or_replace_hydra_override(overrides: list[str], override: str) -> None:
    key = override.split("=", 1)[0]
    for index, existing in enumerate(overrides):
        if existing.split("=", 1)[0] == key:
            overrides[index] = override
            return
    overrides.append(override)


def _is_hydra_group_override(override: str) -> bool:
    key = override.split("=", 1)[0].lstrip("+~").split("@", 1)[0]
    return "." not in key


def main(
    model_folder: Path,
    data_path: Path | None = None,
    headless: bool = True,
    device="cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    video_name: str | None = None,
    video_fps: int = 50,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    motion_list: list[int] = [25],
    robot: str | None = None,
    episode_len: int | None = None,
    no_training_config: bool = False,
    onnx: bool = False,
    onnx_model_folder: Path | None = None,
    onnx_policy_path: Path | None = None,
    onnx_z_encoder_path: Path | None = None,
    precomputed_z_path: Path | None = None,
    z_smoothing_horizon: int = 8,
    mujoco_xml_path: Path | None = None,
    ros_z_topic: str | None = None,
    ros_z_timeout: float = 0.5,
    ros_z_msg_type: str = "bfm_z",
    joint_topic: str | None = None,
    joint_timeout: float = 0.5,
):
    # motion_list: motion ids to evaluate (default [25])
    if save_mp4 and media is None:
        raise ImportError(
            "--save-mp4 requires mediapy. Install it with `pip install mediapy`."
        )
    
    model_folder = _resolve_path(model_folder)
    env_device = "cuda:0" if device == "cuda" else device
    simulator = simulator.lower()

    onnx_mode = onnx or onnx_model_folder is not None or onnx_policy_path is not None or onnx_z_encoder_path is not None
    if ros_z_topic is not None and not onnx_mode:
        raise ValueError("--ros-z-topic requires --onnx because the realtime z is consumed by the ONNX actor")
    if onnx_model_folder is not None:
        onnx_model_folder = _resolve_path(onnx_model_folder)
    inference_model_folder = onnx_model_folder or model_folder
    if onnx_mode:
        model = None
        onnx_policy = OnnxTrackingPolicy(
            inference_model_folder,
            policy_path=_resolve_path(onnx_policy_path) if onnx_policy_path is not None else None,
            z_encoder_path=_resolve_path(onnx_z_encoder_path) if onnx_z_encoder_path is not None else None,
            smoothing_horizon=z_smoothing_horizon,
        )
        model_name = "FBcprAuxModel_onnx"
        print(f"ONNX policy: {onnx_policy.policy_path}")
        print(f"ONNX z encoder: {onnx_policy.z_encoder_path}")
        print(f"ONNX providers: {onnx_policy.policy_session.get_providers()}")
    else:
        model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
        model.to(device)
        model.eval()
        model_name = model.__class__.__name__
    with open(inference_model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)
    resolved_config_path = inference_model_folder / "config.yaml"
    using_training_config = False
    if not no_training_config and resolved_config_path.exists():
        config["env"]["resolved_config_path"] = str(resolved_config_path.resolve())
        using_training_config = True
        print(f"Loading inference YAML config from {resolved_config_path.resolve()}")
    elif no_training_config:
        config["env"].pop("resolved_config_path", None)
        print("Skipping training YAML config; using inference config.json and hydra overrides")

    if data_path is not None:
        resolved_data_path = _resolve_path(data_path)
        if not resolved_data_path.exists():
            raise FileNotFoundError(f"Motion data file not found: {data_path} (resolved to {resolved_data_path})")
        config["env"]["lafan_tail_path"] = str(resolved_data_path)
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"
    if using_training_config:
        saved_hydra_overrides = config["env"].get("hydra_overrides", [])
        stale_group_overrides = [
            override for override in saved_hydra_overrides if _is_hydra_group_override(override)
        ]
        config["env"]["hydra_overrides"] = [
            override for override in saved_hydra_overrides if not _is_hydra_group_override(override)
        ]
        if stale_group_overrides:
            print(
                "Ignoring saved Hydra group overrides because config.yaml is already fully resolved: "
                f"{stale_group_overrides}"
            )

    hydra_overrides = config["env"].setdefault("hydra_overrides", [])
    if robot is not None:
        if robot not in ROBOT_CONFIG_OVERRIDES:
            raise ValueError(f"Unsupported robot {robot!r}. Expected one of: {sorted(ROBOT_CONFIG_OVERRIDES)}")
        _append_or_replace_hydra_override(hydra_overrides, ROBOT_CONFIG_OVERRIDES[robot])
    # import ipdb; ipdb.set_trace()
    _append_or_replace_hydra_override(hydra_overrides, "env.config.max_episode_length_s=10000")
    _append_or_replace_hydra_override(hydra_overrides, f"env.config.headless={headless}")
    _append_or_replace_hydra_override(hydra_overrides, f"simulator={simulator}")
    if simulator == "mujoco":
        if robot in (None, "g1"):
            xml_override = mujoco_xml_path or Path("g1/scene_29dof_freebase_mujoco.xml")
            _append_or_replace_hydra_override(hydra_overrides, f"robot.asset.xml_file={xml_override}")
        elif robot in ("PiPlus_S_12L8A0G2H1W_LSE", "piplus_lse"):
            # Project-local sim2sim XML includes the 23 torque motors.  The
            # source PiPlus XML has no actuator section and cannot be stepped
            # by the MuJoCo backend.
            xml_override = mujoco_xml_path or (
                HUMANOIDVERSE_DIR
                / "data/robots/piplus/mjcf/PiPlus_S_12L8A0G2H1W_LSE_260611_sim2sim.xml"
            )
            _append_or_replace_hydra_override(
                hydra_overrides,
                f"robot.asset.xml_file={_resolve_path(xml_override)}",
            )
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.asset.asset_root=humanoidverse/data/robots",
            )
            # MotionLibRobot also resolves the skeleton XML independently of
            # the simulator asset.  Point it at the copied project-local
            # PiPlus skeleton so no external ``ht_urdf`` package is needed.
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.motion.asset.assetRoot=humanoidverse/data/robots/piplus/mjcf",
            )
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.motion.asset.assetFileName=PiPlus_S_12L8A0G2H1W_LSE_260611.xml",
            )
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.bfm_encoder_fk_xml_file="
                f"{_resolve_path(HUMANOIDVERSE_DIR / 'data/robots/piplus/mjcf/PiPlus_S_12L8A0G2H1W_LSE_260424.xml')}",
            )
        elif robot in ("PiPlus_S_12L8A0G2H0W", "piplus_h0w"):
            # The H0W model's config.yaml already points at the with_armature
            # XML inside ht_urdf (asset_root=package://ht_urdf/...), where the
            # relative ../meshes/ directory resolves correctly.  MuJoCo uses
            # that XML (22 torque motors + 6 floating-base drives) as-is.
            # Clear motion.extend_config: with it non-empty the motion lib
            # emits dof_pos of size num_bodies-1 (26) instead of the 22
            # actuated joints, which no longer matches the robot DOFs.
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.motion.nums_extend_bodies=0",
            )
            _append_or_replace_hydra_override(
                hydra_overrides,
                "robot.motion.extend_config=[]",
            )
    config["env"]["device"] = env_device
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise
    print(f"Inference resolved_config_path: {config['env'].get('resolved_config_path')}")
    print(f"Inference hydra_overrides: {hydra_overrides}")
    print(f"Inference model device: {device}")
    print(f"Inference env device: {env_device}")

    # Outputs under model_folder/tracking_inference (sibling of exported/)
    if not onnx_mode:
        output_dir = model_folder / "exported"
        output_dir.mkdir(parents=True, exist_ok=True)
        export_meta_policy_as_onnx(
            model,
            output_dir,
            f"{model_name}.onnx",
            {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
            z_dim=model.cfg.archi.z_dim,
            history=('history_actor' in model.cfg.archi.actor.input_filter.key),
        )
        print(f"Exported model to {output_dir}/{model_name}.onnx")

    def tracking_inference(obs) -> torch.Tensor:
        if onnx_mode:
            onnx_policy.reset()
            return onnx_policy.encode_z(obs)
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)
    
    output_dir = model_folder / "tracking_inference"
    ros_z_source = (
        RosZSource(ros_z_topic, onnx_policy.z_dim, ros_z_timeout, ros_z_msg_type)
        if ros_z_topic
        else None
    )
    if ros_z_source is not None:
        print(
            f"Waiting for realtime z on {ros_z_topic} "
            f"(timeout={ros_z_timeout:.2f}s, expected_dim={onnx_policy.z_dim})"
        )
    joint_source = (
        RosJointTargetSource(joint_topic, wrapped_env.action_space.shape[-1], joint_timeout)
        if joint_topic
        else None
    )
    if joint_source is not None:
        print(
            f"Comparing joint targets on {joint_topic} "
            f"(timeout={joint_timeout:.2f}s, expected_dim={wrapped_env.action_space.shape[-1]})"
        )

    try:
      for MOTION_ID in motion_list:
        env.set_is_evaluating(MOTION_ID)
        loaded_motion_ids = env._motion_lib._curr_motion_ids.detach().cpu().tolist()
        loaded_motion_keys = env._motion_lib.curr_motion_keys
        local_motion_id = 0
        print(f"Requested motion id {MOTION_ID}; loaded ids {loaded_motion_ids}; loaded keys {loaded_motion_keys}")
        obs, obs_dict = get_backward_observation(env, local_motion_id, use_root_height_obs=use_root_height_obs)

        expert_qpos = np.concatenate([
            obs_dict["ref_body_pos"][:,0].cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:,0].cpu().numpy(),1,axis=-1),
            obs_dict["dof_pos"].cpu().numpy()
        ], axis=-1)

        # import ipdb; ipdb.set_trace()

        if ros_z_source is not None:
            z = ros_z_source.wait_for_first().to(env.device)[None]
            print(f"Received first realtime z on {ros_z_topic}; reference control now comes from ROS")
        elif precomputed_z_path is not None:
            resolved_z_path = _resolve_path(precomputed_z_path)
            if not resolved_z_path.exists():
                raise FileNotFoundError(f"Precomputed z file not found: {resolved_z_path}")
            z_np = np.asarray(joblib.load(resolved_z_path), dtype=np.float32)
            expected_z_dim = onnx_policy.z_dim if onnx_mode else model.cfg.archi.z_dim
            if z_np.ndim != 2 or z_np.shape[1] != expected_z_dim:
                raise ValueError(
                    f"Expected precomputed z with shape [T, {expected_z_dim}], "
                    f"got {z_np.shape} from {resolved_z_path}"
                )
            z = torch.from_numpy(z_np).to(env.device)
            print(f"Loaded precomputed z from {resolved_z_path}: {tuple(z.shape)}")
        else:
            z = tracking_inference(tree_map(lambda x: x[1:], obs))
        output_dir.mkdir(parents=True, exist_ok=True)
        if ros_z_source is None:
            joblib.dump(z.cpu().numpy(), output_dir / f"zs_{MOTION_ID}.pkl")
            print(f"Saved zs_{MOTION_ID}.pkl")

        observation, info = wrapped_env.reset(to_numpy=False)

        # Root state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3). Isaac expects quat as wxyz; motion lib uses xyzw.
        ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
        if simulator == "isaacsim":
            ref_body_rots = ref_body_rots[[3, 0, 1, 2]]  # xyzw -> wxyz for correct humanoid facing in Isaac
        ref_root_init_state = torch.cat(
                [
                    obs_dict["ref_body_pos"][0, 0],
                    ref_body_rots,
                    obs_dict["ref_body_vels"][0, 0],
                    obs_dict["ref_body_angular_vels"][0, 0],
                ]
            )
        dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
        dof_init_state[..., 0] = obs_dict["dof_pos"][0]
        dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
        target_states = {
            "dof_states": dof_init_state,
            "root_states": torch.stack([ref_root_init_state.clone() for i in range(num_envs)])
        }
        env_ids = torch.arange(num_envs, dtype=torch.long, device=wrapped_env._env.device)
        # Use the wrapper reset path, as in the built-in tracking evaluator.
        # Calling base_env.reset_envs_idx directly only stages the target state;
        # it is not pushed to MuJoCo until a later physics step.  The former
        # zero-action workaround therefore polluted the BFM last-action and
        # history inputs before the first policy call.
        observation, info = wrapped_env.reset(target_states=target_states, to_numpy=False)
        if not torch.allclose(wrapped_env._env.simulator.dof_pos[env_ids], dof_init_state[..., 0]):
            raise RuntimeError("MuJoCo reset did not apply the requested reference joint state")
        joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]
        # episode_len=500
        if episode_len is not None:
            current_episode_len = episode_len
        elif ros_z_source is not None and save_mp4:
            # Realtime-ROS mode with --save-mp4: stop when the reference
            # motion finishes so the video has a bounded length.
            current_episode_len = obs_dict["dof_pos"].shape[0]
            print(
                f"ROS z mode with --save-mp4: recording {current_episode_len} steps "
                "(reference motion length)"
            )
        elif ros_z_source is not None:
            current_episode_len = 10**9
        else:
            current_episode_len = z.shape[0]
        if ros_z_source is None and current_episode_len > z.shape[0]:
            print(f"Requested {current_episode_len} steps; cycling {z.shape[0]} inferred latent steps")
        print(f"Saving video for tracking ({current_episode_len} steps)")
        if save_mp4:
            rgb_renderer = IsaacRendererWithMuJoco(
                render_size=256,
                model=wrapped_env._env.simulator.model,
            )
            if ros_z_source is None:
                # Reference motion frames, only shown in the non-ROS comparison video.
                expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + current_episode_len])
            frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

        print(f"Running tracking inference for motion {MOTION_ID} for {current_episode_len} steps")
        joint_errs: list[np.ndarray] = []
        joint_target_log: list[np.ndarray] = []
        joint_sim_log: list[np.ndarray] = []
        try:
            for i in range(current_episode_len):
                if ros_z_source is not None:
                    latest_z, received_count = ros_z_source.latest()
                    rollout_z = latest_z.to(env.device).repeat(num_envs, 1)
                else:
                    rollout_z = z[i % len(z)].repeat(num_envs, 1)

                if joint_source is not None:
                    try:
                        target_dof, _joint_count = joint_source.latest()
                        sim_dof = wrapped_env._env.simulator.dof_pos[0]
                        err = (target_dof.to(env.device) - sim_dof).abs().detach().cpu().numpy()
                        joint_errs.append(err)
                        joint_target_log.append(target_dof.detach().cpu().numpy())
                        joint_sim_log.append(sim_dof.detach().cpu().numpy())
                    except RuntimeError:
                        pass  # target temporarily unavailable; skip this comparison step

                if onnx_mode:
                    action = onnx_policy.act(observation, rollout_z)
                else:
                    action = model.act(observation, rollout_z, mean=True)
                if i == 0:
                    print(
                        "First policy action: "
                        f"min={action.min().item():.4f}, max={action.max().item():.4f}, "
                        f"mean_abs={action.abs().mean().item():.4f}"
                    )
                observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
                joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
                if i == 0 or (i + 1) % 50 == 0:
                    dof_delta = (
                        wrapped_env._env.simulator.dof_pos - dof_init_state[..., 0]
                    ).abs().mean().item()
                    torque_mean = wrapped_env._env.torques.abs().mean().item()
                    reference_index = min(i + 1, obs_dict["dof_pos"].shape[0] - 1)
                    reference_error = (
                        wrapped_env._env.simulator.dof_pos
                        - obs_dict["dof_pos"][reference_index]
                    ).abs().mean().item()
                    print(
                        f"Step {i + 1}/{current_episode_len}: "
                        f"mean_abs_action={action.abs().mean().item():.4f}, "
                        f"mean_abs_torque={torque_mean:.4f}, "
                        f"mean_joint_delta={dof_delta:.4f}, "
                        f"mean_reference_error={reference_error:.4f}"
                        + (f", ros_z_messages={received_count}" if ros_z_source is not None else "")
                    )
                    if joint_source is not None and joint_errs:
                        latest_err = joint_errs[-1]
                        print(
                            f"           joint_target_err: mean={latest_err.mean():.4f}, "
                            f"max={latest_err.max():.4f} rad"
                        )
                if save_mp4:
                    frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        except KeyboardInterrupt:
            print(
                "\nInterrupted by Ctrl+C; ending recording and saving "
                "data/video accumulated so far"
            )
        finally:
            if joint_source is not None and joint_errs:
                _flush_joint_comparison(
                    joint_errs,
                    joint_target_log,
                    joint_sim_log,
                    wrapped_env._env.simulator.dof_names,
                    output_dir,
                    MOTION_ID,
                )
            if save_mp4:
                if ros_z_source is None:
                    # Comparison video: reference (left) + sim (right).
                    new_frames = [
                        np.concatenate([a, b], axis=1) for a, b in zip(expert_video, frames)
                    ]
                else:
                    # Realtime-ROS mode: record only the model-driven sim frames.
                    new_frames = frames
                video_name_resolved = video_name or f"tracking_{MOTION_ID}.mp4"
                if not video_name_resolved.lower().endswith(".mp4"):
                    video_name_resolved += ".mp4"
                video_path = output_dir / video_name_resolved
                media.write_video(str(video_path), new_frames, fps=video_fps)
                print(
                    f"Saved video for tracking: {video_path} "
                    f"({len(new_frames)} frames at {video_fps} fps = "
                    f"{len(new_frames) / video_fps:.1f}s; sim control dt "
                    f"{1 / (env.config.simulator.config.sim.fps / env.config.simulator.config.sim.control_decimation):.4f}s)"
                )

        joint_pos = np.stack(joint_pos, axis=0).squeeze(1)
        stats = {}

        # breakpoint()  # use PYTHONBREAKPOINT=0 to disable, or install ipdb for a nicer debugger

    finally:
        if ros_z_source is not None:
            ros_z_source.close()
        if joint_source is not None:
            joint_source.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
