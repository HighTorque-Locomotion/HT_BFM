"""ROS2-to-stdout bridge for Conda environments with a different Python ABI."""

import json
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String


def _import_bfm_msgs():
    """Import the custom motion_target_msgs BFMZ message (system ROS Python)."""
    try:
        from motion_target_msgs.msg import BFMZ

        return BFMZ
    except ImportError as exc:
        raise ImportError(
            "motion_target_msgs is required for BFMZ topics. Source the "
            "instinct_onboard workspace (e.g. `source ws_instinct_onboard/install/setup.bash`) "
            "so the system Python can import motion_target_msgs."
        ) from exc


class ZBridge(Node):
    def __init__(self, topic: str, msg_type: str = "float_array"):
        super().__init__("humanoidverse_z_stdout_bridge")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        if msg_type == "bfm_z":
            self._bfm_z_type = _import_bfm_msgs()
            self.subscription = self.create_subscription(
                self._bfm_z_type, topic, self._callback, qos
            )
        elif msg_type == "string":
            # String topics carry a JSON payload (e.g. the joint target frame).
            self.subscription = self.create_subscription(
                String, topic, self._callback, qos
            )
        else:
            self.subscription = self.create_subscription(
                Float32MultiArray, topic, self._callback, qos
            )

    def _callback(self, message) -> None:
        # One compact JSON array per line keeps the parent process independent
        # of ROS and avoids importing ROS's Python extension in Conda Python.
        if isinstance(message, String):
            sys.stdout.write(message.data + "\n")
        elif hasattr(self, "_bfm_z_type") and isinstance(message, self._bfm_z_type):
            sys.stdout.write(json.dumps(list(message.z), separators=(",", ":")) + "\n")
        else:
            sys.stdout.write(json.dumps(list(message.data), separators=(",", ":")) + "\n")
        sys.stdout.flush()


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "/bfm_z_realtime"
    msg_type = sys.argv[2] if len(sys.argv) > 2 else "float_array"
    rclpy.init()
    node = ZBridge(topic, msg_type)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        # Ctrl+C raises KeyboardInterrupt; SIGTERM raises ExternalShutdownException.
        # Either way the signal handler already shut the context down, so just exit.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
