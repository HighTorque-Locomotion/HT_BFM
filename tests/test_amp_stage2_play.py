import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanoidverse.agents.envs.humanoidverse_isaac import build_default_pose_target
from humanoidverse.amp_stage2_play import (
    PlaybackMetricsAccumulator,
    command_from_axes,
    latest_stage2_checkpoint,
    mean_absolute_error,
    pearson_correlation,
    regression_slope,
    resolve_play_device,
)


class AmpStage2PlayTest(unittest.TestCase):
    def test_latest_checkpoint_uses_numeric_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "checkpoint_9.pt").touch()
            (folder / "checkpoint_100.pt").touch()
            self.assertEqual(latest_stage2_checkpoint(folder).name, "checkpoint_100.pt")

    def test_command_axes_apply_asymmetric_limits(self):
        command = command_from_axes(
            {0: 0.5, 1: -1.0, 3: -0.5},
            np.asarray([-0.5, -0.2, -0.8], dtype=np.float32),
            np.asarray([1.2, 0.2, 0.8], dtype=np.float32),
            forward_axis=1,
            lateral_axis=0,
            yaw_axis=3,
            forward_sign=-1.0,
            lateral_sign=-1.0,
            yaw_sign=-1.0,
            deadzone=0.1,
        )
        self.assertAlmostEqual(float(command[0]), 1.2, places=5)
        self.assertAlmostEqual(float(command[1]), -0.2 * (0.4 / 0.9), places=5)
        self.assertAlmostEqual(float(command[2]), 0.8 * (0.4 / 0.9), places=5)

    def test_gamepad_mapping_matches_ht_lab_hi(self):
        command = command_from_axes(
            {0: -0.5, 1: -1.0, 3: 0.5},
            np.asarray([-0.5, -0.2, -0.8], dtype=np.float32),
            np.asarray([1.2, 0.2, 0.8], dtype=np.float32),
            forward_axis=1,
            lateral_axis=0,
            yaw_axis=3,
            forward_sign=-1.0,
            lateral_sign=-1.0,
            yaw_sign=-1.0,
            deadzone=0.08,
        )
        # HT_lab_hi: push left stick up => +vx, left/right => +/-vy, right stick left => +wz.
        self.assertAlmostEqual(float(command[0]), 1.2, places=5)
        self.assertGreater(float(command[1]), 0.0)
        self.assertLess(float(command[2]), 0.0)

    def test_gamepad_custom_axis_indices_are_supported(self):
        command = command_from_axes(
            {4: -1.0, 5: 0.5, 6: -0.5},
            np.asarray([-0.5, -0.2, -0.8], dtype=np.float32),
            np.asarray([1.2, 0.2, 0.8], dtype=np.float32),
            forward_axis=5,
            lateral_axis=4,
            yaw_axis=6,
            forward_sign=-1.0,
            lateral_sign=-1.0,
            yaw_sign=-1.0,
            deadzone=0.08,
        )
        self.assertAlmostEqual(float(command[0]), -0.5 * (0.42 / 0.92), places=5)
        self.assertAlmostEqual(float(command[1]), 0.2, places=5)
        self.assertAlmostEqual(float(command[2]), 0.8 * (0.42 / 0.92), places=5)

    def test_play_device_cpu_is_explicit(self):
        self.assertEqual(resolve_play_device("cpu"), torch.device("cpu"))

    def test_play_device_cuda_gets_default_index_when_available(self):
        if torch.cuda.is_available():
            self.assertEqual(resolve_play_device("cuda"), torch.device("cuda:0"))

    def test_default_pose_target_preserves_joint_order_and_velocity_zero(self):
        default = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
        target = build_default_pose_target(default, num_envs=2)
        self.assertEqual(tuple(target.shape), (2, 3, 2))
        torch.testing.assert_close(target[:, :, 0], default.expand(2, -1))
        torch.testing.assert_close(target[:, :, 1], torch.zeros(2, 3))

    def test_mean_absolute_error_uses_reference(self):
        self.assertAlmostEqual(
            mean_absolute_error(np.asarray([0.0, 1.0]), np.asarray([1.0, 0.5])), 0.75, places=6
        )

    def test_pearson_correlation_perfect_and_zero_variance(self):
        series = np.asarray([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(pearson_correlation(series, series), 1.0, places=6)
        self.assertAlmostEqual(pearson_correlation(series, -series), -1.0, places=6)
        self.assertAlmostEqual(pearson_correlation(series, np.full(4, 0.5)), 0.0, places=6)
        self.assertAlmostEqual(pearson_correlation(np.asarray([1.0]), np.asarray([1.0])), 0.0, places=6)

    def test_regression_slope_matches_ramp(self):
        values = np.asarray([0.0, 0.5, 1.0, 1.5])
        self.assertAlmostEqual(regression_slope(values, dt=0.5), 1.0, places=6)
        self.assertAlmostEqual(regression_slope(np.asarray([2.0]), dt=0.5), 0.0, places=6)

    def test_playback_metrics_summary_covers_full_metric_contract(self):
        accumulator = PlaybackMetricsAccumulator(dt=0.02)
        for step_index in range(5):
            vx = 0.1 * step_index
            accumulator.record(
                command=np.asarray([vx, 0.0, 0.0]),
                velocity=np.asarray([vx, 0.0, 0.0]),
                base_height=0.75 + 0.01 * step_index,
                projected_gravity=9.81,
                foot_contact=0.5,
                foot_slip=0.0,
                action_rate=0.02,
                joint_acc=0.1,
                terminated=(step_index == 4),
            )
        summary = accumulator.summary()
        self.assertEqual(set(summary), set(PlaybackMetricsAccumulator.METRIC_NAMES))
        self.assertAlmostEqual(summary["episode_length"], 5.0, places=6)
        self.assertAlmostEqual(summary["termination"], 1.0, places=6)
        self.assertAlmostEqual(summary["base_height"], 0.77, places=6)
        self.assertAlmostEqual(summary["tracking_vx_mae"], 0.0, places=6)
        self.assertAlmostEqual(summary["tracking_yaw_rate_mae"], 0.0, places=6)
        # vx ramps at 0.1 per 0.02s step; vy/yaw commands stay constant so only the
        # vx axis contributes to the correlation.
        self.assertAlmostEqual(summary["response_slope"], 5.0, places=6)
        self.assertAlmostEqual(summary["command_correlation"], 1.0, places=6)
        self.assertAlmostEqual(summary["action_rate"], 0.02, places=6)
        self.assertAlmostEqual(summary["foot_contact"], 0.5, places=6)
        formatted = accumulator.format_summary()
        for name in PlaybackMetricsAccumulator.METRIC_NAMES:
            self.assertIn(f"{name}=", formatted)

    def test_playback_metrics_reset_clears_running_state(self):
        accumulator = PlaybackMetricsAccumulator(dt=0.02)
        accumulator.record(
            command=np.asarray([0.3, 0.0, 0.0]),
            velocity=np.asarray([0.1, 0.0, 0.0]),
            base_height=0.75,
            projected_gravity=9.81,
            foot_contact=0.5,
            foot_slip=0.0,
            action_rate=0.02,
            joint_acc=0.1,
            terminated=True,
        )
        accumulator.reset()
        self.assertFalse(accumulator.has_data())
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["episode_length"], 0.0, places=6)
        self.assertAlmostEqual(summary["termination"], 0.0, places=6)

    def test_playback_metrics_episode_length_averages_completed_episodes(self):
        accumulator = PlaybackMetricsAccumulator(dt=0.02)
        for step_index in range(5):
            accumulator.record(
                command=np.asarray([0.0, 0.0, 0.0]),
                velocity=np.asarray([0.0, 0.0, 0.0]),
                base_height=0.75,
                projected_gravity=0.0,
                foot_contact=0.5,
                foot_slip=0.0,
                action_rate=0.02,
                joint_acc=0.1,
                terminated=(step_index in (2, 4)),
            )
        # Episodes of length 3 and 2 completed; the mean is reported.
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["episode_length"], 2.5, places=6)
        self.assertAlmostEqual(summary["termination"], 1.0, places=6)

    def test_playback_metrics_constant_command_has_zero_correlation(self):
        accumulator = PlaybackMetricsAccumulator(dt=0.02)
        for step_index in range(5):
            accumulator.record(
                command=np.asarray([0.0, 0.0, 0.0]),
                velocity=np.asarray([0.1 * step_index, 0.0, 0.0]),
                base_height=0.75,
                projected_gravity=0.0,
                foot_contact=0.5,
                foot_slip=0.0,
                action_rate=0.02,
                joint_acc=0.1,
                terminated=False,
            )
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["command_correlation"], 0.0, places=6)
        self.assertAlmostEqual(summary["tracking_vx_mae"], 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
