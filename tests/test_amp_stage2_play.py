import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoidverse.amp_stage2_play import command_from_axes, latest_stage2_checkpoint


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


if __name__ == "__main__":
    unittest.main()
