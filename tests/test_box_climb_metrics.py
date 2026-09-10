import unittest

import torch

from humanoidverse.agents.evaluations.humanoidverse_isaac import compute_box_climb_metrics


class BoxClimbMetricsTest(unittest.TestCase):
    def test_success_is_origin_invariant_and_reports_top_reach(self):
        target = torch.zeros(30, 3)
        target[:, 0] = 0.0
        target[:, 1] = torch.linspace(-0.60, 0.50, 30)
        target[:, 2] = torch.linspace(0.45, 0.65, 30)
        actual = target[:, None, :] + torch.tensor([3.0, -4.0, 2.0])

        metrics = compute_box_climb_metrics(actual, target)

        self.assertEqual(metrics["box_climb_success"], 1.0)
        self.assertEqual(metrics["box_crossed_far_side"], 1.0)
        self.assertGreater(metrics["box_max_root_height_m"], 0.60)
        self.assertGreater(metrics["box_top_reach_ratio"], 0.0)

    def test_failure_without_far_side_crossing(self):
        target = torch.zeros(30, 3)
        target[:, 1] = torch.linspace(-0.60, 0.10, 30)
        target[:, 2] = 0.60

        metrics = compute_box_climb_metrics(target[:, None, :], target)

        self.assertEqual(metrics["box_crossed_far_side"], 0.0)
        self.assertEqual(metrics["box_climb_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
