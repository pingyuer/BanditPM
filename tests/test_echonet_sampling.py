import unittest

from tools.echonet_sampling import build_sample_plan


class EchoNetSamplingTests(unittest.TestCase):
    def test_ed_to_es_keeps_endpoint_labels(self):
        plan = build_sample_plan([10, 19], frame_count=40, num_frames=10, mode="ed_to_es")
        self.assertEqual(plan.indices[0], 10)
        self.assertEqual(plan.indices[-1], 19)
        self.assertEqual(plan.label_indices, [0, 9])
        self.assertEqual(plan.window_end, 19)

    def test_full_cycle_expands_window_and_repositions_es_label(self):
        plan = build_sample_plan([10, 19], frame_count=40, num_frames=10, mode="full_cycle")
        self.assertEqual(plan.indices[0], 10)
        self.assertEqual(plan.indices[-1], 28)
        self.assertEqual(plan.window_end, 28)
        self.assertEqual(plan.label_indices, [0, 5])
        self.assertEqual(plan.indices[5], 19)

    def test_full_cycle_uses_fixed_anchor_positions(self):
        plan_a = build_sample_plan([10, 19], frame_count=40, num_frames=10, mode="full_cycle")
        plan_b = build_sample_plan([20, 31], frame_count=80, num_frames=10, mode="full_cycle")

        self.assertEqual(plan_a.label_indices, [0, 5])
        self.assertEqual(plan_b.label_indices, [0, 5])
        self.assertEqual(plan_a.indices[5], 19)
        self.assertEqual(plan_b.indices[5], 31)

    def test_full_cycle_clips_to_video_end(self):
        plan = build_sample_plan([10, 19], frame_count=25, num_frames=10, mode="full_cycle")
        self.assertEqual(plan.window_end, 24)
        self.assertEqual(plan.indices[-1], 24)
        self.assertEqual(plan.label_indices, [0, 5])


if __name__ == "__main__":
    unittest.main()
