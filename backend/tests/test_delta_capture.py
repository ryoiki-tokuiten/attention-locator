from __future__ import annotations

import unittest

import torch

from backend.delta_capture import (
    build_delta_storage_key,
    estimate_delta_storage_bytes,
    repeat_key_value_heads,
)


class DeltaCaptureHelpersTestCase(unittest.TestCase):
    def test_storage_key_is_stable_for_same_text(self):
        self.assertEqual(
            build_delta_storage_key("model-a", "Hello world"),
            build_delta_storage_key("model-a", "Hello world"),
        )

    def test_storage_estimate_accounts_for_all_tensor_groups(self):
        estimate = estimate_delta_storage_bytes(
            layer_count=2,
            token_count=4,
            head_count=3,
            hidden_size=5,
            head_dim=7,
            dtype="float16",
        )

        self.assertGreater(estimate, 0)
        self.assertEqual(estimate, 4 * 4 + (2 * 4 * 5 * 2) + (2 * 3 * 4 * 7 * 2) + (2 * 3 * 4 * 5 * 2) + (2 * 4 * 5 * 2) + (2 * 4 * 5 * 2) + (2 * 4 * 5 * 2))

    def test_repeat_key_value_heads_matches_group_repeat_shape(self):
        value_states = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
        repeated = repeat_key_value_heads(value_states, 3)

        self.assertEqual(tuple(repeated.shape), (2, 3, 3, 4))
        self.assertTrue(torch.equal(repeated[:, 0], repeated[:, 1]))
        self.assertTrue(torch.equal(repeated[:, 1], repeated[:, 2]))


if __name__ == "__main__":
    unittest.main()
