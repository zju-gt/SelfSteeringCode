import math
import unittest

import torch

from riser.router import Router, RouterConfig


class RouterTests(unittest.TestCase):
    def make_router(self):
        return Router(
            RouterConfig(
                hidden_size=4,
                num_primitives=3,
                bottleneck_dim=8,
            )
        )

    def test_forward_supports_vector_and_batch_inputs(self):
        router = self.make_router()

        mask, strength, probs, logits, features = router(torch.zeros(2, 4), hard=True)

        self.assertEqual(mask.shape, (2, 3))
        self.assertEqual(strength.shape, (2, 3))
        self.assertEqual(probs.shape, (2, 3))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(features.shape, (2, 8))
        self.assertTrue(torch.all(strength >= 0).item())
        self.assertTrue(torch.all(strength <= 2).item())

        vector_outputs = router(torch.zeros(4), hard=True)
        self.assertEqual(vector_outputs[0].shape, (1, 3))
        self.assertEqual(vector_outputs[1].shape, (1, 3))

    def test_hard_selection_uses_configured_threshold(self):
        router = self.make_router()
        with torch.no_grad():
            router.selection_head.weight.zero_()
            router.selection_head.bias.copy_(
                torch.tensor([
                    math.log(0.6 / 0.4),
                    math.log(0.8 / 0.2),
                    math.log(0.7 / 0.3),
                ])
            )

        _, _, probs, _, _ = router(torch.zeros(4), hard=True)
        hard_mask = router(torch.zeros(4), hard=True)[0]

        self.assertAlmostEqual(probs[0, 0].item(), 0.6, places=5)
        self.assertEqual(hard_mask.tolist(), [[0.0, 1.0, 1.0]])

    def test_route_composes_selected_vectors(self):
        router = self.make_router()
        with torch.no_grad():
            router.selection_head.weight.zero_()
            router.selection_head.bias.copy_(torch.tensor([6.0, -6.0, 6.0]))
            router.strength_head.weight.zero_()
            router.strength_head.bias.zero_()

        primitives = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        injection, info = router.route(torch.zeros(4), primitives, hard=True)

        expected = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(injection, expected, atol=1e-3, rtol=1e-3)
        self.assertEqual(info["selected_primitives"], [[0, 2]])

    def test_config_round_trip(self):
        config = RouterConfig(
            hidden_size=16,
            num_primitives=5,
            bottleneck_dim=12,
            selection_threshold=0.65,
            max_strength=1.5,
            strength_temperature=0.8,
        )
        restored = RouterConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)


if __name__ == "__main__":
    unittest.main()
