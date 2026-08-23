import unittest

import torch
import torch.nn as nn

from riser.inference.steered_model import SteeredModel
from riser.router import Router, RouterConfig, RouterInference


class TinyLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class TinyGenerationModel(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.fail = fail

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def generate(self, input_ids, **kwargs):
        if self.fail:
            raise RuntimeError("generation failed")
        hidden = input_ids.float()
        for _ in range(3):
            hidden = self.forward(hidden)
        return hidden


class CountingRouterInference(RouterInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def inject_activation(self, hidden_state):
        self.calls += 1
        return super().inject_activation(hidden_state)


def make_router_inference():
    router = Router(RouterConfig(hidden_size=2, num_primitives=1, bottleneck_dim=4))
    with torch.no_grad():
        router.selection_head.weight.zero_()
        router.selection_head.bias.fill_(8.0)
        router.strength_head.weight.zero_()
        router.strength_head.bias.zero_()
    return CountingRouterInference(
        router=router,
        primitive_library=torch.tensor([[1.0, 0.0]]),
        target_layer=0,
        device="cpu",
    )


class SteeredModelTests(unittest.TestCase):
    def test_generate_caches_route_and_removes_hook(self):
        base = TinyGenerationModel()
        routing = make_router_inference()
        model = SteeredModel(base, routing, cache_routing=True)

        output = model.generate(torch.zeros(1, 3, 2))

        self.assertEqual(output.shape, (1, 3, 2))
        self.assertEqual(routing.calls, 1)
        self.assertEqual(len(base.model.layers[0]._forward_hooks), 0)
        self.assertEqual(model.get_last_routing_info()["selected_primitives"], [[0]])

    def test_generation_exception_still_removes_hook(self):
        base = TinyGenerationModel(fail=True)
        model = SteeredModel(base, make_router_inference(), cache_routing=True)

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            model.generate(torch.zeros(1, 2, 2))

        self.assertEqual(len(base.model.layers[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
