import unittest

import torch
import torch.nn as nn

from riser.inference.hooks import ActivationInjectionHook


class TinyLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ActivationInjectionHookTests(unittest.TestCase):
    def test_only_last_sequence_position_is_injected(self):
        model = TinyModel()
        hook = ActivationInjectionHook(
            injection_fn=lambda hidden: (hidden + 3.0, {"called": True}),
            target_layer=0,
        )
        original = torch.zeros(1, 3, 2)
        hook.register(model)
        try:
            injected = model(original)
        finally:
            hook.remove()

        torch.testing.assert_close(injected[:, :-1, :], original[:, :-1, :])
        torch.testing.assert_close(injected[:, -1, :], torch.full((1, 2), 3.0))
        self.assertEqual(hook.get_last_info(), {"called": True})

    def test_remove_makes_later_forward_unchanged(self):
        model = TinyModel()
        hook = ActivationInjectionHook(
            injection_fn=lambda hidden: (hidden + 1.0, {}),
            target_layer=0,
        )
        original = torch.zeros(1, 2, 2)
        hook.register(model)
        injected = model(original)
        hook.remove()
        unchanged = model(original)

        self.assertFalse(torch.equal(injected, original))
        torch.testing.assert_close(unchanged, original)


if __name__ == "__main__":
    unittest.main()
