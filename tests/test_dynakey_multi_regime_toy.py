import unittest

import torch

from model.modules.dynakey import DynaKeyMemoryCore, ODEKeyDictionary


def value_from_z(z):
    return z.view(1, 1, 2, 1, 1).expand(1, 1, 2, 2, 2).contiguous()


class DynaKeyMultiRegimeToyTests(unittest.TestCase):
    def test_fixed_residual_grows_multi_key_and_beats_single_key(self):
        states = []
        z = torch.tensor([0.0, 0.0])
        for step in [torch.tensor([1.0, 0.0])] * 4 + [torch.tensor([0.0, 1.0])] * 4 + [torch.tensor([-1.0, 0.0])] * 4:
            states.append(z.clone())
            z = z + step
        states.append(z.clone())

        core = DynaKeyMemoryCore(
            {
                "BANK_SIZE": 6,
                "POLICY_MODE": "fixed_residual",
                "RESIDUAL_SPAWN_THRESHOLD": 0.05,
                "SPLIT_EPS": 0.1,
                "EMA_ALPHA": 1.0,
            },
            value_dim=2,
        )
        core.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        mask = torch.ones(1, 1, 2, 2)
        preds = []
        for state in states[:-1]:
            _, aux = core(value_from_z(state), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), mask)
            preds.append(core._prev_pred.clone())

        multi_error = torch.stack([
            torch.mean((preds[i][0, 0] - states[i + 1]) ** 2) for i in range(3, len(preds) - 1)
        ]).mean()
        self.assertGreater(core.dictionary.active_key_count().item(), 1)

        single = ODEKeyDictionary(value_dim=2, bank_size=1, ema_alpha=0.0)
        single.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))
        single.spawn(states[0].view(1, 1, 2), torch.tensor([[[1.0, 0.0]]]))
        single_errors = []
        for state, target in zip(states[3:-1], states[4:]):
            pred, _ = single.predict(state.view(1, 1, 2))
            single_errors.append(torch.mean((pred[0, 0] - target) ** 2))
        single_error = torch.stack(single_errors).mean()
        self.assertLess(multi_error.item(), single_error.item())


if __name__ == "__main__":
    unittest.main()
