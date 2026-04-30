import unittest

import torch
from omegaconf import OmegaConf

from model.modules.memory_core import MemoryCore


class DynaKeyMinimalIntegrationTests(unittest.TestCase):
    def test_memory_core_dynakey_smoke_clip(self):
        cfg = OmegaConf.create(
            {
                "type": "dynakey",
                "dynakey": {
                    "BANK_SIZE": 3,
                    "DT": 1.0,
                    "EMA_ALPHA": 0.5,
                    "HIDDEN_DIM": 16,
                    "GATE_INIT": 1.0,
                },
            }
        )
        core = MemoryCore(
            value_dim=4,
            key_dim=2,
            temporal_memory_cfg=OmegaConf.create({"type": "dynakey", "dynakey": cfg.dynakey}),
            memory_core_cfg=cfg,
        )
        core.reset_state(batch_size=1, num_objects=1, device=torch.device("cpu"))

        value = torch.randn(1, 1, 4, 2, 2)
        key = torch.randn(1, 2, 2, 2)
        pix = torch.randn(1, 4, 2, 2)
        mask = torch.ones(1, 1, 8, 8)

        for _ in range(3):
            readout, aux = core(value, key, pix, mask, policy_meta={"training": True})
            self.assertEqual(readout.shape, value.shape)
            self.assertTrue(torch.isfinite(readout).all())
            self.assertEqual(aux["memory_type"], "dynakey")
            self.assertIn("dynakey_aux", aux)
            value = value + 0.1


if __name__ == "__main__":
    unittest.main()
