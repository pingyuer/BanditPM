import os
import unittest
from unittest import mock

from utils.ddp import distributed_setup


class DistributedSetupTests(unittest.TestCase):
    def test_single_process_fallback_without_torchrun_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("utils.ddp.dist.is_initialized", return_value=False):
                with mock.patch("utils.ddp.torch.cuda.is_available", return_value=False):
                    local_rank, world_size = distributed_setup()

        self.assertEqual(local_rank, 0)
        self.assertEqual(world_size, 1)

    def test_init_process_group_when_torchrun_env_present(self):
        env = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("utils.ddp.dist.is_initialized", return_value=False):
                with mock.patch("utils.ddp.torch.cuda.is_available", return_value=True):
                    with mock.patch("utils.ddp.torch.cuda.set_device") as set_device:
                        with mock.patch("utils.ddp.dist.init_process_group") as init_pg:
                            local_rank, world_size = distributed_setup()

        self.assertEqual(local_rank, 1)
        self.assertEqual(world_size, 2)
        set_device.assert_called_once_with(1)
        init_pg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
