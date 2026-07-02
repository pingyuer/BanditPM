import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LEGACY_TEST_FILES = {
    "test_anchor_ode.py",
    "test_delay_ode.py",
    "test_dynakey_q_loss_integration.py",
    "test_dynakey_qlearning_semantics.py",
    "test_faf_losses.py",
    "test_functional_anchor.py",
    "test_functional_anchor_losses.py",
    "test_spatial_dynakey.py",
    "test_unext_faf.py",
}


def pytest_ignore_collect(collection_path, config):
    if os.environ.get("RUN_LEGACY_TESTS") == "1":
        return False
    return Path(str(collection_path)).name in LEGACY_TEST_FILES
