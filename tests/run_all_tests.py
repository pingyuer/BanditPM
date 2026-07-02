"""
Test runner for the project.

Runs all unit tests and reports results.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def discover_and_run_tests():
    """Discover and run all tests in the tests directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).resolve().parent),
        pattern="test_*.py",
        top_level_dir=str(Path(__file__).resolve().parents[1]),
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = discover_and_run_tests()
    sys.exit(0 if success else 1)
