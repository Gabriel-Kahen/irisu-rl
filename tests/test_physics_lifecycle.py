from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import NativeError, NativeSimulator, SyncVectorEnv, find_library  # noqa: E402


try:
    LIBRARY = find_library()
except NativeError:
    LIBRARY = None


@unittest.skipIf(LIBRARY is None, "build the native shared library before lifecycle stress")
class PhysicsLifecycleStressTests(unittest.TestCase):
    def test_repeated_create_configure_reset_hash_destroy(self) -> None:
        expected_config = None
        for _ in range(500):
            with NativeSimulator(LIBRARY, config={"gravity_y": 360.0}) as simulator:
                simulator.reset(123)
                config_hash = simulator.config_hash()
                expected_config = config_hash if expected_config is None else expected_config
                self.assertEqual(config_hash, expected_config)
                simulator.step(0, 0.0, 0.0, 2)
                self.assertNotEqual(simulator.state_hash(), 0)

    def test_many_simultaneous_vector_handles(self) -> None:
        for _ in range(10):
            with SyncVectorEnv(32, library_path=LIBRARY) as vector:
                vector.reset(seed=[91] * 32)
                self.assertEqual(len(set(vector.state_hash())), 1)
                vector.step([{"kind": 0, "wait_ticks": 5}] * 32)
                self.assertEqual(len(set(vector.state_hash())), 1)

    def test_concurrent_independent_c_abi_handles(self) -> None:
        def trace(_: int) -> tuple[int, int]:
            expected = None
            for _ in range(50):
                with NativeSimulator(LIBRARY) as simulator:
                    simulator.reset(77)
                    simulator.step(0, 0.0, 0.0, 5)
                    value = (simulator.config_hash(), simulator.state_hash())
                    expected = value if expected is None else expected
                    if value != expected:
                        raise AssertionError("deterministic handle trace changed")
            assert expected is not None
            return expected

        with ThreadPoolExecutor(max_workers=8) as pool:
            traces = list(pool.map(trace, range(8)))
        self.assertEqual(len(set(traces)), 1)


if __name__ == "__main__":
    unittest.main()
