from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent._nogil import NoGILError, ensure_nogil, is_nogil


class NoGILTests(unittest.TestCase):
    def test_current_test_interpreter_is_free_threaded(self) -> None:
        self.assertTrue(is_nogil())
        self.assertTrue(ensure_nogil(strict=True))

    def test_strict_guard_fails_when_gil_enabled(self) -> None:
        with mock.patch("sysconfig.get_config_var", return_value=0):
            with self.assertRaises(NoGILError):
                ensure_nogil(strict=True)
            self.assertFalse(ensure_nogil(strict=False))


if __name__ == "__main__":
    unittest.main()
