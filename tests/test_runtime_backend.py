import pathlib
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import runtime_backend


class RuntimeBackendTests(unittest.TestCase):
    def test_integrated_graphics_names_are_classified(self):
        self.assertTrue(runtime_backend.is_integrated_graphics_name("Intel(R) UHD Graphics 770"))
        self.assertTrue(runtime_backend.is_integrated_graphics_name("AMD Radeon(TM) Graphics"))
        self.assertFalse(runtime_backend.is_integrated_graphics_name("NVIDIA GeForce RTX 4060 Ti"))

    def test_directml_default_prefers_non_integrated_device(self):
        fake = types.SimpleNamespace(
            device_count=lambda: 2,
            device=lambda index: f"privateuseone:{index}",
            device_name=lambda index: ["AMD Radeon(TM) Graphics", "AMD Radeon RX 7800 XT"][index],
        )
        with mock.patch.object(runtime_backend.importlib, "import_module", return_value=fake):
            backend = runtime_backend._create_directml_backend("dml")
        self.assertEqual("privateuseone:1", str(backend.torch_device))
        self.assertEqual("AMD Radeon RX 7800 XT", backend.label)

    def test_directml_preferred_name_overrides_default(self):
        fake = types.SimpleNamespace(
            device_count=lambda: 2,
            device=lambda index: f"privateuseone:{index}",
            device_name=lambda index: ["GPU A", "GPU B"][index],
        )
        with mock.patch.object(runtime_backend.importlib, "import_module", return_value=fake):
            backend = runtime_backend._create_directml_backend("dml", "GPU B")
        self.assertEqual("privateuseone:1", str(backend.torch_device))


if __name__ == "__main__":
    unittest.main()
