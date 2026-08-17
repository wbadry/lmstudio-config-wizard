import unittest
from unittest.mock import patch

import hardware_utils


class NvidiaSmiParsingTests(unittest.TestCase):
    def test_parses_name_and_numeric_memory(self):
        output = "NVIDIA GeForce RTX 5090 Laptop GPU, 24463\n"

        self.assertEqual(
            hardware_utils._parse_nvidia_smi(output),
            [("NVIDIA GeForce RTX 5090 Laptop GPU", 24463)],
        )

    def test_ignores_malformed_rows(self):
        output = "malformed\nNVIDIA GPU, not available\n"

        self.assertEqual(hardware_utils._parse_nvidia_smi(output), [])


class WindowsGpuParsingTests(unittest.TestCase):
    def test_parses_amd_registry_data(self):
        output = (
            '[{"name":"AMD Radeon RX 7900 XTX",'
            '"memory_bytes":25769803776}]'
        )

        self.assertEqual(
            hardware_utils._parse_windows_gpu_json(output),
            [("AMD Radeon RX 7900 XTX", 24576)],
        )

    def test_accepts_single_adapter_json_object(self):
        output = '{"name":"Intel Arc A770","memory_bytes":17179869184}'

        self.assertEqual(
            hardware_utils._parse_windows_gpu_json(output),
            [("Intel Arc A770", 16384)],
        )

    def test_rejects_invalid_json(self):
        self.assertEqual(hardware_utils._parse_windows_gpu_json("not json"), [])


class LinuxGpuParsingTests(unittest.TestCase):
    def test_parses_amd_lspci_output(self):
        output = (
            "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. "
            "[AMD/ATI] Navi 31 [Radeon RX 7900 XTX] (rev c8)\n"
        )

        self.assertEqual(
            hardware_utils._parse_lspci(output),
            [
                (
                    "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 "
                    "[Radeon RX 7900 XTX]",
                    0,
                )
            ],
        )


class GpuSelectionTests(unittest.TestCase):
    @patch("hardware_utils.platform.system", return_value="Windows")
    @patch("hardware_utils._get_windows_gpus")
    @patch("hardware_utils._get_nvidia_gpus", return_value=[])
    def test_windows_falls_back_to_cross_vendor_detection(
        self, _nvidia_gpus, windows_gpus, _system
    ):
        windows_gpus.return_value = [
            ("Intel Graphics", 2048),
            ("AMD Radeon RX 7900 XTX", 24576),
        ]

        self.assertEqual(
            hardware_utils.get_gpu_info(),
            ("AMD Radeon RX 7900 XTX", 24576),
        )

    @patch("hardware_utils.platform.system", return_value="Windows")
    @patch("hardware_utils._get_windows_gpus", return_value=[])
    @patch("hardware_utils._get_nvidia_gpus", return_value=[])
    def test_returns_stable_defaults_when_no_probe_succeeds(
        self, _nvidia_gpus, _windows_gpus, _system
    ):
        self.assertEqual(hardware_utils.get_gpu_info(), ("Unknown", 0))


if __name__ == "__main__":
    unittest.main()
