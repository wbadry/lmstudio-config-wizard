import unittest

from recommender import _parse_gpu_memory_mb, recommend_settings


class GpuMemoryParsingTests(unittest.TestCase):
    def test_accepts_numeric_memory(self):
        self.assertEqual(_parse_gpu_memory_mb(24576), 24576)

    def test_accepts_legacy_formatted_memory(self):
        self.assertEqual(_parse_gpu_memory_mb("8192 MB"), 8192)

    def test_invalid_memory_defaults_to_zero(self):
        self.assertEqual(_parse_gpu_memory_mb("Unknown"), 0)


class GpuRecommendationTests(unittest.TestCase):
    def test_capable_gpu_enables_offload_and_flash_attention(self):
        hardware = {
            "ram_gb": 32,
            "logical_cores": 16,
            "gpu": "AMD Radeon RX 7900 XTX",
            "gpu_memory_mb": 24576,
        }
        model = {"model_size": "4-8 GB", "goal": "Balanced/general purpose"}

        config = recommend_settings(hardware, model)

        self.assertEqual(config["gpu_offload"], 999)
        self.assertTrue(config["flash_attention"])


if __name__ == "__main__":
    unittest.main()
