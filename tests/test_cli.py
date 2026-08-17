import sys
import unittest
from unittest.mock import patch

import cli


class CliTests(unittest.TestCase):
    @patch("cli.display_config")
    @patch("cli.recommend_settings", return_value={"gpu_offload": 999})
    @patch("cli.ask_model_profile", return_value={"model_name": "test-model"})
    @patch(
        "cli.get_hardware_profile",
        return_value={"gpu": "AMD Radeon RX 7900 XTX", "gpu_memory_mb": 24576},
    )
    def test_main_passes_detected_hardware_to_recommender(
        self,
        get_hardware_profile,
        ask_model_profile,
        recommend_settings,
        display_config,
    ):
        with patch.object(sys, "argv", ["cli.py"]):
            cli.main()

        hardware = get_hardware_profile.return_value
        profile = ask_model_profile.return_value
        recommend_settings.assert_called_once_with(hardware, profile)
        display_config.assert_called_once_with({"gpu_offload": 999})


if __name__ == "__main__":
    unittest.main()
