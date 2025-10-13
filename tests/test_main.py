# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch

import pytest

from aiperf.gpu_telemetry.constants import DEFAULT_DCGM_ENDPOINT


class TestMainFunction:
    """Test the main() function from __main__.py"""

    @pytest.fixture(autouse=True)
    def _restore_argv(self):
        """Save and restore sys.argv after each test"""
        original = sys.argv.copy()
        yield
        sys.argv = original

    @pytest.mark.parametrize(
        "argv,expected_argv,expected_gpu_telemetry",
        [
            # --gpu-telemetry at end without value
            (
                ["--gpu-telemetry"],
                ["--gpu-telemetry", DEFAULT_DCGM_ENDPOINT],
                [DEFAULT_DCGM_ENDPOINT],
            ),
            # --gpu-telemetry followed by single dash flag
            (
                ["--gpu-telemetry", "-v"],
                ["--gpu-telemetry", DEFAULT_DCGM_ENDPOINT, "-v"],
                [DEFAULT_DCGM_ENDPOINT],
            ),
            # --gpu-telemetry followed by double dash flag
            (
                ["--gpu-telemetry", "--verbose"],
                ["--gpu-telemetry", DEFAULT_DCGM_ENDPOINT, "--verbose"],
                [DEFAULT_DCGM_ENDPOINT],
            ),
            # --gpu-telemetry with custom value
            (
                ["--gpu-telemetry", "http://custom:9401"],
                ["--gpu-telemetry", "http://custom:9401"],
                ["http://custom:9401"],
            ),
            # No --gpu-telemetry flag
            (
                [],
                [],
                None,
            ),
        ],
    )  # fmt: skip
    def test_gpu_telemetry_argument_handling(
        self, argv, expected_argv, expected_gpu_telemetry
    ):
        """Test --gpu-telemetry flag handling with various argument combinations"""
        from aiperf.__main__ import main

        captured_user_config = None

        def mock_run_system_controller(user_config, service_config):
            nonlocal captured_user_config
            captured_user_config = user_config

        sys.argv = ["aiperf", "profile", "-m", "test-model", *argv]

        with patch(
            "aiperf.cli_runner.run_system_controller",
            side_effect=mock_run_system_controller,
        ):
            main()

        assert sys.argv == ["aiperf", "profile", "-m", "test-model", *expected_argv]
        assert captured_user_config.gpu_telemetry == expected_gpu_telemetry
