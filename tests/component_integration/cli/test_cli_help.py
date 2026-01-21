# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version as get_package_version

import pytest

from tests.harness.utils import AIPerfCLI


@pytest.fixture
def disabled_parameters() -> list[str]:
    """Parameters that should NOT appear in help due to DisableCLI()."""
    return [
        "--service-config.zmq-tcp.event-bus-proxy-config.frontend-port",
        "--service-config.zmq-tcp.event-bus-proxy-config.backend-port",
        "--service-config.zmq-tcp.event-bus-proxy-config.control-port",
        "--service-config.zmq-tcp.event-bus-proxy-config.capture-port",
        "--service-config.zmq-tcp.dataset-manager-proxy-config.frontend-port",
        "--service-config.zmq-tcp.dataset-manager-proxy-config.backend-port",
        "--service-config.zmq-tcp.dataset-manager-proxy-config.control-port",
        "--service-config.zmq-tcp.dataset-manager-proxy-config.capture-port",
        "--service-config.zmq-tcp.raw-inference-proxy-config.frontend-port",
        "--service-config.zmq-tcp.raw-inference-proxy-config.backend-port",
        "--service-config.zmq-tcp.raw-inference-proxy-config.control-port",
        "--service-config.zmq-tcp.raw-inference-proxy-config.capture-port",
    ]


class TestCLIHelp:
    def test_profile_help_does_not_show_parameters(self, cli: AIPerfCLI):
        """Ensure help text for profile command does not show miscellaneous un-grouped parameters."""
        result = cli.run_sync("aiperf profile -h", assert_success=False)
        assert "─ Parameters ─" not in result.stdout

    def test_no_args_does_not_crash(self, cli: AIPerfCLI):
        """Ensure CLI does not crash when no arguments are provided."""
        result = cli.run_sync("aiperf", assert_success=False)
        assert "Usage: aiperf COMMAND" in result.stdout
        assert "─ Commands ─" in result.stdout

    def test_disable_cli_parameters_not_in_help(
        self, cli: AIPerfCLI, disabled_parameters: list[str]
    ):
        """Test that parameters marked with DisableCLI() are not shown in help text."""
        result = cli.run_sync("aiperf profile -h", assert_success=False)
        for param in disabled_parameters:
            assert param not in result.stdout, (
                f"DisableCLI parameter '{param}' should not appear in help text"
            )


class TestCLIVersion:
    def test_version_flag(self, cli: AIPerfCLI):
        """Test that --version returns the correct package version."""
        result = cli.run_sync("aiperf --version", assert_success=False)
        assert result.exit_code == 0

        expected_version = get_package_version("aiperf")
        actual_version = result.stdout.strip()
        assert actual_version == expected_version
