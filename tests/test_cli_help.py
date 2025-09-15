# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest
from cyclopts.exceptions import UnknownOptionError

from aiperf.cli import app


@pytest.fixture
def disabled_parameters() -> list[str]:
    """Parameters that should NOT appear in help due to DisableCLI()"""
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
    def test_profile_help_does_not_show_parameters(self, capsys):
        """This test is to ensure that the help text for the profile command does
        not show miscellaneous un-grouped parameters."""
        app(["profile", "-h"])
        assert "─ Parameters ─" not in capsys.readouterr().out

    def test_no_args_does_not_crash(self, capsys):
        """This test is to ensure that the CLI does not crash when no arguments are provided."""
        app([])
        out = capsys.readouterr().out
        assert "Usage: aiperf COMMAND" in out
        assert "─ Commands ─" in out

    def test_disable_cli_parameters_not_in_help(self, capsys, disabled_parameters):
        """Test that certain parameters marked with DisableCLI() are not shown in help text."""
        app(["profile", "-h"])
        help_output = capsys.readouterr().out

        for param in disabled_parameters:
            assert param not in help_output, (
                f"DisableCLI parameter '{param}' should not appear in help text"
            )

    def test_disable_cli_parameters_not_recognized(
        self, disabled_parameters: list[str]
    ) -> None:
        """Test that certain parameters marked with DisableCLI() are not recognized."""
        for param in disabled_parameters:
            with contextlib.suppress(SystemExit):  # noqa: SIM117
                with pytest.raises(UnknownOptionError):
                    # Note: For now we just assume that "123" is a valid value for the parameter
                    app(["profile", param, "123"])
