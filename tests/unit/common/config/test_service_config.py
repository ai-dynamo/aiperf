# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 CLIConfig tests.

The v1 CLIConfig is now a CLI input DTO with no validators or derived
properties (see `aiperf.config.flags.cli_config` module docstring). All the
behavior the older test cases asserted - `_comm_config` build-up, `comm_config`
property, `api_enabled` derivation, TTY-aware ui_type defaulting, the both-
configs error - now lives on the v2 BenchmarkRun pipeline (build_comm_config in
`aiperf.config.comm.build`, CLIConfig in `aiperf.config.runtime`, and
the AIPerfConfig validator). Those tests have been deleted alongside this
note rather than ported, because the behavior under test no longer attaches
to the class under test.
"""

import pytest

from aiperf.config.flags.cli_config import CLIConfig


class TestCLIConfigAPIFields:
    """Test api_port, api_host fields on the v1 input DTO."""

    def test_api_port_default_none(self) -> None:
        config = CLIConfig()
        assert config.api_port is None

    def test_api_host_default_none(self) -> None:
        config = CLIConfig()
        assert config.api_host is None

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_api_port_rejects_invalid_values(self, port: int) -> None:
        with pytest.raises(ValueError):
            CLIConfig(api_port=port)

    @pytest.mark.parametrize("port", [1, 8080, 65535])
    def test_api_port_accepts_valid_values(self, port: int) -> None:
        config = CLIConfig(api_port=port)
        assert config.api_port == port
