# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ConsoleGpuVendorDisclaimerExporter."""

import pytest
from rich.console import Console

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.console_gpu_vendor_disclaimer_exporter import (
    ConsoleGpuVendorDisclaimerExporter,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.exporters.conftest import make_exporter_config


def _cfg() -> CLIConfig:
    return CLIConfig(
        endpoint_type=EndpointType.CHAT, streaming=True, model_names=["test-model"]
    )


class TestConsoleGpuVendorDisclaimerExporter:
    @pytest.mark.asyncio
    async def test_renders_platform_warning_box(
        self, sample_telemetry_results, capsys
    ) -> None:
        exporter = ConsoleGpuVendorDisclaimerExporter(
            make_exporter_config(
                cli_config=_cfg(), telemetry_results=sample_telemetry_results
            )
        )
        await exporter.export(Console(width=120))
        output = capsys.readouterr().out

        assert "GPU Telemetry Platform" in output
        assert "nvidia" in output
        assert "semantics" in output and "validation" in output

    @pytest.mark.asyncio
    async def test_omits_when_no_telemetry_results(self, capsys) -> None:
        exporter = ConsoleGpuVendorDisclaimerExporter(
            make_exporter_config(cli_config=_cfg(), telemetry_results=None)
        )
        await exporter.export(Console(width=120))
        assert capsys.readouterr().out.strip() == ""
