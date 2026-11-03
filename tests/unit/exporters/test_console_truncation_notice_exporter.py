# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from io import StringIO

import pytest
from pytest import param
from rich.console import Console

from aiperf.common.models import ProfileResults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.console_truncation_notice_exporter import (
    ConsoleTruncationNoticeExporter,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import create_exporter_config


def _results(ended: int | None, skipped: int | None) -> ProfileResults:
    return ProfileResults(
        records=[],
        completed=10,
        start_ns=1,
        end_ns=2,
        conversations_ended_on_truncation=ended,
        turns_skipped_on_truncation=skipped,
    )


async def _export(results: ProfileResults) -> str:
    cfg = CLIConfig(model_names=["test-model"], endpoint_type=EndpointType.CHAT)
    output = StringIO()
    exporter = ConsoleTruncationNoticeExporter(create_exporter_config(results, cfg))
    await exporter.export(Console(file=output, width=200, legacy_windows=False))
    return output.getvalue()


@pytest.mark.asyncio
async def test_export_conversations_ended_prints_counts() -> None:
    output = await _export(_results(ended=3, skipped=5))
    assert "Conversations Ended on Truncation" in output
    assert "3 conversations ended early" in output
    assert "5 remaining turns" in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ended,skipped",
    [
        param(None, None, id="option-off"),
        param(0, 0, id="none-ended"),
    ],
)  # fmt: skip
async def test_export_nothing_ended_prints_nothing(
    ended: int | None, skipped: int | None
) -> None:
    assert await _export(_results(ended=ended, skipped=skipped)) == ""
