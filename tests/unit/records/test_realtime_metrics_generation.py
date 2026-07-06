# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``generate_realtime_metrics`` fan-out over accumulators.

A persistently failing accumulator must not crash the realtime tick, but it
must leave a trail: the failure is logged (warning) with the accumulator's
class name so a stale dashboard/log block is diagnosable.
"""

import logging

import pytest

from aiperf.common.models import MetricResult
from aiperf.records.records_manager_processing import generate_realtime_metrics


def _metric(tag: str) -> MetricResult:
    return MetricResult(tag=tag, header=tag, unit="ms", avg=1.0)


class HealthyAccumulator:
    """Accumulator whose summarize returns a plain list of MetricResult."""

    async def summarize(self) -> list[MetricResult]:
        return [_metric("time_to_first_token")]


class ExplodingAccumulator:
    """Accumulator whose summarize always raises."""

    async def summarize(self) -> list[MetricResult]:
        raise RuntimeError("summarize blew up")


@pytest.mark.asyncio
async def test_generate_realtime_metrics_failing_accumulator_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.WARNING, logger="aiperf.records.records_manager_processing"
    )

    flat = await generate_realtime_metrics(
        [HealthyAccumulator(), ExplodingAccumulator()]
    )

    assert [m.tag for m in flat] == ["time_to_first_token"]
    assert "ExplodingAccumulator" in caplog.text
    assert "summarize blew up" in caplog.text


@pytest.mark.asyncio
async def test_generate_realtime_metrics_all_healthy_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.WARNING, logger="aiperf.records.records_manager_processing"
    )

    flat = await generate_realtime_metrics([HealthyAccumulator()])

    assert [m.tag for m in flat] == ["time_to_first_token"]
    assert caplog.text == ""
