# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.operator.metrics."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from aiperf.operator.metrics import (
    HANDLER_DURATION,
    HANDLER_TOTAL,
    track_handler,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    """Reset registry-wide samples between tests (not strictly required but isolates assertions)."""
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()
    yield


@pytest.mark.asyncio
async def test_track_handler_increments_success_counter() -> None:
    @track_handler("test_handler")
    async def fake_handler() -> None:
        return None

    await fake_handler()

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_total"
        and s.labels.get("handler") == "test_handler"
        and s.labels.get("outcome") == "success"
    ]
    assert any(s.value >= 1 for s in samples)


@pytest.mark.asyncio
async def test_track_handler_increments_error_counter_on_exception() -> None:
    @track_handler("flaky_handler")
    async def fake_handler() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await fake_handler()

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_total"
        and s.labels.get("handler") == "flaky_handler"
        and s.labels.get("outcome") == "error"
    ]
    assert any(s.value >= 1 for s in samples)


@pytest.mark.asyncio
async def test_track_handler_records_duration_histogram() -> None:
    @track_handler("timed_handler")
    async def fake_handler() -> None:
        return None

    await fake_handler()

    samples = [
        s
        for m in REGISTRY.collect()
        for s in m.samples
        if s.name == "aiperf_operator_handler_duration_seconds_count"
        and s.labels.get("handler") == "timed_handler"
    ]
    assert any(s.value >= 1 for s in samples)
