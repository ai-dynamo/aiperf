# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for bugs found during a docstring audit of `aiperf.common`."""

import asyncio
import logging
import time

import pytest
from pytest import param

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.mixins.task_manager_mixin import TaskManagerMixin
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.common.models.telemetry_models import (
    TelemetryHierarchy,
    TelemetryMetrics,
    TelemetryRecord,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        param("TRACE", logging.DEBUG - 5, id="trace"),
        param("DEBUG", logging.DEBUG, id="debug"),
        param("INFO", logging.INFO, id="info"),
        param("NOTICE", logging.WARNING - 5, id="notice"),
        param("WARNING", logging.WARNING, id="warning"),
        param("SUCCESS", logging.WARNING + 5, id="success"),
        param("ERROR", logging.ERROR, id="error"),
        param("CRITICAL", logging.CRITICAL, id="critical"),
        param("debug", logging.DEBUG, id="lowercase"),
        param("Notice", logging.WARNING - 5, id="mixed-case"),
    ],
)  # fmt: skip
def test_get_level_number_string_level_returns_numeric_level(
    name: str, expected: int
) -> None:
    assert AIPerfLogger.get_level_number(name) == expected


def test_get_level_number_int_level_returns_unchanged() -> None:
    assert AIPerfLogger.get_level_number(logging.INFO) == logging.INFO


def test_get_level_number_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        AIPerfLogger.get_level_number("NOT_A_LEVEL")


def _record(latency_ns: int | None) -> ServerMetricsRecord:
    return ServerMetricsRecord(
        endpoint_url="http://localhost:8081/metrics",
        timestamp_ns=1,
        endpoint_latency_ns=latency_ns,
        metrics={
            "foo": MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="d",
                samples=[MetricSample(value=1.0)],
            )
        },
    )


def test_to_slim_missing_endpoint_latency_preserves_none() -> None:
    slim = _record(None).to_slim()
    assert slim.endpoint_latency_ns is None


def test_to_slim_with_endpoint_latency_preserves_value() -> None:
    assert _record(1234).to_slim().endpoint_latency_ns == 1234


def test_add_record_preserves_pci_bus_id_and_device() -> None:
    hierarchy = TelemetryHierarchy()
    hierarchy.add_record(
        TelemetryRecord(
            gpu_index=0,
            gpu_uuid="GPU-abc",
            gpu_model_name="NVIDIA H100",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            timestamp_ns=1,
            telemetry_source_url="http://node1:9401/metrics",
            telemetry_data=TelemetryMetrics(),
        )
    )
    metadata = hierarchy.telemetry_source_endpoints["http://node1:9401/metrics"][
        "GPU-abc"
    ].metadata
    assert metadata.pci_bus_id == "00000000:02:00.0"
    assert metadata.device == "nvidia0"


class _TaskManager(TaskManagerMixin):
    pass


@pytest.mark.asyncio
async def test_cancel_all_tasks_awaits_task_cleanup() -> None:
    manager = _TaskManager()
    cleaned_up = asyncio.Event()

    async def _long_running() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cleaned_up.set()
            raise

    task = manager.execute_async(_long_running())
    await asyncio.sleep(0)

    await manager.cancel_all_tasks(timeout=5.0)

    assert cleaned_up.is_set()
    assert task.done()


@pytest.mark.asyncio
async def test_cancel_all_tasks_uncancellable_task_returns_within_timeout() -> None:
    manager = _TaskManager()

    async def _uncancellable() -> None:
        # to_thread cannot be interrupted by cancellation until the thread returns.
        await asyncio.to_thread(time.sleep, 0.5)

    manager.execute_async(_uncancellable())
    await asyncio.sleep(0)

    await asyncio.wait_for(manager.cancel_all_tasks(timeout=0.05), timeout=5.0)


def test_cancel_all_tasks_no_tasks_returns_immediately() -> None:
    manager = _TaskManager()
    asyncio.run(manager.cancel_all_tasks(timeout=0.01))
