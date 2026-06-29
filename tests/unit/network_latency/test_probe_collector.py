# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NetworkLatencyProbeCollector.probe_once error isolation + sampling.

The probe opens a fresh ``asyncio.open_connection`` per probe; these tests patch
that call so the handshake "succeeds" or "fails" deterministically without a real
socket. Every probe must produce exactly one NetworkLatencySample and never raise.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import NetworkLatencySample
from aiperf.network_latency.probe import NetworkLatencyProbeCollector


def _make_collector(record_callback) -> NetworkLatencyProbeCollector:
    return NetworkLatencyProbeCollector(
        target_url="http://localhost:8000/v1/chat/completions",
        target_host="localhost",
        target_port=8000,
        ping_interval=0.05,
        connect_timeout=1.0,
        record_callback=record_callback,
        collector_id="localhost:8000",
    )


def _mock_open_connection_success():
    """Return an AsyncMock standing in for a successful open_connection."""
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return AsyncMock(return_value=(reader, writer))


class TestProbeOnceSuccess:
    @pytest.mark.asyncio
    async def test_successful_handshake_records_positive_rtt(self) -> None:
        recorded: list[NetworkLatencySample] = []

        async def record_callback(samples, collector_id):
            recorded.extend(samples)

        collector = _make_collector(record_callback)

        with patch(
            "aiperf.network_latency.probe.asyncio.open_connection",
            _mock_open_connection_success(),
        ):
            await collector.probe_once()

        assert len(recorded) == 1
        sample = recorded[0]
        assert sample.success is True
        assert sample.rtt_ns is not None
        assert sample.rtt_ns > 0
        assert sample.error is None
        assert sample.target_host == "localhost"
        assert sample.target_port == 8000
        assert sample.probe_type == "tcp_connect"
        assert collector.successful_samples == 1

    @pytest.mark.asyncio
    async def test_successful_handshake_closes_writer(self) -> None:
        async def record_callback(samples, collector_id):
            pass

        collector = _make_collector(record_callback)
        open_conn = _mock_open_connection_success()
        _reader, writer = open_conn.return_value

        with patch("aiperf.network_latency.probe.asyncio.open_connection", open_conn):
            await collector.probe_once()

        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()


class TestProbeOnceFailure:
    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionRefusedError("refused"),
            asyncio.TimeoutError(),
            OSError("network unreachable"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_failed_handshake_records_failed_sample_without_raising(
        self, exc: BaseException
    ) -> None:
        recorded: list[NetworkLatencySample] = []

        async def record_callback(samples, collector_id):
            recorded.extend(samples)

        collector = _make_collector(record_callback)

        with patch(
            "aiperf.network_latency.probe.asyncio.open_connection",
            AsyncMock(side_effect=exc),
        ):
            await collector.probe_once()

        assert len(recorded) == 1
        sample = recorded[0]
        assert sample.success is False
        assert sample.rtt_ns is None
        assert sample.error is not None
        assert collector.successful_samples == 0

    @pytest.mark.asyncio
    async def test_timeout_wrapped_failure_records_failed_sample(self) -> None:
        """A handshake that outlives connect_timeout becomes a failed sample."""
        recorded: list[NetworkLatencySample] = []

        async def record_callback(samples, collector_id):
            recorded.extend(samples)

        collector = NetworkLatencyProbeCollector(
            target_url="http://localhost:8000/v1",
            target_host="localhost",
            target_port=8000,
            ping_interval=0.05,
            connect_timeout=0.01,
            record_callback=record_callback,
            collector_id="localhost:8000",
        )

        with patch(
            "aiperf.network_latency.probe.asyncio.wait_for",
            AsyncMock(side_effect=asyncio.TimeoutError()),
        ):
            await collector.probe_once()

        assert len(recorded) == 1
        assert recorded[0].success is False
        assert recorded[0].rtt_ns is None
        assert recorded[0].error is not None


class TestProbeOnceCallbackIsolation:
    @pytest.mark.asyncio
    async def test_record_callback_error_is_routed_to_error_callback(self) -> None:
        errors = []

        async def failing_record_callback(samples, collector_id):
            raise RuntimeError("callback boom")

        async def error_callback(error, collector_id):
            errors.append(error)

        collector = NetworkLatencyProbeCollector(
            target_url="http://localhost:8000/v1",
            target_host="localhost",
            target_port=8000,
            ping_interval=0.05,
            connect_timeout=1.0,
            record_callback=failing_record_callback,
            error_callback=error_callback,
            collector_id="localhost:8000",
        )

        with patch(
            "aiperf.network_latency.probe.asyncio.open_connection",
            _mock_open_connection_success(),
        ):
            await collector.probe_once()

        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_no_record_callback_does_not_raise(self) -> None:
        collector = NetworkLatencyProbeCollector(
            target_url="http://localhost:8000/v1",
            target_host="localhost",
            target_port=8000,
            ping_interval=0.05,
            connect_timeout=1.0,
            record_callback=None,
            collector_id="localhost:8000",
        )

        with patch(
            "aiperf.network_latency.probe.asyncio.open_connection",
            _mock_open_connection_success(),
        ):
            await collector.probe_once()

        assert collector.successful_samples == 1
