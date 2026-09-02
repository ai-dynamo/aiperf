# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch/error-path coverage for NetworkLatencyProbeCollector.

Complements ``test_probe_collector.py`` (success/failure sampling) by covering
``resolve()`` (success + failure), the ``ping_interval`` property, the
``_probe_loop`` background tick, the writer-close OSError branch, and the nested
error-callback failure isolation in ``_send_sample``.
"""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.network_latency.probe import NetworkLatencyProbeCollector


def _make_collector(**overrides) -> NetworkLatencyProbeCollector:
    kwargs = dict(
        target_url="http://localhost:8000/v1/chat/completions",
        target_host="localhost",
        target_port=8000,
        ping_interval=0.05,
        connect_timeout=1.0,
        collector_id="localhost:8000",
    )
    kwargs.update(overrides)
    return NetworkLatencyProbeCollector(**kwargs)


class TestPingIntervalProperty:
    def test_ping_interval_reports_configured_value(self) -> None:
        collector = _make_collector(ping_interval=0.25)
        assert collector.ping_interval == 0.25


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_success_caches_resolved_address(self) -> None:
        collector = _make_collector()
        sockaddr = ("127.0.0.1", 8000)
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", sockaddr)]

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.getaddrinfo = AsyncMock(return_value=infos)
            mock_get_loop.return_value = mock_loop

            await collector.resolve()

        assert collector._resolved_host == "127.0.0.1"
        assert collector._resolved_family == socket.AF_INET

    @pytest.mark.asyncio
    async def test_resolve_empty_infos_keeps_original_host(self) -> None:
        collector = _make_collector()

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.getaddrinfo = AsyncMock(return_value=[])
            mock_get_loop.return_value = mock_loop

            await collector.resolve()

        # No infos returned -> falls back to the original host.
        assert collector._resolved_host == "localhost"

    @pytest.mark.asyncio
    async def test_resolve_oserror_is_non_fatal(self) -> None:
        collector = _make_collector()

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.getaddrinfo = AsyncMock(side_effect=OSError("dns failure"))
            mock_get_loop.return_value = mock_loop

            # Must not raise; host stays the original for per-connect resolution.
            await collector.resolve()

        assert collector._resolved_host == "localhost"


class TestProbeLoop:
    @pytest.mark.asyncio
    async def test_probe_loop_fires_probe_once(self) -> None:
        collector = _make_collector()
        collector.probe_once = MagicMock(return_value=AsyncMock()())
        collector.execute_async = MagicMock()

        await collector._probe_loop()

        collector.execute_async.assert_called_once()


class TestWriterCloseError:
    @pytest.mark.asyncio
    async def test_writer_close_oserror_still_records_success(self) -> None:
        """A failure closing the probe socket must not void a successful handshake."""
        recorded = []

        async def record_callback(samples, collector_id):
            recorded.extend(samples)

        collector = _make_collector(record_callback=record_callback)

        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock(side_effect=OSError("close failed"))
        writer.wait_closed = AsyncMock()

        with patch(
            "aiperf.network_latency.probe.asyncio.open_connection",
            AsyncMock(return_value=(reader, writer)),
        ):
            await collector.probe_once()

        assert len(recorded) == 1
        assert recorded[0].success is True
        assert recorded[0].rtt_ns is not None
        assert collector.successful_samples == 1


class TestSendSampleNestedCallbackFailure:
    @pytest.mark.asyncio
    async def test_record_and_error_callback_both_fail_does_not_raise(self) -> None:
        """When both record and error callbacks raise, errors are isolated."""

        async def failing_record_callback(samples, collector_id):
            raise RuntimeError("record boom")

        async def failing_error_callback(error, collector_id):
            raise RuntimeError("error boom")

        collector = _make_collector(
            record_callback=failing_record_callback,
            error_callback=failing_error_callback,
        )

        with patch(
            "aiperf.network_latency.probe.asyncio.open_connection",
            AsyncMock(return_value=(MagicMock(), MagicMock(wait_closed=AsyncMock()))),
        ):
            # Must not raise despite both callbacks failing.
            await collector.probe_once()
