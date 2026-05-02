# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.exceptions import IncompatibleMetricsEndpointError
from aiperf.common.mixins.base_metrics_collector_mixin import (
    BaseMetricsCollectorMixin,
)
from aiperf.transports.http_defaults import AioHttpDefaults


class ConcreteCollector(BaseMetricsCollectorMixin[dict]):
    """Minimal concrete subclass for testing the abstract mixin."""

    async def _collect_and_process_metrics(self) -> None:
        pass


class TestTrustEnvPassedToSessions:
    """Test that trust_env is consistently passed to all aiohttp.ClientSession constructors."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trust_env_value", [True, False])
    async def test_trust_env_passed_to_all_sessions(
        self,
        trust_env_value: bool,
        monkeypatch,
    ) -> None:
        """Test that TRUST_ENV is passed to both the persistent and temporary sessions."""
        monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", trust_env_value)

        collector = ConcreteCollector(
            endpoint_url="http://localhost:9400/metrics",
            collection_interval=1.0,
            reachability_timeout=5.0,
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            # First call: _initialize_http_client creates the persistent session
            mock_persistent = MagicMock()
            mock_persistent.close = AsyncMock()

            # Second call: is_url_reachable creates a temporary session (as context manager)
            mock_response = MagicMock(status=200)
            mock_response_cm = MagicMock()
            mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_cm.__aexit__ = AsyncMock(return_value=None)
            mock_temp = MagicMock()
            mock_temp.head = MagicMock(return_value=mock_response_cm)
            mock_temp_cm = MagicMock()
            mock_temp_cm.__aenter__ = AsyncMock(return_value=mock_temp)
            mock_temp_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.side_effect = [mock_persistent, mock_temp_cm]

            with patch(
                "aiperf.common.mixins.base_metrics_collector_mixin.create_tcp_connector"
            ) as mock_create:
                mock_conn = AsyncMock()
                mock_conn.close = AsyncMock()
                mock_create.return_value = mock_conn

                await collector._initialize_http_client()
                # Reset _session so is_url_reachable takes the temporary session path
                collector._session = None
                await collector.is_url_reachable()

            assert mock_session_class.call_count == 2
            for call in mock_session_class.call_args_list:
                assert call[1]["trust_env"] == trust_env_value


class _RaisingCollector(BaseMetricsCollectorMixin[dict]):
    """Concrete subclass whose `_collect_and_process_metrics` always raises
    IncompatibleMetricsEndpointError, simulating the TRT-LLM JSON `/metrics`
    bug as it would surface from either fetch-side content-type rejection
    or parser-side reclassification."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: int = 0

    async def _collect_and_process_metrics(self) -> None:
        self.calls += 1
        raise IncompatibleMetricsEndpointError(
            f"endpoint {self._endpoint_url!r} returned non-Prometheus content"
        )


class TestAutoDisableOnIncompatibleEndpoint:
    """The collector should permanently disable itself after one
    IncompatibleMetricsEndpointError instead of re-failing every scrape
    interval (the failure mode that turned a 30min benchmark into 8hrs)."""

    @pytest.mark.asyncio
    async def test_first_failure_disables_collector_and_invokes_callback_once(
        self,
    ) -> None:
        error_cb = AsyncMock()
        collector = _RaisingCollector(
            endpoint_url="http://localhost:9999/metrics",
            collection_interval=0.1,
            reachability_timeout=1.0,
            error_callback=error_cb,
        )

        await collector.collect_and_process_metrics()

        assert collector._endpoint_disabled is True
        assert collector.calls == 1
        assert error_cb.await_count == 1
        # The callback receives ErrorDetails describing the underlying
        # IncompatibleMetricsEndpointError, not the bare exception.
        (error_details, collector_id), _ = error_cb.await_args
        assert collector_id == collector.id
        assert "Incompatible" in error_details.type or "Incompatible" in str(
            error_details
        )

    @pytest.mark.asyncio
    async def test_subsequent_calls_short_circuit_after_disable(self) -> None:
        error_cb = AsyncMock()
        collector = _RaisingCollector(
            endpoint_url="http://localhost:9999/metrics",
            collection_interval=0.1,
            reachability_timeout=1.0,
            error_callback=error_cb,
        )

        # Three successive scrape cycles
        await collector.collect_and_process_metrics()
        await collector.collect_and_process_metrics()
        await collector.collect_and_process_metrics()

        # The underlying _collect_and_process_metrics ran exactly once;
        # subsequent cycles short-circuited at the disabled gate, so no
        # additional error callbacks fired (no parse-error spam).
        assert collector.calls == 1
        assert error_cb.await_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_calls_log_disable_warning_only_once(self) -> None:
        """The real-world trigger: `_collect_metrics_loop` calls
        ``execute_async(self.collect_and_process_metrics())`` every interval
        without awaiting prior cycles, so multiple scrape coroutines can
        be in flight when the first one raises. Every concurrent call
        will reach the except block, but the warning + error_callback
        must fire exactly once across the cohort.
        """
        error_cb = AsyncMock()
        collector = _RaisingCollector(
            endpoint_url="http://localhost:9999/metrics",
            collection_interval=0.1,
            reachability_timeout=1.0,
            error_callback=error_cb,
        )

        await asyncio.gather(
            *(collector.collect_and_process_metrics() for _ in range(8))
        )

        assert collector._endpoint_disabled is True
        # Concurrent cohort all raised, but only the first arrival into
        # the except block flipped the flag; subsequent arrivals saw the
        # disabled check and short-circuited before logging.
        assert error_cb.await_count == 1


class TestFetchRejectsJsonContentType:
    """Sanity check on the fetch-side guard: a 200 response with
    `application/json` content-type must raise
    IncompatibleMetricsEndpointError before the body is read, even though
    the status is OK. This is what differentiates the TRT-LLM /metrics bug
    from a transient network issue."""

    @pytest.mark.asyncio
    async def test_application_json_response_raises_incompatible(self) -> None:
        collector = ConcreteCollector(
            endpoint_url="http://localhost:9999/metrics",
            collection_interval=0.1,
            reachability_timeout=1.0,
        )

        # Build a mock aiohttp response context manager whose `headers` mimics
        # TRT-LLM's `/metrics` (Content-Type: application/json, body `[]`).
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = AsyncMock(return_value="[]")
        response_cm = MagicMock()
        response_cm.__aenter__ = AsyncMock(return_value=mock_response)
        response_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=response_cm)
        collector._session = mock_session

        with pytest.raises(IncompatibleMetricsEndpointError):
            await collector._fetch_metrics_text()

        # The body should never have been read — Content-Type rejection is
        # cheaper than full parse and avoids any work on a bad endpoint.
        mock_response.text.assert_not_awaited()
