# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.progress_client module."""

import gzip
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import zstandard as zstd
from pytest import param

from aiperf.common.enums import WorkerStartupState
from aiperf.operator.progress_client import (
    RETRYABLE_STATUS_CODES,
    JobProgress,
    ProgressClient,
)


class TestJobProgress:
    """Tests for JobProgress model."""

    def test_default_values(self) -> None:
        """Test JobProgress has expected defaults."""
        progress = JobProgress()
        assert progress.phases == {}
        assert progress.current_phase is None
        assert progress.error is None
        assert progress.connection_error is None

    def test_is_complete_no_profiling(self) -> None:
        """Test is_complete returns False when no profiling phase."""
        progress = JobProgress()
        assert progress.is_complete is False

    def test_is_complete_profiling_not_complete(self) -> None:
        """Test is_complete returns False when profiling not complete."""
        from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

        progress = JobProgress(
            phases={
                "profiling": CombinedPhaseStats(
                    phase="profiling",
                    requests_completed=50,
                    total_expected_requests=100,
                )
            }
        )
        assert progress.is_complete is False

    def test_is_complete_profiling_complete(self) -> None:
        """Test is_complete returns True when requests AND records are complete."""
        from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

        progress = JobProgress(
            phases={
                "profiling": CombinedPhaseStats(
                    phase="profiling",
                    requests_completed=100,
                    total_expected_requests=100,
                    requests_end_ns=1000000000,
                    records_end_ns=2000000000,  # Records also complete
                )
            }
        )
        assert progress.is_complete is True

    def test_is_not_complete_records_still_processing(self) -> None:
        """Test is_complete returns False when requests done but records still processing."""
        from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

        progress = JobProgress(
            phases={
                "profiling": CombinedPhaseStats(
                    phase="profiling",
                    requests_completed=100,
                    total_expected_requests=100,
                    requests_end_ns=1000000000,
                    records_end_ns=None,  # Records NOT complete
                )
            }
        )
        assert progress.is_complete is False

    def test_profiling_stats_property(self) -> None:
        """Test profiling_stats returns profiling phase stats."""
        from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

        stats = CombinedPhaseStats(
            phase="profiling",
            requests_completed=75,
            total_expected_requests=100,
        )
        progress = JobProgress(phases={"profiling": stats})
        assert progress.profiling_stats == stats

    def test_profiling_stats_none(self) -> None:
        """Test profiling_stats returns None when not available."""
        progress = JobProgress()
        assert progress.profiling_stats is None

    def test_warmup_stats_property(self) -> None:
        """Test warmup_stats returns warmup phase stats."""
        from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

        stats = CombinedPhaseStats(
            phase="warmup",
            requests_completed=10,
            total_expected_requests=10,
        )
        progress = JobProgress(phases={"warmup": stats})
        assert progress.warmup_stats == stats

    def test_warmup_stats_none(self) -> None:
        """Test warmup_stats returns None when not available."""
        progress = JobProgress()
        assert progress.warmup_stats is None


class TestProgressClientInit:
    """Tests for ProgressClient initialization."""

    def test_default_port(self) -> None:
        """Test ProgressClient uses default port from K8sEnvironment."""
        client = ProgressClient()
        # Default port should be API_SERVICE (9090)
        assert client._port == 9090

    def test_custom_port(self) -> None:
        """Test ProgressClient accepts custom port."""
        client = ProgressClient(port=8080)
        assert client._port == 8080

    def test_session_initially_none(self) -> None:
        """Test session is None before context manager."""
        client = ProgressClient()
        assert client._session is None


class TestProgressClientContextManager:
    """Tests for ProgressClient async context manager."""

    @pytest.mark.asyncio
    async def test_enter_creates_session(self) -> None:
        """Test entering context creates aiohttp session."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                assert client._session is not None

    @pytest.mark.asyncio
    async def test_exit_closes_session(self) -> None:
        """Test exiting context closes session."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                pass

            mock_session.close.assert_called_once()


class TestProgressClientGetProgress:
    """Tests for ProgressClient.get_progress method."""

    @pytest.mark.asyncio
    async def test_get_progress_not_in_context(self) -> None:
        """Test get_progress raises error when not in context."""
        client = ProgressClient()
        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await client.get_progress("host.example.com")

    @pytest.mark.asyncio
    async def test_get_progress_success(
        self, progress_api_response_running: dict[str, Any]
    ) -> None:
        """Test get_progress returns JobProgress on success."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=progress_api_response_running)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                progress = await client.get_progress("controller.default")

            assert progress.current_phase == "profiling"
            assert "profiling" in progress.phases
            assert "warmup" in progress.phases

    @pytest.mark.asyncio
    async def test_get_progress_connection_error_returns_empty_with_error(self) -> None:
        """Test get_progress returns empty JobProgress with connection_error on failure."""
        import aiohttp

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientError("Connection refused")
                    )
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                progress = await client.get_progress("unreachable.host")

            assert progress.phases == {}
            assert progress.current_phase is None
            assert progress.connection_error is not None
            assert "ClientError" in progress.connection_error

    @pytest.mark.asyncio
    async def test_get_progress_builds_correct_url(self) -> None:
        """Test get_progress builds correct URL."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={"phases": {}})

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                await client.get_progress("my-controller.ns.svc")

            # Verify the URL that was called
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            url = call_args[0][0]
            assert "http://my-controller.ns.svc:9090" in url
            assert "/api/progress" in url


class TestProgressClientGetWorkerStartupStates:
    """Tests for ProgressClient.get_worker_startup_states method."""

    @pytest.mark.asyncio
    async def test_get_worker_startup_states_success(self) -> None:
        """Test worker startup states are parsed from /api/workers."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(
                return_value={
                    "workers": {
                        "worker-1": {"worker_id": "worker-1", "startup_state": "ready"},
                        "worker-2": {
                            "worker_id": "worker-2",
                            "startup_state": "waiting_for_dataset",
                        },
                    }
                }
            )

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                states = await client.get_worker_startup_states("controller.default")

            assert states == {
                "worker-1": WorkerStartupState.READY,
                "worker-2": WorkerStartupState.WAITING_FOR_DATASET,
            }
            url = mock_session.get.call_args[0][0]
            assert url == "http://controller.default:9090/api/workers"


class TestProgressClientParseResponse:
    """Tests for ProgressClient._parse_progress_response method."""

    def test_parse_empty_response(self) -> None:
        """Test parsing empty response."""
        client = ProgressClient()
        progress = client._parse_progress_response({})

        assert progress.phases == {}
        assert progress.current_phase is None
        assert progress.error is None

    def test_parse_with_phases(
        self, progress_api_response_running: dict[str, Any]
    ) -> None:
        """Test parsing response with phase data."""
        client = ProgressClient()
        progress = client._parse_progress_response(progress_api_response_running)

        assert "profiling" in progress.phases
        assert "warmup" in progress.phases
        assert progress.phases["profiling"].requests_completed == 250

    def test_parse_with_worker_aggregate_status(
        self, progress_api_response_running: dict[str, Any]
    ) -> None:
        """Test parsing response with aggregate worker status."""
        client = ProgressClient()
        progress = client._parse_progress_response(
            {
                **progress_api_response_running,
                "workers": {
                    "ready": 4,
                    "total": 8,
                    "dispatchable": 3,
                    "router_connected": 6,
                    "ready_record_processors": 2,
                    "declared_record_processors": 4,
                    "ready_pods": 2,
                    "total_pods": 4,
                    "degraded_pods": 1,
                },
            }
        )

        assert progress.workers.dispatchable == 3
        assert progress.workers.total_pods == 4

    def test_parse_malformed_workers_payload_falls_back_to_default(
        self, progress_api_response_running: dict[str, Any]
    ) -> None:
        """Test malformed worker payloads degrade to the default aggregate status."""
        client = ProgressClient()
        progress = client._parse_progress_response(
            {
                **progress_api_response_running,
                "workers": {
                    "ready": "not-an-int",
                    "dispatchable": object(),
                },
            }
        )

        assert progress.workers.model_dump() == {
            "ready": 0,
            "total": 0,
            "dispatchable": 0,
            "router_connected": 0,
            "ready_record_processors": 0,
            "declared_record_processors": 0,
            "ready_pods": 0,
            "total_pods": 0,
            "degraded_pods": 0,
        }

    def test_parse_current_phase_computed_from_start_ns(
        self, progress_api_response_running: dict[str, Any]
    ) -> None:
        """Test current_phase is computed from phases with most recent start_ns."""
        client = ProgressClient()
        progress = client._parse_progress_response(progress_api_response_running)

        assert progress.current_phase == "profiling"

    def test_parse_with_error(
        self, progress_api_response_with_error: dict[str, Any]
    ) -> None:
        """Test parsing response with error."""
        client = ProgressClient()
        progress = client._parse_progress_response(progress_api_response_with_error)

        assert progress.error == "Connection refused to endpoint"

    def test_parse_invalid_phase_ignored(self) -> None:
        """Test parsing response with invalid phase name."""
        client = ProgressClient()
        progress = client._parse_progress_response(
            {
                "phases": {
                    "invalid_phase": {
                        "phase": "invalid_phase",
                        "requests_completed": 10,
                    },
                    "profiling": {
                        "phase": "profiling",
                        "requests_completed": 100,
                        "total_expected_requests": 1000,
                    },
                }
            }
        )

        # CreditPhase is str, so all phases are valid - "invalid_phase" is kept
        assert "invalid_phase" in progress.phases
        assert "profiling" in progress.phases

    def test_parse_empty_phases_current_phase_is_none(self) -> None:
        """Test current_phase is None when no phases exist."""
        client = ProgressClient()
        progress = client._parse_progress_response({"phases": {}})

        assert progress.current_phase is None


class TestProgressClientCheckHealth:
    """Tests for ProgressClient.check_health method."""

    @pytest.mark.asyncio
    async def test_check_health_not_in_context(self) -> None:
        """Test check_health raises error when not in context."""
        client = ProgressClient()
        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await client.check_health("host.example.com")

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        """Test check_health returns True on 200 response."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                is_healthy = await client.check_health("controller.host")

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self) -> None:
        """Test check_health returns False on error."""
        import aiohttp

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(side_effect=aiohttp.ClientError())
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                is_healthy = await client.check_health("unreachable.host")

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_health_uses_health_endpoint(self) -> None:
        """Test check_health uses /health endpoint (API service endpoint)."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                await client.check_health("my-controller")

            call_args = mock_session.get.call_args
            url = call_args[0][0]
            assert "/health" in url
            assert url.endswith("/health")  # Should be /health not /healthz


class TestProgressClientGetMetrics:
    """Tests for ProgressClient.get_metrics method."""

    @pytest.mark.asyncio
    async def test_get_metrics_not_in_context(self) -> None:
        """Test get_metrics raises error when not in context."""
        client = ProgressClient()
        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await client.get_metrics("host.example.com")

    @pytest.mark.asyncio
    async def test_get_metrics_success(self) -> None:
        """Test get_metrics returns metrics dict on success."""
        metrics_data = {
            "benchmark_id": "test-123",
            "metrics": [{"tag": "throughput", "value": 100.0}],
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=metrics_data)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                metrics = await client.get_metrics("controller.host")

            assert metrics == metrics_data

    @pytest.mark.asyncio
    async def test_get_metrics_failure_returns_none(self) -> None:
        """Test get_metrics returns None on error."""
        import aiohttp

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(side_effect=aiohttp.ClientError())
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                metrics = await client.get_metrics("unreachable.host")

            assert metrics is None

    @pytest.mark.asyncio
    async def test_get_metrics_uses_correct_endpoint(self) -> None:
        """Test get_metrics uses /api/metrics endpoint."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={})

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                await client.get_metrics("my-controller")

            call_args = mock_session.get.call_args
            url = call_args[0][0]
            assert "/api/metrics" in url


class TestProgressClientErrorMessages:
    """Tests for error message content in ProgressClient."""

    @pytest.mark.asyncio
    async def test_connection_error_includes_host_and_port(self) -> None:
        """Test connection_error message includes host and port for debugging."""
        import aiohttp

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientError("Connection refused")
                    )
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                progress = await client.get_progress(
                    "my-controller.ns.svc.cluster.local"
                )

            # Error message should include host and port for debugging
            assert progress.connection_error is not None
            assert "my-controller.ns.svc.cluster.local" in progress.connection_error
            assert "9090" in progress.connection_error
            # Should include helpful hint
            assert "Check if controller pod is running" in progress.connection_error

    @pytest.mark.asyncio
    async def test_connection_error_includes_error_type(self) -> None:
        """Test connection_error message includes the exception type."""
        import aiohttp

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientConnectorError(
                            MagicMock(), OSError("Connection refused")
                        )
                    )
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                progress = await client.get_progress("test-host")

            assert progress.connection_error is not None
            # Should indicate the type of error
            assert (
                "ClientConnectorError" in progress.connection_error
                or "connecting to" in progress.connection_error
            )

    @pytest.mark.asyncio
    async def test_retries_exhausted_error_includes_url(self) -> None:
        """Test error message when retries exhausted includes the URL."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Return non-success status code repeatedly
            mock_response = AsyncMock()
            mock_response.status = 503

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            # Use 0 retries to speed up test
            client = ProgressClient(port=9090, max_retries=0)
            async with client:
                progress = await client.get_progress("test-host")

            # Should get meaningful error about retries
            assert progress.connection_error is not None
            # Should include URL or retry info
            assert (
                "retry" in progress.connection_error.lower()
                or "test-host" in progress.connection_error
            )


class TestRequestWithRetry:
    """Tests for ProgressClient._request_with_retry method."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [
            param(408, id="request_timeout"),
            param(429, id="too_many_requests"),
            param(500, id="internal_server_error"),
            param(502, id="bad_gateway"),
            param(503, id="service_unavailable"),
            param(504, id="gateway_timeout"),
        ],
    )  # fmt: skip
    async def test_retryable_status_codes_trigger_retry(self, status_code: int) -> None:
        """Test that retryable status codes trigger retry attempts."""
        assert status_code in RETRYABLE_STATUS_CODES

        call_count = 0

        def create_response(url: str):
            nonlocal call_count
            call_count += 1
            mock_response = AsyncMock()
            mock_response.status = status_code
            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=2, initial_backoff=0.001)
            async with client:
                result = await client._request_with_retry("http://example.com/api")

            # Should have made 3 attempts (1 initial + 2 retries)
            assert call_count == 3
            # Should return None after exhausting retries
            assert result is None

    @pytest.mark.asyncio
    async def test_exponential_backoff_increases_delay(self) -> None:
        """Test that backoff increases exponentially between retries."""
        sleep_times: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_times.append(delay)

        with (
            patch("aiohttp.ClientSession") as mock_session_class,
            patch("asyncio.sleep", side_effect=track_sleep),
        ):
            mock_response = AsyncMock()
            mock_response.status = 503

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=3, initial_backoff=0.5)
            async with client:
                await client._request_with_retry("http://example.com/api")

            # Should have 3 sleep calls (before retries 2, 3, 4)
            assert len(sleep_times) == 3
            # Backoff has jitter (0.8-1.2x), so check approximate ranges
            assert 0.4 <= sleep_times[0] <= 0.6  # ~0.5
            # Each subsequent sleep is roughly 2x the previous (with jitter)
            assert sleep_times[1] > sleep_times[0]
            assert sleep_times[2] > sleep_times[1]

    @pytest.mark.asyncio
    async def test_connection_error_triggers_retry(self) -> None:
        """Test that connection errors trigger retry attempts."""
        call_count = 0

        def create_response(url: str):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # First two attempts fail
                return AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientError("Connection failed")
                    )
                )
            # Third attempt succeeds
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"success": True})
            mock_response.raise_for_status = MagicMock()
            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=3, initial_backoff=0.001)
            async with client:
                result = await client._request_with_retry("http://example.com/api")

            assert call_count == 3
            assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_connection_error_exhausts_retries_raises(self) -> None:
        """Test that connection errors raise after exhausting retries."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientError("Connection failed")
                    )
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=2, initial_backoff=0.001)
            async with client:
                with pytest.raises(aiohttp.ClientError, match="Connection failed"):
                    await client._request_with_retry("http://example.com/api")

    @pytest.mark.asyncio
    async def test_success_on_first_attempt_no_retry(self) -> None:
        """Test that successful first attempt does not trigger retries."""
        call_count = 0

        def create_response(url: str):
            nonlocal call_count
            call_count += 1
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "value"})
            mock_response.raise_for_status = MagicMock()
            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=3, initial_backoff=0.001)
            async with client:
                result = await client._request_with_retry("http://example.com/api")

            assert call_count == 1
            assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises(self) -> None:
        """Test that non-retryable status codes raise immediately."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=404,
                )
            )

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=3, initial_backoff=0.001)
            async with client:
                with pytest.raises(aiohttp.ClientResponseError):
                    await client._request_with_retry("http://example.com/api")


class TestGetServerMetrics:
    """Tests for ProgressClient.get_server_metrics method."""

    @pytest.mark.asyncio
    async def test_get_server_metrics_success(self) -> None:
        """Test get_server_metrics returns metrics dict on success."""
        server_metrics_data = {
            "server_name": "vllm",
            "metrics": {"queue_depth": 10, "running_requests": 5},
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=server_metrics_data)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                metrics = await client.get_server_metrics("controller.host")

            assert metrics == server_metrics_data

    @pytest.mark.asyncio
    async def test_get_server_metrics_failure_returns_none(self) -> None:
        """Test get_server_metrics returns None on error."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(side_effect=aiohttp.ClientError())
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                metrics = await client.get_server_metrics("unreachable.host")

            assert metrics is None

    @pytest.mark.asyncio
    async def test_get_server_metrics_uses_correct_endpoint(self) -> None:
        """Test get_server_metrics uses /api/server-metrics endpoint."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={})

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                await client.get_server_metrics("my-controller")

            call_args = mock_session.get.call_args
            url = call_args[0][0]
            assert "/api/server-metrics" in url

    @pytest.mark.asyncio
    async def test_get_server_metrics_returns_none_when_request_returns_none(
        self,
    ) -> None:
        """Test get_server_metrics returns None when _request_with_retry returns None."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Simulate retryable failure that exhausts retries
            mock_response = AsyncMock()
            mock_response.status = 503

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=0, initial_backoff=0.001)
            async with client:
                metrics = await client.get_server_metrics("controller.host")

            assert metrics is None


class TestGetResultsList:
    """Tests for ProgressClient.get_results_list method."""

    @pytest.mark.asyncio
    async def test_get_results_list_success(self) -> None:
        """Test get_results_list returns file list on success."""
        files_data = {
            "files": [
                {"name": "metrics.json", "size": 1024},
                {"name": "records.jsonl", "size": 2048},
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=files_data)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                result = await client.get_results_list("controller.host")

            assert result == files_data["files"]
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_results_list_empty_files(self) -> None:
        """Test get_results_list returns empty list when no files."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={"files": []})

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                result = await client.get_results_list("controller.host")

            assert result == []

    @pytest.mark.asyncio
    async def test_get_results_list_failure_returns_none(self) -> None:
        """Test get_results_list returns None on error."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(side_effect=aiohttp.ClientError())
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                result = await client.get_results_list("unreachable.host")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_results_list_uses_correct_endpoint(self) -> None:
        """Test get_results_list uses /api/results/list endpoint."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={"files": []})

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(port=9090)
            async with client:
                await client.get_results_list("my-controller")

            call_args = mock_session.get.call_args
            url = call_args[0][0]
            assert "/api/results/list" in url

    @pytest.mark.asyncio
    async def test_get_results_list_returns_none_when_request_returns_none(
        self,
    ) -> None:
        """Test get_results_list returns None when _request_with_retry returns None."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 503

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=0, initial_backoff=0.001)
            async with client:
                result = await client.get_results_list("controller.host")

            assert result is None


class TestDownloadResultFile:
    """Tests for ProgressClient.download_result_file method."""

    @pytest.fixture(autouse=True)
    def _disable_compress_on_disk(self) -> None:
        """Disable on-disk compression so these tests verify the original raw-file behavior."""
        from aiperf.operator.environment import OperatorEnvironment

        original = OperatorEnvironment.RESULTS.COMPRESS_ON_DISK
        OperatorEnvironment.RESULTS.COMPRESS_ON_DISK = False
        yield
        OperatorEnvironment.RESULTS.COMPRESS_ON_DISK = original

    @pytest.mark.asyncio
    async def test_download_result_file_not_in_context(self) -> None:
        """Test download_result_file raises error when not in context."""
        client = ProgressClient()
        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await client.download_result_file(
                "host.example.com",
                "metrics.json",
                Path("/tmp/metrics.json"),
            )

    @pytest.mark.asyncio
    async def test_download_result_file_404_returns_false(self, tmp_path: Path) -> None:
        """Test download_result_file returns False on 404 response."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "missing.json",
                    tmp_path / "missing.json",
                )

            assert result is False

    @pytest.mark.asyncio
    async def test_download_result_file_raw_content(self, tmp_path: Path) -> None:
        """Test download_result_file downloads raw content correctly."""
        test_content = b'{"metric": "value"}'

        async def iter_chunked(chunk_size: int):
            yield test_content

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_path = tmp_path / "metrics.json"
            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    dest_path,
                )

            assert result is True
            assert dest_path.exists()
            assert dest_path.read_bytes() == test_content

    @pytest.mark.asyncio
    async def test_download_result_file_gzip_decompression(
        self, tmp_path: Path
    ) -> None:
        """Test download_result_file decompresses gzip content correctly."""
        original_content = b'{"metric": "gzipped_value"}'
        compressed_content = gzip.compress(original_content)

        async def iter_chunked(chunk_size: int):
            yield compressed_content

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "gzip"}
            mock_response.content.iter_chunked = iter_chunked

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_path = tmp_path / "metrics.json"
            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    dest_path,
                )

            assert result is True
            assert dest_path.exists()
            assert dest_path.read_bytes() == original_content

    @pytest.mark.asyncio
    async def test_download_result_file_zstd_decompression(
        self, tmp_path: Path
    ) -> None:
        """Test download_result_file decompresses zstd content correctly."""
        original_content = b'{"metric": "zstd_value"}'
        cctx = zstd.ZstdCompressor()
        compressed_content = cctx.compress(original_content)

        async def iter_chunked(chunk_size: int):
            yield compressed_content

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "zstd"}
            mock_response.content.iter_chunked = iter_chunked

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_path = tmp_path / "metrics.json"
            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    dest_path,
                )

            assert result is True
            assert dest_path.exists()
            assert dest_path.read_bytes() == original_content

    @pytest.mark.asyncio
    async def test_download_result_file_x_filename_header(self, tmp_path: Path) -> None:
        """Test download_result_file uses X-Filename header to determine filename."""
        test_content = b'{"metric": "value"}'

        async def iter_chunked(chunk_size: int):
            yield test_content

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {
                "Content-Encoding": "identity",
                "X-Filename": "custom_filename.json",
            }
            mock_response.content.iter_chunked = iter_chunked

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_path = tmp_path / "original.json"
            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    dest_path,
                )

            assert result is True
            # Should use X-Filename header value
            actual_path = tmp_path / "custom_filename.json"
            assert actual_path.exists()
            assert actual_path.read_bytes() == test_content

    @pytest.mark.asyncio
    async def test_download_result_file_client_error_returns_false(
        self, tmp_path: Path
    ) -> None:
        """Test download_result_file returns False on ClientError."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientError("Network error")
                    )
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    tmp_path / "metrics.json",
                )

            assert result is False

    @pytest.mark.asyncio
    async def test_download_result_file_creates_parent_directory(
        self, tmp_path: Path
    ) -> None:
        """Test download_result_file creates parent directories if needed."""
        test_content = b'{"metric": "value"}'

        async def iter_chunked(chunk_size: int):
            yield test_content

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            # Path with non-existent parent directories
            dest_path = tmp_path / "nested" / "dir" / "metrics.json"
            assert not dest_path.parent.exists()

            client = ProgressClient()
            async with client:
                result = await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    dest_path,
                )

            assert result is True
            assert dest_path.exists()

    @pytest.mark.asyncio
    async def test_download_result_file_sets_accept_encoding_header(
        self, tmp_path: Path
    ) -> None:
        """Test download_result_file sends correct Accept-Encoding header."""
        test_content = b'{"metric": "value"}'

        async def iter_chunked(chunk_size: int):
            yield test_content

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked

            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient()
            async with client:
                await client.download_result_file(
                    "controller.host",
                    "metrics.json",
                    tmp_path / "metrics.json",
                )

            # Check headers were set
            call_kwargs = mock_session.get.call_args[1]
            assert "headers" in call_kwargs
            assert "Accept-Encoding" in call_kwargs["headers"]
            assert "zstd" in call_kwargs["headers"]["Accept-Encoding"]
            assert "gzip" in call_kwargs["headers"]["Accept-Encoding"]


class TestDownloadAllResults:
    """Tests for ProgressClient.download_all_results method."""

    @pytest.fixture(autouse=True)
    def _disable_compress_on_disk(self) -> None:
        """Disable on-disk compression so these tests verify the original raw-file behavior."""
        from aiperf.operator.environment import OperatorEnvironment

        original = OperatorEnvironment.RESULTS.COMPRESS_ON_DISK
        OperatorEnvironment.RESULTS.COMPRESS_ON_DISK = False
        yield
        OperatorEnvironment.RESULTS.COMPRESS_ON_DISK = original

    @pytest.mark.asyncio
    async def test_download_all_results_success(self, tmp_path: Path) -> None:
        """Test download_all_results discovers and downloads all available files."""
        downloaded_filenames: list[str] = []
        available_files = [
            {"name": "metrics.json", "size": 100},
            {"name": "records.jsonl", "size": 200},
        ]

        async def iter_chunked(chunk_size: int):
            yield b'{"data": "test"}'

        def create_response(url: str, **kwargs):
            if "/api/results/list" in url:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.raise_for_status = MagicMock()
                mock_response.json = AsyncMock(return_value={"files": available_files})
                return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

            filename = url.split("/api/results/files/")[-1]
            downloaded_filenames.append(filename)

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked

            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_dir = tmp_path / "results"
            client = ProgressClient()
            async with client:
                downloaded = await client.download_all_results(
                    "controller.host", dest_dir
                )

            assert len(downloaded) == 2
            assert dest_dir.exists()
            assert "metrics.json" in downloaded_filenames
            assert "records.jsonl" in downloaded_filenames

    @pytest.mark.asyncio
    async def test_download_all_results_returns_empty_when_list_fails(
        self, tmp_path: Path
    ) -> None:
        """Test download_all_results returns empty list when list endpoint fails."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                side_effect=aiohttp.ClientError("Connection refused")
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            client = ProgressClient(max_retries=0, initial_backoff=0.001)
            async with client:
                downloaded = await client.download_all_results(
                    "controller.host", tmp_path / "results"
                )

            assert downloaded == []

    @pytest.mark.asyncio
    async def test_download_all_results_returns_only_successful(
        self, tmp_path: Path
    ) -> None:
        """Test download_all_results only returns successfully downloaded files."""
        available_files = [
            {"name": "metrics.json", "size": 100},
            {"name": "records.jsonl", "size": 200},
            {"name": "raw.jsonl", "size": 300},
        ]

        async def iter_chunked(chunk_size: int):
            yield b'{"data": "test"}'

        def create_response(url: str, **kwargs):
            if "/api/results/list" in url:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.raise_for_status = MagicMock()
                mock_response.json = AsyncMock(return_value={"files": available_files})
                return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

            if "records.jsonl" in url or "raw.jsonl" in url:
                mock_response = AsyncMock()
                mock_response.status = 404
                return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked

            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_dir = tmp_path / "results"
            client = ProgressClient()
            async with client:
                downloaded = await client.download_all_results(
                    "controller.host", dest_dir
                )

            assert "records.jsonl" not in downloaded
            assert "raw.jsonl" not in downloaded
            assert "metrics.json" in downloaded

    @pytest.mark.asyncio
    async def test_download_all_results_respects_max_concurrent(
        self, tmp_path: Path
    ) -> None:
        """Test download_all_results limits concurrency via semaphore."""

        available_files = [{"name": f"file{i}.json", "size": 100} for i in range(10)]

        async def iter_chunked(chunk_size: int):
            yield b'{"data": "test"}'

        def create_response(url: str, **kwargs):
            if "/api/results/list" in url:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.raise_for_status = MagicMock()
                mock_response.json = AsyncMock(return_value={"files": available_files})
                return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked

            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_dir = tmp_path / "results"
            client = ProgressClient()
            async with client:
                downloaded = await client.download_all_results(
                    "controller.host", dest_dir, max_concurrent=3
                )

            assert len(downloaded) == 10

    @pytest.mark.asyncio
    async def test_download_all_results_default_max_concurrent(
        self, tmp_path: Path
    ) -> None:
        """Test download_all_results uses default max_concurrent=5."""
        import inspect

        sig = inspect.signature(ProgressClient.download_all_results)
        assert sig.parameters["max_concurrent"].default == 5

    @pytest.mark.asyncio
    async def test_download_all_results_handles_mixed_exceptions(
        self, tmp_path: Path
    ) -> None:
        """Test download_all_results filters out exceptions from gather results."""
        available_files = [
            {"name": "good.json", "size": 100},
            {"name": "bad.json", "size": 200},
        ]

        async def iter_chunked(chunk_size: int):
            yield b'{"data": "test"}'

        def create_response(url: str, **kwargs):
            if "/api/results/list" in url:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.raise_for_status = MagicMock()
                mock_response.json = AsyncMock(return_value={"files": available_files})
                return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

            if "bad.json" in url:
                return AsyncMock(
                    __aenter__=AsyncMock(
                        side_effect=aiohttp.ClientError("Download failed")
                    )
                )

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Encoding": "identity"}
            mock_response.content.iter_chunked = iter_chunked
            return AsyncMock(__aenter__=AsyncMock(return_value=mock_response))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session

            dest_dir = tmp_path / "results"
            client = ProgressClient()
            async with client:
                downloaded = await client.download_all_results(
                    "controller.host", dest_dir
                )

            # Only the successful download should be in the list
            assert "good.json" in downloaded
            assert "bad.json" not in downloaded
