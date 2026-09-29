# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.environment import Environment
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


class TestReadTimeoutSanityWarning:
    """The collector should warn at init when the socket read timeout is far
    above the collection cadence, but stay quiet for well-proportioned pairs
    (including the shipped defaults)."""

    @pytest.mark.asyncio
    async def test_warns_for_badly_skewed_pair(self, monkeypatch) -> None:
        monkeypatch.setattr(Environment.HTTP, "METRICS_SCRAPE_READ_TIMEOUT", 60.0)
        collector = ConcreteCollector(
            endpoint_url="http://localhost:9400/metrics",
            collection_interval=0.05,  # 60 / 0.05 = 1200x
            reachability_timeout=5.0,
        )
        try:
            with patch.object(collector, "warning") as mock_warning:
                await collector._initialize_http_client()

            assert mock_warning.call_count == 1
        finally:
            await collector._session.close()

    @pytest.mark.asyncio
    async def test_does_not_warn_for_default_read_timeout_and_interval(
        self,
    ) -> None:
        collector = ConcreteCollector(
            endpoint_url="http://localhost:9400/metrics",
            collection_interval=Environment.SERVER_METRICS.COLLECTION_INTERVAL,
            reachability_timeout=5.0,
        )
        try:
            with patch.object(collector, "warning") as mock_warning:
                await collector._initialize_http_client()

            mock_warning.assert_not_called()
        finally:
            await collector._session.close()

    @pytest.mark.asyncio
    async def test_does_not_warn_for_sane_custom_pair(self) -> None:
        collector = ConcreteCollector(
            endpoint_url="http://localhost:9400/metrics",
            collection_interval=1.0,
            reachability_timeout=5.0,
        )
        try:
            with patch.object(collector, "warning") as mock_warning:
                await collector._initialize_http_client()

            mock_warning.assert_not_called()
        finally:
            await collector._session.close()


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


class MalformedHeadServer:
    """A real HTTP server whose HEAD reply illegally carries a body.

    RFC 9110 forbids content on a HEAD response. Triton's `/metrics` frontend
    violates this: every non-GET method hits `RETURN_AND_RESPOND_WITH_ERR`,
    which writes `{"error":"Method Not Allowed"}` into the output buffer and
    sends it. Those stray bytes break the client's parse -- aiohttp >= 3.14
    reports them as a bad status line. Because the connection is keep-alive the
    damage is timing-dependent: the HEAD itself fails when the body shares the
    headers' TCP segment, otherwise the next request to reuse the connection
    fails instead. Both must leave a GET-serving endpoint reachable.

    Served over a raw socket because no compliant HTTP framework will emit a
    body on a HEAD reply, which is precisely the behavior under test.

    Args:
        split_write: send the stray body in a separate write from the headers,
            making the poison surface on the reused connection rather than on
            the HEAD itself.
        ignore_connection_close: keep the connection alive even when the client
            asked to close it, modelling a maximally uncooperative server.
    """

    GET_BODY = b"# HELP up Server is up\n# TYPE up gauge\nup 1\n"
    HEAD_ERROR_BODY = b'{"error":"Method Not Allowed"}'

    def __init__(
        self,
        *,
        split_write: bool = False,
        ignore_connection_close: bool = False,
    ) -> None:
        self.split_write = split_write
        self.ignore_connection_close = ignore_connection_close
        self.methods_seen: list[str] = []
        self._server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> "MalformedHeadServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/metrics"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    return

                wants_close = False
                while True:
                    header = await reader.readline()
                    if header in (b"\r\n", b"\n", b""):
                        break
                    if header.lower().startswith(b"connection:"):
                        wants_close = b"close" in header.lower()

                method = request_line.split(b" ")[0].decode()
                self.methods_seen.append(method)

                if method == "HEAD":
                    await self._reply_to_head(writer)
                else:
                    await self._reply_to_get(writer)

                if wants_close and not self.ignore_connection_close:
                    return
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def _reply_to_head(self, writer: asyncio.StreamWriter) -> None:
        """Reply 405 and -- illegally -- include the error body."""
        body = self.HEAD_ERROR_BODY
        headers = (
            b"HTTP/1.1 405 Method Not Allowed\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        )
        if self.split_write:
            writer.write(headers)
            await writer.drain()
            writer.write(body)
        else:
            writer.write(headers + body)
        await writer.drain()

    async def _reply_to_get(self, writer: asyncio.StreamWriter) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: "
            + str(len(self.GET_BODY)).encode()
            + b"\r\n\r\n"
            + self.GET_BODY
        )
        await writer.drain()


class TestReachabilityHeadFallback:
    """Reachability must fall back to GET whenever the HEAD probe is unusable.

    These drive the real aiohttp client against a real socket server. The
    pre-existing reachability tests mock `_check_reachability_with_session`
    outright, so they cannot catch a regression in the fallback itself.
    """

    def _collector(self, url: str) -> ConcreteCollector:
        return ConcreteCollector(
            endpoint_url=url,
            collection_interval=1.0,
            reachability_timeout=5.0,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "split_write",
        [
            param(False, id="poison_surfaces_on_head"),
            param(True, id="poison_surfaces_on_reused_connection"),
        ],
    )  # fmt: skip
    async def test_body_on_head_reply_still_falls_back_to_get(
        self, split_write: bool
    ) -> None:
        """A body on a HEAD reply must not mark a GET-serving endpoint unreachable."""
        async with MalformedHeadServer(split_write=split_write) as server:
            assert await self._collector(server.url).is_url_reachable() is True
            assert server.methods_seen.count("GET") == 1, (
                "GET fallback must reach the server after an unusable HEAD probe"
            )

    @pytest.mark.asyncio
    async def test_falls_back_when_server_ignores_connection_close(self) -> None:
        """The GET must not inherit poisoned bytes even if the server keeps the socket open."""
        async with MalformedHeadServer(
            split_write=True, ignore_connection_close=True
        ) as server:
            assert await self._collector(server.url).is_url_reachable() is True

    @pytest.mark.asyncio
    async def test_unreachable_endpoint_reports_false(self) -> None:
        """Nothing listening means neither probe succeeds."""
        collector = ConcreteCollector(
            endpoint_url="http://127.0.0.1:1/metrics",
            collection_interval=1.0,
            reachability_timeout=2.0,
        )

        assert await collector.is_url_reachable() is False


class TestReachabilityProbeRedactsCredentials:
    """The reachability probe must never log endpoint credentials.

    Two separate leak paths: the endpoint URL itself may embed userinfo, and
    some aiohttp errors render the requested URL verbatim in their repr --
    InvalidUrlClientError is an aiohttp.ClientError subclass, so it reaches the
    probe's except clause with credentials attached.
    """

    SECRET = "sup3rs3cret"  # noqa: S105 - test fixture, not a real credential

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "endpoint_url",
        [
            param(f"http://admin:{SECRET}@127.0.0.1:1/metrics", id="unreachable_host"),
            param(f"http://admin:{SECRET}@/metrics", id="invalid_url_error"),
        ],
    )  # fmt: skip
    async def test_probe_failure_never_logs_credentials(
        self, endpoint_url: str, caplog
    ) -> None:
        collector = ConcreteCollector(
            endpoint_url=endpoint_url,
            collection_interval=1.0,
            reachability_timeout=2.0,
        )

        with caplog.at_level("DEBUG"):
            assert await collector.is_url_reachable() is False

        assert self.SECRET not in caplog.text, (
            "endpoint credentials leaked into the reachability probe log"
        )
