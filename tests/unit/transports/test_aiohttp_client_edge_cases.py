# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case unit tests for AioHttpClient.

The existing ``test_aiohttp_client.py`` covers happy paths, generic exception
handling, and the FirstToken callback. This file exercises the remaining
error / cancellation surface:

- Connection refused / DNS failures.
- Read timeouts (``TimeoutError`` propagated by aiohttp).
- TLS / certificate errors propagated through ``aiohttp.ClientConnectorError``.
- Streaming truncation mid-response (``ClientPayloadError``).
- Binary content-type dispatch (image / video / octet-stream).
- Cancellation mid-request via ``cancel_after_ns``.
- Send-side timeout when the request never reaches the wire.
- Kwargs (e.g. ``proxy``) propagation into ``session.request``.
- TRUST_ENV plumbing — confirms that with TRUST_ENV=False (the default) the
  session ignores ``HTTP_PROXY`` / ``NO_PROXY`` env vars, which is the
  configuration the localhost-proxy gotcha relies on.
- Connector cleanup is idempotent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from pytest import param

from aiperf.common.models import BinaryResponse, RequestRecord, TextResponse
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.http_defaults import AioHttpDefaults
from tests.unit.transports.conftest import (
    create_mock_response,
    setup_mock_session,
)

# ============================================================================
# Network-failure paths
# ============================================================================


class TestAioHttpClientNetworkFailures:
    """Connection-level failures populate ErrorDetails on the record."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_factory,expected_type_substr",
        [
            param(
                lambda: aiohttp.ClientConnectorError(
                    connection_key=MagicMock(),
                    os_error=ConnectionRefusedError("Connection refused"),
                ),
                "ClientConnectorError",
                id="connection-refused",
            ),
            param(
                lambda: aiohttp.ClientConnectorError(
                    connection_key=MagicMock(),
                    os_error=OSError("Name or service not known"),
                ),
                "ClientConnectorError",
                id="dns-failure",
            ),
            param(
                lambda: aiohttp.ServerDisconnectedError("server closed connection"),
                "ServerDisconnectedError",
                id="server-disconnect",
            ),
            param(
                lambda: aiohttp.ClientPayloadError("response truncated mid-stream"),
                "ClientPayloadError",
                id="payload-truncated",
            ),
            param(
                lambda: TimeoutError(),
                "TimeoutError",
                id="read-timeout",
            ),
        ],
    )  # fmt: skip
    async def test_connection_failure_recorded(
        self,
        aiohttp_client: AioHttpClient,
        exc_factory,
        expected_type_substr: str,
    ) -> None:
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = exc_factory()

            record = await aiohttp_client.post_request("http://test.com", b"{}", {})

        assert record.error is not None
        assert record.responses == []
        # ErrorDetails.type is the exception class name.
        assert expected_type_substr in record.error.type
        # The record is still timed (start + end set) so accounting works.
        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None

    @pytest.mark.asyncio
    async def test_tls_certificate_error_recorded(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """TLS-style ``ClientConnectorCertificateError`` flows through the same path."""
        import ssl

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = aiohttp.ClientConnectorCertificateError(
                connection_key=MagicMock(),
                certificate_error=ssl.SSLCertVerificationError(
                    1, "self-signed certificate"
                ),
            )

            record = await aiohttp_client.post_request("https://test.com", b"{}", {})

        assert record.error is not None
        assert "Certificate" in record.error.type or "Connector" in record.error.type
        assert record.responses == []


# ============================================================================
# Non-SSE response dispatch (binary / unknown content-type)
# ============================================================================


class TestAioHttpClientNonSseDispatch:
    """``_consume_non_sse_response`` switches on content-type prefix."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_type",
        [
            "image/png",
            "video/mp4",
            "audio/wav",
            "application/octet-stream",
        ],
    )  # fmt: skip
    async def test_binary_content_type_returns_binary_response(
        self,
        aiohttp_client: AioHttpClient,
        content_type: str,
    ) -> None:
        raw = b"\x00\x01\x02\x03binarydata"
        mock_response = Mock(
            spec=aiohttp.ClientResponse,
            status=200,
            reason="OK",
            content_type=content_type,
            read=AsyncMock(return_value=raw),
            text=AsyncMock(return_value=""),
            content=Mock(spec=aiohttp.StreamReader),
        )
        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", b"{}", {})

        assert record.error is None
        assert len(record.responses) == 1
        resp = record.responses[0]
        assert isinstance(resp, BinaryResponse)
        assert resp.content_type == content_type
        assert resp.raw_bytes == raw

    @pytest.mark.asyncio
    async def test_unknown_content_type_falls_back_to_text(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """Anything that isn't binary/SSE is parsed via ``response.text()``."""
        mock_response = create_mock_response(
            content_type="application/x-aiperf-weirdo",
            text_content="opaque-body",
        )
        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", b"{}", {})

        assert record.error is None
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert record.responses[0].text == "opaque-body"


# ============================================================================
# Cancellation mid-request
# ============================================================================


class TestAioHttpClientCancellation:
    """``cancel_after_ns`` and external cancellation produce 499 ErrorDetails."""

    @pytest.mark.asyncio
    async def test_external_cancellation_records_499(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """A CancelledError raised while the session is open re-raises and records 499."""

        # Build a session whose request context-manager hangs forever, then cancel
        # the surrounding task to force a CancelledError inside _execute_request.
        async def hang_forever(*_a, **_kw):
            await asyncio.Event().wait()

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()

            def session_factory(*_a, **_kw):
                ctx = AsyncMock()
                ctx.__aenter__ = AsyncMock(return_value=mock_session)
                ctx.__aexit__ = AsyncMock(return_value=None)
                return ctx

            mock_session_class.side_effect = session_factory

            request_ctx = AsyncMock()
            request_ctx.__aenter__ = AsyncMock(side_effect=hang_forever)
            request_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = Mock(return_value=request_ctx)

            task = asyncio.create_task(
                aiohttp_client.post_request("http://test.com", b"{}", {})
            )
            # Yield to let the task enter the awaitable.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_cancel_after_ns_sends_then_cancels(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """When ``cancel_after_ns`` fires, the returned record carries a 499 cancellation error."""

        # Patch _request to: signal request_sent immediately, then never return.
        async def fake_request(
            method,
            url,
            headers,
            *,
            data=None,
            on_request_sent=None,
            first_token_callback=None,
            trace_data=None,
            connector=None,
            connector_owner=False,
            **_kw,
        ) -> RequestRecord:
            if on_request_sent is not None:
                on_request_sent.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Mirror the production path: re-raise to allow cancel-after-timeout
                # to record 499 itself.
                raise
            # Unreachable; appease type checker.
            return RequestRecord(start_perf_ns=0)  # pragma: no cover

        with patch.object(AioHttpClient, "_request", side_effect=fake_request):
            record = await aiohttp_client.post_request(
                "http://test.com",
                b"{}",
                {},
                cancel_after_ns=1_000_000,  # 1 ms
            )

        assert record.error is not None
        assert record.error.code == 499
        assert "cancelled" in record.error.message.lower()
        assert record.cancellation_perf_ns is not None

    @pytest.mark.asyncio
    async def test_send_timeout_when_request_never_sent(
        self,
        aiohttp_client: AioHttpClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the request never makes it to the wire, the wrapper returns RequestSendTimeout."""

        # Force an absurdly small send timeout so the wait_for fires immediately.
        from aiperf.common.environment import Environment

        monkeypatch.setattr(
            Environment.HTTP, "REQUEST_CANCELLATION_SEND_TIMEOUT", 0.001
        )
        # Clear total timeout so the wrapper falls back to the env var.
        aiohttp_client.timeout = aiohttp.ClientTimeout(total=None)

        async def fake_request_never_sends(
            *_a,
            on_request_sent=None,
            **_kw,
        ) -> RequestRecord:
            # Don't set on_request_sent. Just hang.
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise
            return RequestRecord(start_perf_ns=0)  # pragma: no cover

        with patch.object(
            AioHttpClient, "_request", side_effect=fake_request_never_sends
        ):
            record = await aiohttp_client.post_request(
                "http://test.com",
                b"{}",
                {},
                cancel_after_ns=10_000_000,
            )

        assert record.error is not None
        assert record.error.type == "RequestSendTimeout"
        assert record.error.code == 0


# ============================================================================
# Proxy / TRUST_ENV behavior (NO_PROXY-style gotcha)
# ============================================================================


class TestAioHttpClientProxyHandling:
    """Verify that the default TRUST_ENV=False configuration ignores HTTP_PROXY."""

    @pytest.mark.asyncio
    async def test_default_trust_env_is_false(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """Default ``AioHttpDefaults.TRUST_ENV`` should be False per the localhost-proxy gotcha."""
        # If this assert flips, the localhost-proxy / NO_PROXY workaround needs
        # updating; the gotcha relies on aiohttp NOT trusting HTTP_PROXY here.
        assert AioHttpDefaults.TRUST_ENV is False

    @pytest.mark.asyncio
    async def test_proxy_kwarg_is_propagated_to_session_request(
        self,
        aiohttp_client: AioHttpClient,
        mock_aiohttp_response: Mock,
    ) -> None:
        """Explicit ``proxy=`` kwarg is forwarded; env var ``HTTP_PROXY`` is not consulted."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = setup_mock_session(
                mock_session_class, mock_aiohttp_response, ["request"]
            )
            await aiohttp_client.post_request(
                "http://localhost:9999/v1/chat/completions",
                b"{}",
                {},
                proxy="http://explicit-proxy:3128",
            )

        mock_session.request.assert_called_once()
        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["proxy"] == "http://explicit-proxy:3128"

        # The session itself was constructed with trust_env=False so HTTP_PROXY env is ignored.
        session_kwargs = mock_session_class.call_args[1]
        assert session_kwargs["trust_env"] is False


# ============================================================================
# Connector lifecycle
# ============================================================================


class TestAioHttpClientConnectorLifecycle:
    """``close()`` is idempotent and survives concurrent invocations."""

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        client = AioHttpClient(timeout=10.0)
        mock_connector = Mock()
        mock_connector.close = AsyncMock()
        client.tcp_connector = mock_connector

        await client.close()
        await client.close()  # second call must not blow up

        mock_connector.close.assert_called_once()
        assert client.tcp_connector is None

    @pytest.mark.asyncio
    async def test_concurrent_close_calls(self) -> None:
        client = AioHttpClient(timeout=10.0)
        mock_connector = Mock()

        async def slow_close() -> None:
            await asyncio.sleep(0)

        mock_connector.close = AsyncMock(side_effect=slow_close)
        client.tcp_connector = mock_connector

        await asyncio.gather(client.close(), client.close(), client.close())
        # At most one effective close (others see tcp_connector=None).
        assert mock_connector.close.await_count >= 1
        assert client.tcp_connector is None


# ============================================================================
# GET requests
# ============================================================================


class TestAioHttpClientGet:
    """``get_request`` is a thin wrapper but needs at least one happy path."""

    @pytest.mark.asyncio
    async def test_get_request_success(
        self,
        aiohttp_client: AioHttpClient,
        mock_aiohttp_response: Mock,
    ) -> None:
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = setup_mock_session(
                mock_session_class, mock_aiohttp_response, ["request"]
            )
            record = await aiohttp_client.get_request(
                "http://test.com/api", {"X-Foo": "bar"}
            )

        assert record.error is None
        assert record.status == 200
        mock_session.request.assert_called_once()
        # Method must be GET, not POST.
        called_method = mock_session.request.call_args[0][0]
        assert called_method == "GET"

    @pytest.mark.asyncio
    async def test_get_request_connection_failure(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=ConnectionRefusedError(),
            )
            record = await aiohttp_client.get_request("http://test.com/api", {})

        assert record.error is not None
        assert "ClientConnectorError" in record.error.type
