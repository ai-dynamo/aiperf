# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import orjson

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import (
    ErrorDetails,
    RequestInfo,
    RequestRecord,
    SSEField,
    SSEMessage,
)
from aiperf.common.redact import redact_headers
from aiperf.common.utils import is_truthy_flag
from aiperf.plugin import plugins
from aiperf.plugin.enums import TransportType
from aiperf.transports.base_transports import (
    BaseTransport,
    FirstTokenCallback,
    TransportMetadata,
)

# Terminal Responses lifecycle events over the socket. ``response.completed``
# ends a turn successfully; the others end it in failure.
_TERMINAL_EVENT_TYPES = frozenset(
    {"response.completed", "response.failed", "response.incomplete", "error"}
)
_FAILURE_EVENT_TYPES = frozenset({"response.failed", "response.incomplete", "error"})

# stream_id charset per the OpenAI WebSocket-mode spec: 1-256 characters,
# alphanumeric plus underscore, hyphen, and period.
_STREAM_ID_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]")
_STREAM_ID_MAX_LEN = 256

# Response-create envelope keys that are HTTP-transport-only and must be
# stripped before sending over the socket (the socket always streams events).
_HTTP_ONLY_PAYLOAD_KEYS = ("stream", "stream_options", "background")

# Sentinel returned by frame decoding when the socket closed or errored.
_SOCKET_CLOSED = object()


def _decode_event(data: str) -> dict[str, Any] | None:
    """Parse a frame payload into an event dict, or None if not a JSON object."""
    try:
        event = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _has_ws_scheme(url: str) -> bool:
    """Return True if ``url`` already starts with ``ws://`` or ``wss://``."""
    lowered = url.lower()
    return lowered.startswith(("ws://", "wss://"))


def _sanitize_stream_id(raw: str | None) -> str | None:
    """Coerce a session identifier into a spec-valid ``stream_id`` lane name.

    Returns ``None`` (the default lane) when there is no usable identifier.
    """
    if not raw:
        return None
    cleaned = _STREAM_ID_ALLOWED.sub("_", raw)[:_STREAM_ID_MAX_LEN]
    return cleaned or None


class WebSocketTransport(BaseTransport):
    """WebSocket transport for the OpenAI Responses API (``wss://.../v1/responses``).

    WebSocket mode keeps a persistent socket per conversation and streams the
    same ``response.*`` lifecycle events the HTTP SSE path emits, so the
    Responses endpoint's ``parse_response`` works unchanged: each event is
    appended as an :class:`SSEMessage`.

    Each conversation holds a dedicated socket for its whole lifetime -- a hard
    lease keyed on ``x_correlation_id``, mirroring the HTTP
    ``STICKY_USER_SESSIONS`` strategy. Every turn of a conversation therefore
    reuses the same connection, so ``previous_response_id`` chaining stays inside
    the server's connection-local cache even when turns from other conversations
    interleave. The socket is opened on the first turn and closed on the final
    turn (or on error). Requests carry a ``stream_id`` derived from the session
    so forked conversations replaying against the same server stay addressable.

    Open sockets track the number of concurrently active conversations, not the
    number of in-flight requests. ``--concurrency`` caps concurrent sessions (a
    session slot is held first turn through final turn), so at saturation the
    socket count equals the session concurrency; under request-rate load, where
    session concurrency is unbounded, the socket count instead tracks how many
    conversations are active at once. A request with no conversation identity
    (no ``x_correlation_id``) gets a fresh socket that is closed on completion.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session: aiohttp.ClientSession | None = None
        # Hard per-conversation leases (x_correlation_id -> its pinned socket)
        # and the set of all live sockets for teardown. Guarded by ``_lock``.
        self._leases: dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._all: set[aiohttp.ClientWebSocketResponse] = set()
        self._lock = asyncio.Lock()

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return WebSocket transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.WEBSOCKET,
            url_schemes=["ws", "wss"],
        )

    @on_init
    async def _init_session(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None)
        self._session = aiohttp.ClientSession(timeout=timeout)

    @on_stop
    async def _close_session(self) -> None:
        async with self._lock:
            sockets = list(self._all)
            self._all.clear()
            self._leases.clear()
        for ws in sockets:
            if not ws.closed:
                await ws.close()
        if self._session is not None:
            session = self._session
            self._session = None
            await session.close()

    def get_url(self, request_info: RequestInfo) -> str:
        """Build the ``ws(s)://host/v1/responses`` URL from the base URL.

        Reuses the shared overlap-aware path-join (:meth:`_dedup_path_overlap`)
        against the endpoint metadata ``endpoint_path`` while preserving the
        ``ws``/``wss`` scheme, so a ``ws(s)://host/v1`` base and a metadata path
        of ``v1/responses`` collapse to ``/v1/responses`` rather than
        ``/v1/v1/responses``. A base URL without a scheme defaults to ``ws://``.
        """
        endpoint_info = request_info.model_endpoint.endpoint
        raw_base_url = endpoint_info.get_url(request_info.url_index)
        if not _has_ws_scheme(raw_base_url):
            raw_base_url = f"ws://{raw_base_url}"

        split = urlsplit(raw_base_url)
        base_path = split.path.rstrip("/")

        if endpoint_info.custom_endpoint is not None:
            sub_path = endpoint_info.custom_endpoint.lstrip("/")
        else:
            endpoint_metadata = plugins.get_endpoint_metadata(endpoint_info.type)
            sub_path = (endpoint_metadata.endpoint_path or "").lstrip("/")

        new_path = self._dedup_path_overlap(base_path, sub_path)

        return urlunsplit(
            (split.scheme, split.netloc, new_path, split.query, split.fragment)
        )

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any] | bytes,
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send one ``response.create`` over a socket and read until terminal.

        Args:
            request_info: Request context and metadata.
            payload: Endpoint-formatted Responses payload (dict or JSON bytes).
            first_token_callback: Fired on the first content-bearing event.

        Returns:
            Record whose ``responses`` hold the streamed events as SSEMessages.
        """
        if self._session is None:
            raise NotInitializedError(
                "WebSocketTransport not initialized. Call initialize() before "
                "send_request()."
            )

        envelope = self._build_envelope(payload, request_info)
        headers = self.build_headers(request_info)
        correlation_id = request_info.x_correlation_id
        # store may be requested endpoint-wide (--extra-inputs) or per turn (a
        # dataset row's extra, which lands on the envelope). Either persists the
        # response server-side, so both chaining checks below must honor it.
        store_requested = self._store_requested() or is_truthy_flag(
            envelope.get("store")
        )

        dirty = False
        start_perf_ns = time.perf_counter_ns()
        record = RequestRecord(
            request_info=request_info,
            timestamp_ns=time.time_ns(),
            start_perf_ns=start_perf_ns,
        )
        # Opening the socket can fail (refused connection, or a handshake that
        # exceeds the endpoint timeout); fold both into the failed-record path
        # rather than letting them escape send_request.
        try:
            ws, reconnected = await self._acquire(request_info, headers)
        except asyncio.CancelledError:
            record.cancellation_perf_ns = time.perf_counter_ns()
            record.error = ErrorDetails(
                code=499, type="CancelledError", message="Request cancelled"
            )
            raise
        except TimeoutError:
            record.error = ErrorDetails(
                code=408,
                type="TimeoutError",
                message=(
                    "WebSocket handshake exceeded the "
                    f"{self.model_endpoint.endpoint.timeout}s endpoint timeout"
                ),
            )
            record.end_perf_ns = time.perf_counter_ns()
            record.request_headers = redact_headers(headers)
            return record
        except Exception as e:
            record.error = ErrorDetails.from_exception(e)
            record.end_perf_ns = time.perf_counter_ns()
            record.request_headers = redact_headers(headers)
            return record
        # A mid-conversation reconnect loses the connection-local response cache
        # that non-stored chaining depends on. Sending the chained turn (only the
        # newest turn plus previous_response_id) would draw a confusing
        # previous_response_not_found from the server. Fail the turn with a clear
        # cause instead. When store was requested the id is persisted server-side,
        # so a fresh socket still resolves it and the turn proceeds normally.
        if reconnected and request_info.previous_response_id and not store_requested:
            record.error = ErrorDetails(
                code=None,
                type="ChainingContextLost",
                message=(
                    "WebSocket reconnected mid-conversation; "
                    f"previous_response_id {request_info.previous_response_id!r} "
                    "was cached on the closed socket and cannot be resolved "
                    "without server-side store (--extra-inputs store:true)."
                ),
            )
            record.end_perf_ns = time.perf_counter_ns()
            record.request_headers = redact_headers(headers)
            await self._release(ws, correlation_id, request_info.is_final_turn, True)
            return record
        try:
            # The Responses WebSocket contract requires JSON text frames; binary
            # frames are rejected with an invalid_request_error.
            await ws.send_str(orjson.dumps(envelope).decode())
            # cancel_after_ns (cancellation benchmarks) is measured from the moment
            # the request is fully sent, matching the HTTP transport. Bound the read
            # by it and return a 499 record on expiry rather than reading through to
            # a terminal event or the endpoint timeout.
            cancel_after_ns = request_info.cancel_after_ns
            if cancel_after_ns is not None:
                sent_perf_ns = time.perf_counter_ns()
                try:
                    dirty = await asyncio.wait_for(
                        self._read_until_terminal(
                            ws, record, start_perf_ns, first_token_callback
                        ),
                        timeout=cancel_after_ns / NANOS_PER_SECOND,
                    )
                except TimeoutError:
                    dirty = True
                    cancel_perf_ns = time.perf_counter_ns()
                    record.cancellation_perf_ns = cancel_perf_ns
                    elapsed_s = (cancel_perf_ns - sent_perf_ns) / NANOS_PER_SECOND
                    record.error = ErrorDetails(
                        code=499,
                        type="RequestCancellationError",
                        message=f"Request cancelled {elapsed_s:.3f}s after being sent",
                    )
            else:
                dirty = await self._read_until_terminal(
                    ws, record, start_perf_ns, first_token_callback
                )
        except asyncio.CancelledError:
            dirty = True
            record.cancellation_perf_ns = time.perf_counter_ns()
            record.error = ErrorDetails(
                code=499, type="CancelledError", message="Request cancelled"
            )
            raise
        except Exception as e:
            dirty = True
            record.error = ErrorDetails.from_exception(e)
        finally:
            record.end_perf_ns = time.perf_counter_ns()
            record.request_headers = redact_headers(headers)
            await self._release(ws, correlation_id, request_info.is_final_turn, dirty)
        return record

    def _build_envelope(
        self, payload: dict[str, Any] | bytes, request_info: RequestInfo
    ) -> dict[str, Any]:
        """Wrap the endpoint payload in a ``response.create`` event.

        Strips HTTP-only transport fields (``stream`` / ``stream_options`` /
        ``background``) and tags the turn with a session-derived ``stream_id``
        lane so parallel/forked conversations stay addressable.
        """
        if isinstance(payload, bytes):
            envelope: dict[str, Any] = orjson.loads(payload)
        else:
            envelope = dict(payload)
        for key in _HTTP_ONLY_PAYLOAD_KEYS:
            envelope.pop(key, None)
        envelope["type"] = "response.create"
        stream_id = _sanitize_stream_id(request_info.x_correlation_id)
        if stream_id is not None:
            envelope["stream_id"] = stream_id
        return envelope

    async def _read_until_terminal(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        record: RequestRecord,
        start_perf_ns: int,
        first_token_callback: FirstTokenCallback | None,
    ) -> bool:
        """Read events into ``record`` until a terminal response event.

        One request is in flight per socket, so every event on the socket
        belongs to this request; no ``stream_id`` demultiplexing is needed (and
        servers such as vLLM agentic-api do not echo ``stream_id`` on events
        anyway). Returns True when the socket became unusable (closed/errored)
        and must not be returned to the idle pool.
        """
        first_token_found = False
        deadline = self._receive_deadline(start_perf_ns)
        while True:
            try:
                msg = await self._receive(ws, deadline)
            except TimeoutError:
                record.error = ErrorDetails(
                    code=408,
                    type="TimeoutError",
                    message=(
                        "WebSocket turn exceeded the "
                        f"{self.model_endpoint.endpoint.timeout}s endpoint timeout "
                        "without a terminal response event"
                    ),
                )
                return True
            perf_ns = time.perf_counter_ns()

            data = self._frame_payload(msg, ws, record)
            if data is _SOCKET_CLOSED:
                return True
            if data is None:
                continue

            event = _decode_event(data)
            if event is None:
                continue

            if record.recv_start_perf_ns is None:
                record.recv_start_perf_ns = perf_ns

            sse = SSEMessage(
                perf_ns=perf_ns, packets=[SSEField(name="data", value=data)]
            )
            record.responses.append(sse)

            if not first_token_found and first_token_callback is not None:
                ttft_ns = perf_ns - start_perf_ns
                first_token_found = await first_token_callback(ttft_ns, sse)

            terminal = self._apply_terminal_event(event, record)
            if terminal is not None:
                return terminal

    def _receive_deadline(self, start_perf_ns: int) -> int | None:
        """Per-turn ``perf_counter_ns`` deadline, or None when timeout is disabled.

        Bounds a whole turn by the endpoint timeout (matching the HTTP transport's
        total-request timeout), so a silent or never-terminating peer cannot pin
        the leased socket open forever. ``timeout=0`` means no timeout.
        """
        timeout = self.model_endpoint.endpoint.timeout
        if not timeout or timeout <= 0:
            return None
        return start_perf_ns + int(timeout * NANOS_PER_SECOND)

    @staticmethod
    async def _receive(
        ws: aiohttp.ClientWebSocketResponse, deadline: int | None
    ) -> aiohttp.WSMessage:
        """Receive one frame, bounded by ``deadline`` (a ``perf_counter_ns`` value).

        Uses ``asyncio.wait_for`` rather than ``ws.receive(timeout=...)`` so the
        behavior is stable across aiohttp versions (``ws_receive`` defaults to
        ``None`` -- wait forever -- in aiohttp >=3.14.3). Raises ``TimeoutError``
        when the deadline passes.
        """
        if deadline is None:
            return await ws.receive()
        remaining = (deadline - time.perf_counter_ns()) / NANOS_PER_SECOND
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(ws.receive(), timeout=remaining)

    def _store_requested(self) -> bool:
        """Whether the run requested server-side storage via endpoint ``extra``.

        Mirrors ``ResponsesEndpoint._store_requested``: with ``store: true`` the
        server persists responses, so ``previous_response_id`` survives a socket
        reconnect; without it, chaining lives only in the connection-local cache.
        """
        store = dict(self.model_endpoint.endpoint.extra or []).get("store")
        return is_truthy_flag(store)

    def _frame_payload(
        self,
        msg: aiohttp.WSMessage,
        ws: aiohttp.ClientWebSocketResponse,
        record: RequestRecord,
    ) -> str | object | None:
        """Decode one frame into its JSON text payload.

        Returns the payload string for TEXT/BINARY frames, ``None`` for frames
        to skip, and the ``_SOCKET_CLOSED`` sentinel when the socket closed or
        errored (in which case ``record.error`` is set).
        """
        if msg.type is aiohttp.WSMsgType.TEXT:
            return msg.data
        if msg.type is aiohttp.WSMsgType.BINARY:
            return msg.data.decode("utf-8")
        if msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
        ):
            record.error = ErrorDetails(
                code=1000,
                type="ConnectionClosed",
                message="WebSocket closed before a terminal response event",
            )
            return _SOCKET_CLOSED
        if msg.type is aiohttp.WSMsgType.ERROR:
            record.error = ErrorDetails.from_exception(
                ws.exception() or Exception("WebSocket error")
            )
            return _SOCKET_CLOSED
        return None

    def _apply_terminal_event(
        self, event: dict[str, Any], record: RequestRecord
    ) -> bool | None:
        """Handle a terminal response event, if this one is terminal.

        Returns ``None`` for non-terminal events, ``False`` on success (socket
        reusable), and ``True`` on failure (socket must be discarded).
        """
        event_type = event.get("type")
        if event_type not in _TERMINAL_EVENT_TYPES:
            return None
        if event_type in _FAILURE_EVENT_TYPES:
            record.error = self._error_from_event(event)
            return True
        record.status = 200
        return False

    @staticmethod
    def _error_from_event(event: dict[str, Any]) -> ErrorDetails:
        """Build ErrorDetails from a failure event's nested error/response."""
        err = event.get("error")
        if isinstance(err, dict):
            return ErrorDetails(
                code=err.get("code"),
                type=err.get("type") or event.get("type"),
                message=err.get("message") or "WebSocket response failed",
            )
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            nested = response["error"]
            return ErrorDetails(
                code=nested.get("code"),
                type=nested.get("type") or event.get("type"),
                message=nested.get("message") or "WebSocket response failed",
            )
        return ErrorDetails(
            code=None,
            type=event.get("type"),
            message="WebSocket response did not complete successfully",
        )

    async def _acquire(
        self, request_info: RequestInfo, headers: dict[str, str]
    ) -> tuple[aiohttp.ClientWebSocketResponse, bool]:
        """Reuse the conversation's leased socket, or open and lease a new one.

        Session concurrency serializes a conversation's turns, so its leased
        socket is never in use by two turns at once. A request with no
        ``x_correlation_id`` always gets a fresh socket (closed on release).

        Returns ``(socket, reconnected)`` where ``reconnected`` is True when a
        prior lease for this conversation existed but its socket was already
        closed, so a fresh one was opened. The connection-local response cache
        that ``previous_response_id`` chaining relies on died with that socket.
        """
        correlation_id = request_info.x_correlation_id
        stale: aiohttp.ClientWebSocketResponse | None = None
        if correlation_id:
            async with self._lock:
                leased = self._leases.get(correlation_id)
                if leased is not None and not leased.closed:
                    return leased, False
                stale = leased

        # Open outside the lock so concurrent opens do not serialize.
        ws = await self._open(request_info, headers)
        async with self._lock:
            # Drop the peer-closed socket the new one replaces; it never passes
            # through _release, so leaving it in _all leaks a ClientWebSocketResponse
            # per reconnect until shutdown.
            if stale is not None:
                self._all.discard(stale)
            self._all.add(ws)
            if correlation_id:
                self._leases[correlation_id] = ws
        return ws, stale is not None

    async def _open(
        self, request_info: RequestInfo, headers: dict[str, str]
    ) -> aiohttp.ClientWebSocketResponse:
        """Open a socket, bounding the handshake by the endpoint timeout.

        ``ws_connect`` (through a session with ``ClientTimeout(total=None)``) has
        no handshake deadline of its own, so a peer that accepts the TCP
        connection but never completes the upgrade would hang the turn forever.
        Cap the handshake at ``endpoint.timeout`` (``timeout<=0`` disables it);
        a breach raises ``TimeoutError``, which ``send_request`` turns into the
        normal failed-record path.
        """
        assert self._session is not None
        url = self.build_url(request_info)

        async def _connect() -> aiohttp.ClientWebSocketResponse:
            return await self._session.ws_connect(url, headers=headers, autoping=True)

        timeout = self.model_endpoint.endpoint.timeout
        if not timeout or timeout <= 0:
            return await _connect()
        return await asyncio.wait_for(_connect(), timeout=timeout)

    async def _release(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        correlation_id: str | None,
        is_final_turn: bool,
        dirty: bool,
    ) -> None:
        """Keep the leased socket for the next turn, or close it and drop the lease.

        A non-final clean turn leaves the socket pinned to its conversation. The
        final turn, an errored turn, a socket the peer closed, and any request
        without a conversation identity drop the lease and close the socket.
        """
        keep = (
            bool(correlation_id) and not is_final_turn and not dirty and not ws.closed
        )
        to_close: aiohttp.ClientWebSocketResponse | None = None
        async with self._lock:
            if keep:
                return
            if correlation_id and self._leases.get(correlation_id) is ws:
                self._leases.pop(correlation_id, None)
            self._all.discard(ws)
            to_close = ws if not ws.closed else None
        if to_close is not None:
            await to_close.close()
