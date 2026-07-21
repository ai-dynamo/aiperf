# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures and utilities for ZMQ testing.

This module provides reusable fixtures, mocks, and helpers for testing ZMQ functionality.
"""

import asyncio
import contextlib
import itertools
import socket as stdlib_socket
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import zmq.asyncio

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums import CreditPhase, LifecycleState
from aiperf.common.messages import HeartbeatMessage
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    WorkerReady,
    WorkerShutdown,
)
from aiperf.credit.structs import Credit
from aiperf.zmq.fd_reader import FdEdgeReader as _RealFdEdgeReader


@pytest.fixture(scope="session")
def event_loop_policy():
    """Force the selector loop on Windows so the FD-driver can register.

    The FD-driver calls ``loop.add_reader(getsockopt(zmq.FD), ...)``, which the
    default Windows ``ProactorEventLoop`` does not implement (raises
    ``NotImplementedError``). Production switches to the selector policy in
    ``bootstrap._configure_event_loop_policy_for_platform``; the test loop never
    runs bootstrap, so mirror that here. No-op on Linux/macOS.
    """
    if IS_WINDOWS:
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


async def _block_forever():
    """Block forever by awaiting a Future that never completes."""
    await asyncio.Future()  # Never resolves


def _create_recv_function_with_blocking(side_effect_items):
    """Create an async recv function that processes items then blocks forever.

    Args:
        side_effect_items: Iterator of items (bytes, exceptions, etc.)

    Returns:
        Async function that returns/raises items, then blocks forever
    """
    side_effect_iter = iter(side_effect_items)

    async def recv_with_blocking():
        """Return items from list, then block forever to prevent busy loop."""
        try:
            item = next(side_effect_iter)
            # If item is an exception, raise it
            if isinstance(item, Exception):
                raise item
            return item
        except StopIteration:
            # List exhausted, block forever to prevent busy loop
            await _block_forever()

    return recv_with_blocking


def _install_fd_support(socket):
    """Make a mock socket drivable by the FdEdgeReader / sync-FD path.

    The streaming/push/pub/sub/pull clients now drive the raw socket FD: they
    register ``loop.add_reader(getsockopt(zmq.FD), ...)`` and drain with the
    UNBOUND sync ops ``zmq.Socket.send/recv`` + ``recv_into`` (bypassing the
    asyncio overrides). This wires a mock socket for that path using a frame
    queue so ``getsockopt(zmq.EVENTS)`` reflects whether data is pending:

    - ``getsockopt(zmq.FD)`` returns a real (idle) socket fd so
      ``loop.add_reader`` installs cleanly (the socket is never written, so the
      reader fires only via the bootstrap drain at start()). A socketpair is used
      rather than ``os.pipe`` because pipe fds are not selectable on Windows; a
      socket fd is selectable under the selector loop on every platform.
    - ``getsockopt(zmq.EVENTS)`` reports ``POLLIN`` iff ``_recv_frames`` is
      non-empty (always ``POLLOUT``). When the queue drains, EVENTS goes to
      ``POLLOUT`` only, so the edge-triggered drain stops — no busy loop.
    - ``_sync_recv`` / ``recv_into`` pop one frame; ``RCVMORE`` reflects the
      last-popped frame's more-flag (multipart). A queued ``Exception`` is
      raised instead of returned.
    - ``_sync_send`` records sync sends. The async ``send``/``recv`` mocks are
      left intact for the unchanged REQ/REP clients.

    Use ``enqueue_recv_frames(socket, [b"a", b"b"])`` for a multipart message or
    ``enqueue_recv_messages(socket, [m1, m2])`` for N single-frame messages.
    """
    import collections

    r_sock, w_sock = stdlib_socket.socketpair()
    r_sock.setblocking(False)
    socket._test_socketpair = (r_sock, w_sock)
    fd = r_sock.fileno()
    # Each entry: (payload_or_exc, more: bool).
    socket._recv_frames = collections.deque()
    socket._last_more = False
    socket._pollout = True

    def _getsockopt(opt):
        if opt == zmq.FD:
            return fd
        if opt == zmq.EVENTS:
            events = zmq.POLLOUT if socket._pollout else 0
            if socket._recv_frames:
                events |= zmq.POLLIN
            return events
        if opt == zmq.RCVMORE:
            return 1 if socket._last_more else 0
        return 0

    def _pop_frame():
        if not socket._recv_frames:
            raise zmq.Again()
        payload, more = socket._recv_frames.popleft()
        socket._last_more = more
        if isinstance(payload, BaseException):
            raise payload
        return payload

    def _recv_into(buf, *args, **kwargs):
        data = _pop_frame()
        n = len(data)
        buf[:n] = data
        return n

    socket.getsockopt = Mock(side_effect=_getsockopt)
    socket._sync_recv = MagicMock(side_effect=lambda *a, **k: _pop_frame())
    socket.recv_into = MagicMock(side_effect=_recv_into)
    socket._sync_send = MagicMock(name="zmq.Socket.send(sync)")
    return socket


def enqueue_recv_frames(socket, frames):
    """Queue one multipart message (list of byte frames) for the FD drain."""
    n = len(frames)
    for i, frame in enumerate(frames):
        socket._recv_frames.append((frame, i < n - 1))


def enqueue_recv_messages(socket, messages):
    """Queue N single-frame messages (bytes or Exceptions) for the FD drain."""
    for msg in messages:
        socket._recv_frames.append((msg, False))


def _close_fd_support(socket):
    for sock in getattr(socket, "_test_socketpair", ()):
        with contextlib.suppress(OSError):
            sock.close()


class _TestSafeFdEdgeReader(_RealFdEdgeReader):
    """FdEdgeReader that refuses to reschedule on a non-int EVENTS value.

    An uninstrumented mock socket returns a truthy ``Mock`` from
    ``getsockopt(zmq.EVENTS)``; the real ``_rearm`` would then ``call_soon`` the
    pump forever (busy loop / OOM). Treat a non-int events value as "no work",
    so such sockets leave the driver idle. Instrumented sockets (EVENTS = int
    via ``_install_fd_support``) behave exactly as in production.
    """

    def _rearm(self, events=None):
        if events is None:
            try:
                events = self._socket.getsockopt(zmq.EVENTS)
            except Exception:  # mock teardown / closed socket
                return
        if not isinstance(events, int):
            return
        super()._rearm(events)


@pytest.fixture(autouse=True)
def _safe_fd_edge_reader(monkeypatch):
    """Swap the FdEdgeReader the clients import for the busy-loop-safe variant."""
    for module in (
        "pull_client",
        "sub_client",
        "streaming_dealer_client",
        "streaming_router_client",
        "streaming_pull_client",
    ):
        with contextlib.suppress(AttributeError):
            monkeypatch.setattr(
                f"aiperf.zmq.{module}.FdEdgeReader", _TestSafeFdEdgeReader
            )


@pytest.fixture(autouse=True)
def _patch_sync_zmq_ops(monkeypatch):
    """Route the unbound sync ZMQ ops to per-socket recorders.

    The FD-driver / sync-send path calls ``zmq.Socket.send/recv(self.socket, ...)``
    (the base-class ops) which, on a mock socket, would hit the real C functions
    ("Socket operation on non-socket"). Delegate them to ``socket._sync_send`` /
    ``socket._sync_recv``. ``zmq.asyncio.Socket`` overrides ``send``/``recv`` with
    its async versions, so the REQ/REP clients' awaited calls are unaffected.
    """

    def _send(self, *args, **kwargs):
        try:
            rec = self._sync_send
        except AttributeError:
            rec = MagicMock(name="zmq.Socket.send(sync)")
            self._sync_send = rec
        return rec(*args, **kwargs)

    def _recv(self, *args, **kwargs):
        try:
            rec = self._sync_recv
        except AttributeError:
            rec = MagicMock(name="zmq.Socket.recv(sync)", side_effect=zmq.Again())
            self._sync_recv = rec
        return rec(*args, **kwargs)

    monkeypatch.setattr(zmq.Socket, "send", _send)
    monkeypatch.setattr(zmq.Socket, "recv", _recv)


@pytest.fixture
def mock_zmq_socket():
    """Create a mock ZMQ socket with common methods.

    Mocks the methods actually used by ZMQ clients:
    - send() / recv() - used by dealer, push, pull clients
    - send_multipart() / recv_multipart() - used by pub, sub, router clients

    By default, recv methods block forever (await on a never-completing Future)
    to avoid busy loops with mocked sleep. Tests should override these when
    they need specific return values. FD-driver support (real idle pipe FD,
    EVENTS=0, sync recorders) is installed via ``_install_fd_support``.
    """
    socket = AsyncMock(spec=zmq.asyncio.Socket)
    socket.bind = Mock()
    socket.connect = Mock()
    socket.close = Mock()
    socket.setsockopt = Mock()
    socket.send = AsyncMock()
    socket.send_multipart = AsyncMock()
    # Block forever instead of raising zmq.Again in a loop
    socket.recv = AsyncMock(side_effect=_block_forever)
    socket.recv_multipart = AsyncMock(side_effect=_block_forever)
    socket.closed = False
    _install_fd_support(socket)
    yield socket
    _close_fd_support(socket)


@pytest.fixture
def mock_zmq_context(mock_zmq_socket):
    """Create a mock ZMQ context that returns mock sockets."""
    context = MagicMock(spec=zmq.asyncio.Context)
    context.socket = Mock(return_value=mock_zmq_socket)
    context.term = Mock()
    return context


@pytest.fixture
def fd_enqueue():
    """Queue messages/frames for a mock socket's FD drain.

    ``messages``: list of single-frame payloads (DEALER/PULL).
    ``frames``: one multipart message as a list of frames (ROUTER/SUB).
    """

    def _enq(socket, messages=None, frames=None):
        if messages is not None:
            enqueue_recv_messages(socket, messages)
        if frames is not None:
            enqueue_recv_frames(socket, frames)

    return _enq


@pytest.fixture(autouse=True)
def auto_mock_zmq_context(mock_zmq_context, monkeypatch):
    """Automatically mock ZMQ context for all tests in this module.

    This prevents real ZMQ connections from being created during tests,
    which can cause freezing and crashes.
    """
    # Mock at the zmq.asyncio level to catch all Context.instance() calls
    monkeypatch.setattr("zmq.asyncio.Context.instance", lambda: mock_zmq_context)
    return mock_zmq_context


@pytest.fixture
def sample_message():
    """Create a sample heartbeat message for testing."""
    return HeartbeatMessage(
        service_id="test-service",
        state=LifecycleState.RUNNING,
        service_type="test",
        request_id="test-request-123",
    )


@pytest.fixture
def sample_message_json(sample_message):
    """Create a sample message JSON string."""
    return sample_message.model_dump_json()


# =============================================================================
# Credit/Worker Struct Fixtures (for streaming dealer/router tests)
# =============================================================================


@pytest.fixture
def sample_credit():
    """Create a sample credit struct for testing."""
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-001",
        x_correlation_id="corr-001",
        turn_index=0,
        num_turns=1,
        issued_at_ns=1000000000,
    )


@pytest.fixture
def sample_worker_ready():
    """Create a sample WorkerReady struct for testing."""
    return WorkerReady(worker_id="worker-1")


@pytest.fixture
def sample_worker_shutdown():
    """Create a sample WorkerShutdown struct for testing."""
    return WorkerShutdown(worker_id="worker-1")


@pytest.fixture
def sample_credit_return(sample_credit):
    """Create a sample CreditReturn struct for testing."""
    return CreditReturn(credit=sample_credit)


@pytest.fixture
def sample_cancel_credits():
    """Create a sample CancelCredits struct for testing."""
    return CancelCredits(credit_ids={1, 2, 3})


@pytest.fixture
def assert_socket_configured():
    """Helper to assert socket was configured with default options."""

    def _assert(socket: Mock) -> None:
        """Assert that socket was configured with expected options."""
        assert socket.setsockopt.called
        # Check that common socket options were set
        calls = socket.setsockopt.call_args_list
        option_names = [call[0][0] for call in calls]

        # Verify key socket options were set
        assert zmq.RCVTIMEO in option_names
        assert zmq.SNDTIMEO in option_names
        assert zmq.SNDHWM in option_names
        assert zmq.RCVHWM in option_names

    return _assert


@pytest.fixture
async def wait_for_background_task():
    """Helper to wait for background tasks to start."""

    async def _wait(iterations: int = 3) -> None:
        """Wait for background tasks to run by yielding to the event loop once.

        The iterations parameter is kept for API compatibility but a single
        yield is sufficient - tight loops of yields cause starvation.
        """
        await asyncio.sleep(0)

    return _wait


class BaseClientTestHelper:
    """Base helper class for ZMQ client tests with common functionality."""

    def __init__(self, mock_zmq_context, wait_for_background_task=None):
        self.mock_zmq_context = mock_zmq_context
        self.wait_for_background_task = wait_for_background_task

    def setup_mock_socket(
        self,
        recv_side_effect=None,
        recv_return_value=None,
        recv_multipart_side_effect=None,
        send_side_effect=None,
        send_multipart_side_effect=None,
    ):
        """Setup a mock socket with specified behavior.

        Args:
            recv_side_effect: Side effect for socket.recv() (used by dealer, pull)
            recv_return_value: Single return value for socket.recv(), then blocks forever
            recv_multipart_side_effect: Side effect for socket.recv_multipart() (used by sub, router)
            send_side_effect: Side effect for socket.send() (used by dealer, push)
            send_multipart_side_effect: Side effect for socket.send_multipart() (used by pub, router)
        """

        async def _block_forever():
            """Block forever by awaiting a Future that never completes."""
            await asyncio.Future()  # Never resolves

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()

        # Setup send methods
        if send_side_effect is not None:
            mock_socket.send = AsyncMock(side_effect=send_side_effect)
        else:
            mock_socket.send = AsyncMock()

        if send_multipart_side_effect is not None:
            mock_socket.send_multipart = AsyncMock(
                side_effect=send_multipart_side_effect
            )
        else:
            mock_socket.send_multipart = AsyncMock()

        # Setup recv
        if recv_side_effect is not None:
            # Chain side_effect with block_forever to prevent busy loop after exhaustion
            if isinstance(recv_side_effect, list):
                mock_socket.recv = _create_recv_function_with_blocking(recv_side_effect)
            else:
                # If it's a callable or single value, use as-is
                mock_socket.recv = AsyncMock(side_effect=recv_side_effect)
        elif recv_return_value is not None:
            # Return value once, then block forever instead of busy loop
            # Use generator expression to create fresh coroutines on each call
            mock_socket.recv = AsyncMock(
                side_effect=itertools.chain(
                    [recv_return_value], (_block_forever() for _ in itertools.count())
                )
            )
        else:
            # Default to blocking forever to prevent busy loop
            mock_socket.recv = AsyncMock(side_effect=_block_forever)

        # Setup recv_multipart
        if recv_multipart_side_effect is not None:
            # Chain side_effect with block_forever to prevent busy loop after exhaustion
            if isinstance(recv_multipart_side_effect, list):
                mock_socket.recv_multipart = _create_recv_function_with_blocking(
                    recv_multipart_side_effect
                )
            else:
                # If it's a callable or single value, use as-is
                mock_socket.recv_multipart = AsyncMock(
                    side_effect=recv_multipart_side_effect
                )
        else:
            # Default to blocking forever to prevent busy loop
            mock_socket.recv_multipart = AsyncMock(side_effect=_block_forever)

        _install_fd_support(mock_socket)

        # Feed the FD-drain queue: the production path reads via the raw FD
        # (sync recv/recv_into), not the async recv/recv_multipart. Translate the
        # configured side-effects so existing receiver tests deliver through it.
        # A bare zmq.Again means "no data" (idle), so it is skipped.
        def _as_list(items):
            if items is None:
                return []
            return items if isinstance(items, list) else [items]

        _skip = (zmq.Again, asyncio.CancelledError)
        for item in _as_list(recv_side_effect):
            if isinstance(item, _skip):
                continue
            enqueue_recv_messages(mock_socket, [item])
        for msg in _as_list(recv_multipart_side_effect):
            if isinstance(msg, _skip):
                continue
            if isinstance(msg, BaseException):
                enqueue_recv_messages(mock_socket, [msg])
            else:
                enqueue_recv_frames(mock_socket, list(msg))

        # The sync FD/send path records on _sync_send, not the async send /
        # send_multipart. Route any configured send error there so error-path
        # tests still fire (ROUTER/streaming framing goes frame-by-frame through
        # the sync send).
        if send_side_effect is not None:
            mock_socket._sync_send.side_effect = send_side_effect
        if send_multipart_side_effect is not None:
            mock_socket._sync_send.side_effect = send_multipart_side_effect

        self.mock_zmq_context.socket = Mock(return_value=mock_socket)
        return mock_socket

    @asynccontextmanager
    async def create_client(
        self,
        client_class,
        address="tcp://127.0.0.1:5555",
        bind=False,
        auto_start=False,
        client_kwargs=None,
        **mock_kwargs,
    ):
        """Create and manage a ZMQ client with optional mock setup.

        Args:
            client_class: The client class to instantiate
            address: Address for the client
            bind: Whether to bind or connect
            auto_start: Whether to start the client automatically
            client_kwargs: Additional kwargs for client constructor
            **mock_kwargs: Arguments passed to setup_mock_socket
        """
        if mock_kwargs:
            self.setup_mock_socket(**mock_kwargs)

        # Build client kwargs
        kwargs = {"address": address, "bind": bind}
        if client_kwargs:
            kwargs.update(client_kwargs)

        client = client_class(**kwargs)
        await client.initialize()

        if auto_start and self.wait_for_background_task:
            await client.start()
            await self.wait_for_background_task()

        try:
            yield client
        finally:
            await client.stop()
            socket = getattr(self.mock_zmq_context, "socket", None)
            if socket is not None and hasattr(socket, "return_value"):
                _close_fd_support(socket.return_value)


@pytest.fixture
def dealer_test_helper(mock_zmq_context, wait_for_background_task):
    """Provide a helper for ZMQDealerRequestClient tests."""
    from aiperf.zmq.dealer_request_client import ZMQDealerRequestClient

    helper = BaseClientTestHelper(mock_zmq_context, wait_for_background_task)

    # Create a wrapper that passes the client class
    class DealerHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self,
            address="tcp://127.0.0.1:5555",
            bind=False,
            auto_start=False,
            **mock_kwargs,
        ):
            async with self._base.create_client(
                ZMQDealerRequestClient,
                address=address,
                bind=bind,
                auto_start=auto_start,
                **mock_kwargs,
            ) as client:
                yield client

    return DealerHelper(helper)


@pytest.fixture
def router_test_helper(mock_zmq_context, wait_for_background_task):
    """Provide a helper for ZMQRouterReplyClient tests."""
    from aiperf.zmq.router_reply_client import ZMQRouterReplyClient

    helper = BaseClientTestHelper(mock_zmq_context, wait_for_background_task)

    class RouterHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self,
            address="tcp://127.0.0.1:5555",
            bind=True,
            auto_start=False,
            **mock_kwargs,
        ):
            async with self._base.create_client(
                ZMQRouterReplyClient,
                address=address,
                bind=bind,
                auto_start=auto_start,
                **mock_kwargs,
            ) as client:
                yield client

    return RouterHelper(helper)


@pytest.fixture
def pub_test_helper(mock_zmq_context):
    """Provide a helper for ZMQPubClient tests."""
    from aiperf.zmq.pub_client import ZMQPubClient

    helper = BaseClientTestHelper(mock_zmq_context)

    class PubHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self, address="tcp://127.0.0.1:5555", bind=True, **mock_kwargs
        ):
            async with self._base.create_client(
                ZMQPubClient,
                address=address,
                bind=bind,
                auto_start=False,
                **mock_kwargs,
            ) as client:
                yield client

    return PubHelper(helper)


@pytest.fixture
def sub_test_helper(mock_zmq_context, wait_for_background_task):
    """Provide a helper for ZMQSubClient tests."""
    from aiperf.zmq.sub_client import ZMQSubClient

    helper = BaseClientTestHelper(mock_zmq_context, wait_for_background_task)

    class SubHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self,
            address="tcp://127.0.0.1:5555",
            bind=False,
            auto_start=False,
            **mock_kwargs,
        ):
            async with self._base.create_client(
                ZMQSubClient,
                address=address,
                bind=bind,
                auto_start=auto_start,
                **mock_kwargs,
            ) as client:
                yield client

    return SubHelper(helper)


@pytest.fixture
def push_test_helper(mock_zmq_context):
    """Provide a helper for ZMQPushClient tests."""
    from aiperf.zmq.push_client import ZMQPushClient

    helper = BaseClientTestHelper(mock_zmq_context)

    class PushHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self, address="tcp://127.0.0.1:5555", bind=True, **mock_kwargs
        ):
            async with self._base.create_client(
                ZMQPushClient,
                address=address,
                bind=bind,
                auto_start=False,
                **mock_kwargs,
            ) as client:
                yield client

    return PushHelper(helper)


@pytest.fixture
def pull_test_helper(mock_zmq_context, wait_for_background_task):
    """Provide a helper for ZMQPullClient tests."""
    from aiperf.zmq.pull_client import ZMQPullClient

    helper = BaseClientTestHelper(mock_zmq_context, wait_for_background_task)

    class PullHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self,
            address="tcp://127.0.0.1:5555",
            bind=False,
            auto_start=False,
            max_pull_concurrency=None,
            **mock_kwargs,
        ):
            client_kwargs = {}
            if max_pull_concurrency is not None:
                client_kwargs["max_pull_concurrency"] = max_pull_concurrency

            async with self._base.create_client(
                ZMQPullClient,
                address=address,
                bind=bind,
                auto_start=auto_start,
                client_kwargs=client_kwargs,
                **mock_kwargs,
            ) as client:
                yield client

    return PullHelper(helper)


@pytest.fixture
def streaming_router_test_helper(mock_zmq_context, wait_for_background_task):
    """Provide a helper for ZMQStreamingRouterClient tests."""
    from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

    helper = BaseClientTestHelper(mock_zmq_context, wait_for_background_task)

    class StreamingRouterHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self,
            address="tcp://*:5555",
            bind=True,
            auto_start=False,
            **mock_kwargs,
        ):
            async with self._base.create_client(
                ZMQStreamingRouterClient,
                address=address,
                bind=bind,
                auto_start=auto_start,
                **mock_kwargs,
            ) as client:
                yield client

    return StreamingRouterHelper(helper)


@pytest.fixture
def streaming_dealer_test_helper(mock_zmq_context, wait_for_background_task):
    """Provide a helper for ZMQStreamingDealerClient tests."""
    from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

    helper = BaseClientTestHelper(mock_zmq_context, wait_for_background_task)

    class StreamingDealerHelper:
        def __init__(self, base_helper):
            self._base = base_helper

        def setup_mock_socket(self, **kwargs):
            return self._base.setup_mock_socket(**kwargs)

        @asynccontextmanager
        async def create_client(
            self,
            address="tcp://127.0.0.1:5555",
            identity="worker-1",
            bind=False,
            auto_start=False,
            **mock_kwargs,
        ):
            client_kwargs = {"identity": identity}
            async with self._base.create_client(
                ZMQStreamingDealerClient,
                address=address,
                bind=bind,
                auto_start=auto_start,
                client_kwargs=client_kwargs,
                **mock_kwargs,
            ) as client:
                yield client

    return StreamingDealerHelper(helper)


# Shared test data and error scenarios
@pytest.fixture(
    params=[
        asyncio.CancelledError(),
        zmq.ContextTerminated(),
    ],
    ids=["cancelled_error", "context_terminated"],
)
def graceful_error(request):
    """Errors that should be handled gracefully without raising."""
    return request.param


@pytest.fixture(
    params=[
        RuntimeError("Test error"),
        ValueError("Invalid value"),
        Exception("Generic error"),
    ],
    ids=["runtime_error", "value_error", "generic_error"],
)
def non_graceful_error(request):
    """Errors that should raise CommunicationError."""
    return request.param


@pytest.fixture(
    params=[
        ("tcp://127.0.0.1:5555", True),
        ("tcp://127.0.0.1:5556", False),
        ("ipc:///tmp/test.ipc", True),
        ("ipc:///tmp/test.ipc", False),
    ],
    ids=["tcp_bind", "tcp_connect", "ipc_bind", "ipc_connect"],
)  # fmt: skip
def address_and_bind(request):
    """Common address and bind parameter combinations."""
    return request.param


@pytest.fixture
def create_callback_tracker():
    """Factory to create callback trackers for testing async callbacks."""

    def _create():
        """Create a new callback tracker."""
        event = asyncio.Event()
        received_messages = []

        async def callback(msg):
            """Track received messages and set event."""
            received_messages.append(msg)
            event.set()

        return callback, event, received_messages

    return _create


@pytest.fixture
def multiple_identities():
    """Common worker identities for testing."""
    return ["worker-1", "worker-2", "worker-3"]


@pytest.fixture(
    params=[
        "worker-1",
        "worker_2",
        "worker.3",
        "worker:4",
        "worker@host",
    ],
    ids=["dash", "underscore", "dot", "colon", "at-sign"],
)  # fmt: skip
def special_identity(request):
    """Various identity formats with special characters."""
    return request.param
