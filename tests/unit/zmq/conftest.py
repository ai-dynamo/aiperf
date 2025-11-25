# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures and utilities for ZMQ testing.

This module provides reusable fixtures, mocks, and helpers for testing ZMQ functionality.
"""

import asyncio
import itertools
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import zmq.asyncio

from aiperf.common.enums import LifecycleState
from aiperf.common.messages import HeartbeatMessage
from aiperf.common.utils import yield_to_event_loop


@pytest.fixture
def mock_zmq_socket():
    """Create a mock ZMQ socket with common methods.

    Mocks the methods actually used by ZMQ clients:
    - send() / recv() - used by dealer, push, pull clients
    - send_multipart() / recv_multipart() - used by pub, sub, router clients

    By default, recv methods block forever (await on a never-completing Future)
    to avoid busy loops with mocked sleep. Tests should override these when
    they need specific return values.
    """

    async def _block_forever():
        """Block forever by awaiting a Future that never completes."""
        await asyncio.Future()  # Never resolves

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
    return socket


@pytest.fixture
def mock_zmq_context(mock_zmq_socket):
    """Create a mock ZMQ context that returns mock sockets."""
    context = MagicMock(spec=zmq.asyncio.Context)
    context.socket = Mock(return_value=mock_zmq_socket)
    context.term = Mock()
    return context


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
        """Wait for a few event loop iterations to let background tasks run."""
        for _ in range(iterations):
            await yield_to_event_loop()

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
            mock_socket.recv_multipart = AsyncMock(
                side_effect=recv_multipart_side_effect
            )
        else:
            # Default to blocking forever to prevent busy loop
            mock_socket.recv_multipart = AsyncMock(side_effect=_block_forever)

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
