# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for streaming_dealer_client.py - ZMQStreamingDealerClient class.
"""

import asyncio

import msgspec.msgpack
import pytest
import zmq

from aiperf.common.enums import CreditPhase, LifecycleState
from aiperf.common.exceptions import NotInitializedError
from aiperf.credit.messages import RouterToWorkerMessage, WorkerReady
from aiperf.credit.structs import Credit
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient


class TestZMQStreamingDealerClientInitialization:
    """Test ZMQStreamingDealerClient initialization."""

    def test_init_creates_dealer_socket(self, mock_zmq_context):
        """Test that initialization creates a DEALER socket."""
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity="worker-1",
            bind=False,
        )

        assert client.socket_type == zmq.SocketType.DEALER
        assert client.identity == "worker-1"
        assert client._receiver_handler is None

    @pytest.mark.parametrize(
        "address,identity,bind",
        [
            ("tcp://127.0.0.1:5555", "worker-1", False),
            ("tcp://127.0.0.1:5556", "worker-2", True),
            ("ipc:///tmp/test.ipc", "worker-3", False),
            ("ipc:///tmp/test.ipc", "worker-4", True),
        ],
        ids=["tcp_connect", "tcp_bind", "ipc_connect", "ipc_bind"],
    )  # fmt: skip
    def test_init_with_various_addresses(
        self, address, identity, bind, mock_zmq_context
    ):
        """Test initialization with various address types."""
        client = ZMQStreamingDealerClient(
            address=address,
            identity=identity,
            bind=bind,
        )

        assert client.address == address
        assert client.identity == identity
        assert client.bind == bind

    def test_init_sets_identity_socket_option(self, mock_zmq_context):
        """Test that initialization sets IDENTITY socket option."""
        identity = "test-worker"
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity=identity,
            bind=False,
        )

        # Check that identity is in socket_ops
        assert zmq.IDENTITY in client.socket_ops
        assert client.socket_ops[zmq.IDENTITY] == identity.encode()

    def test_init_with_custom_socket_options(self, mock_zmq_context):
        """Test initialization with custom socket options."""
        identity = "test-worker"
        custom_ops = {zmq.IMMEDIATE: 1}
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity=identity,
            bind=False,
            socket_ops=custom_ops,
        )

        # Should have both identity and custom options
        assert zmq.IDENTITY in client.socket_ops
        assert zmq.IMMEDIATE in client.socket_ops

    def test_init_sets_client_id(self, mock_zmq_context):
        """Test that initialization sets client_id to identity."""
        identity = "test-worker"
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity=identity,
            bind=False,
        )

        assert client.client_id == identity


class TestZMQStreamingDealerClientRegisterReceiver:
    """Test ZMQStreamingDealerClient.register_receiver method."""

    @pytest.mark.asyncio
    async def test_register_receiver_succeeds(self, mock_zmq_context):
        """Test that register_receiver successfully registers a handler."""
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity="worker-1",
            bind=False,
        )

        async def handler(message: RouterToWorkerMessage) -> None:
            pass

        client.register_receiver(handler)

        assert client._receiver_handler == handler

    @pytest.mark.asyncio
    async def test_register_receiver_raises_when_already_registered(
        self, mock_zmq_context
    ):
        """Test that register_receiver raises ValueError if already registered."""
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity="worker-1",
            bind=False,
        )

        async def handler1(message: RouterToWorkerMessage) -> None:
            pass

        async def handler2(message: RouterToWorkerMessage) -> None:
            pass

        client.register_receiver(handler1)

        with pytest.raises(ValueError, match="already registered"):
            client.register_receiver(handler2)


class TestZMQStreamingDealerClientSend:
    """Test ZMQStreamingDealerClient.send method."""

    @pytest.mark.asyncio
    async def test_send_sends_struct(
        self, streaming_dealer_test_helper, sample_worker_ready
    ):
        """Test that send sends struct correctly using msgpack."""
        async with streaming_dealer_test_helper.create_client() as client:
            mock_socket = client.socket

            await client.send(sample_worker_ready)

            mock_socket.send.assert_called_once()
            sent_data = mock_socket.send.call_args[0][0]
            decoded = msgspec.msgpack.decode(sent_data, type=WorkerReady)
            assert decoded.worker_id == sample_worker_ready.worker_id

    @pytest.mark.asyncio
    async def test_send_multiple_structs(
        self, streaming_dealer_test_helper, sample_credit_return
    ):
        """Test sending multiple structs."""
        async with streaming_dealer_test_helper.create_client() as client:
            mock_socket = client.socket
            structs = [WorkerReady(worker_id=f"worker-{i}") for i in range(3)]

            for struct in structs:
                await client.send(struct)

            assert mock_socket.send.call_count == len(structs)

    @pytest.mark.asyncio
    async def test_send_raises_when_not_initialized(
        self, streaming_dealer_test_helper, sample_worker_ready
    ):
        """Test that send raises NotInitializedError when not initialized."""
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity="worker-1",
            bind=False,
        )
        client.socket = None

        with pytest.raises(NotInitializedError, match="Socket not initialized"):
            await client.send(sample_worker_ready)

    @pytest.mark.asyncio
    async def test_send_handles_send_failure(
        self, streaming_dealer_test_helper, sample_worker_ready
    ):
        """Test that send handles send failures."""
        async with streaming_dealer_test_helper.create_client(
            send_side_effect=Exception("Send failed")
        ) as client:
            with pytest.raises(Exception, match="Send failed"):
                await client.send(sample_worker_ready)


class TestZMQStreamingDealerClientReceiver:
    """Test ZMQStreamingDealerClient receiver background task."""

    @pytest.mark.asyncio
    async def test_receiver_task_starts_on_start(self, streaming_dealer_test_helper):
        """Test that receiver task starts when client starts."""
        async with streaming_dealer_test_helper.create_client(
            auto_start=True
        ) as client:
            assert client.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_receiver_calls_handler_on_credit(
        self, streaming_dealer_test_helper, sample_credit, create_callback_tracker
    ):
        """Test that receiver calls handler when credit arrives."""
        callback, event, received = create_callback_tracker()

        async def test_handler(message: RouterToWorkerMessage) -> None:
            await callback(message)

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msgspec.msgpack.encode(sample_credit)
            await asyncio.Future()  # Block forever after first call

        streaming_dealer_test_helper.setup_mock_socket(recv_side_effect=mock_recv)

        async with streaming_dealer_test_helper.create_client() as client:
            # Register handler BEFORE starting to avoid race condition
            client.register_receiver(test_handler)
            await client.start()

            await asyncio.wait_for(event.wait(), timeout=1.0)
            assert len(received) == 1
            recv_credit = received[0]
            assert isinstance(recv_credit, Credit)
            assert recv_credit.id == sample_credit.id
            assert recv_credit.conversation_id == sample_credit.conversation_id

    @pytest.mark.asyncio
    async def test_receiver_warns_when_no_handler_registered(
        self, streaming_dealer_test_helper, sample_credit, wait_for_background_task
    ):
        """Test that receiver logs warning when no handler is registered."""
        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msgspec.msgpack.encode(sample_credit)
            await asyncio.Future()  # Block forever after first call

        streaming_dealer_test_helper.setup_mock_socket(recv_side_effect=mock_recv)

        async with streaming_dealer_test_helper.create_client(auto_start=True):
            # Don't register handler
            await wait_for_background_task(iterations=5)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception,iterations",
        [
            (zmq.Again(), 3),
            (RuntimeError("Test error"), 3),
        ],
        ids=["zmq_again", "generic_error"],
    )  # fmt: skip
    async def test_receiver_handles_exceptions(
        self,
        streaming_dealer_test_helper,
        exception,
        iterations,
        time_traveler,
    ):
        """Test that receiver handles exceptions gracefully."""
        call_count = 0
        done_event = asyncio.Event()

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count < iterations:
                raise exception
            done_event.set()
            await asyncio.Future()  # Block forever after

        streaming_dealer_test_helper.setup_mock_socket(recv_side_effect=mock_recv)

        async with streaming_dealer_test_helper.create_client(auto_start=True):
            await asyncio.wait_for(done_event.wait(), timeout=1.0)
            assert call_count >= iterations

    @pytest.mark.asyncio
    async def test_receiver_stops_on_cancelled_error(
        self, streaming_dealer_test_helper, wait_for_background_task
    ):
        """Test that receiver stops gracefully on CancelledError."""
        streaming_dealer_test_helper.setup_mock_socket(
            recv_side_effect=asyncio.CancelledError()
        )

        async with streaming_dealer_test_helper.create_client(
            auto_start=True
        ) as client:
            await wait_for_background_task()
            # The receiver task should exit gracefully without raising an unhandled exception
            # Client remains in RUNNING state until explicitly stopped
            assert client.state == LifecycleState.RUNNING


class TestZMQStreamingDealerClientLifecycle:
    """Test ZMQStreamingDealerClient lifecycle management."""

    @pytest.mark.asyncio
    async def test_clear_receiver_on_stop(self, streaming_dealer_test_helper):
        """Test that receiver handler is cleared on stop."""
        async with streaming_dealer_test_helper.create_client() as client:

            async def handler(message: RouterToWorkerMessage) -> None:
                pass

            client.register_receiver(handler)
            assert client._receiver_handler == handler

        # After context exits (which calls stop), handler should be cleared
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self, streaming_dealer_test_helper, wait_for_background_task
    ):
        """Test full client lifecycle: initialize -> start -> stop."""
        async with streaming_dealer_test_helper.create_client() as client:

            async def handler(message: RouterToWorkerMessage) -> None:
                pass

            client.register_receiver(handler)
            assert client.state == LifecycleState.INITIALIZED

            await client.start()
            await wait_for_background_task()
            assert client.state == LifecycleState.RUNNING

        # Context exit calls stop
        assert client.state == LifecycleState.STOPPED
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_send_after_stop_raises(
        self, streaming_dealer_test_helper, sample_worker_ready
    ):
        """Test that send raises after client is stopped."""
        async with streaming_dealer_test_helper.create_client() as client:
            pass

        # Client is now stopped after context exit
        with pytest.raises(asyncio.CancelledError, match="Socket was stopped"):
            await client.send(sample_worker_ready)


class TestZMQStreamingDealerClientEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sends(
        self, streaming_dealer_test_helper, sample_worker_ready
    ):
        """Test multiple concurrent sends."""
        async with streaming_dealer_test_helper.create_client() as client:
            mock_socket = client.socket
            num_structs = 5

            await asyncio.gather(
                *[client.send(sample_worker_ready) for _ in range(num_structs)]
            )

            assert mock_socket.send.call_count == num_structs

    @pytest.mark.asyncio
    async def test_different_struct_types(
        self, streaming_dealer_test_helper, sample_worker_ready, sample_credit_return
    ):
        """Test sending different struct types."""
        async with streaming_dealer_test_helper.create_client() as client:
            mock_socket = client.socket
            structs = [sample_worker_ready, sample_credit_return]

            for struct in structs:
                await client.send(struct)

            assert mock_socket.send.call_count == len(structs)

    @pytest.mark.asyncio
    async def test_receiver_with_multiple_credits(
        self, streaming_dealer_test_helper, sample_credit
    ):
        """Test receiver processing multiple credits."""
        credits = [
            Credit(
                id=i,
                phase=CreditPhase.PROFILING,
                conversation_id=f"conv-{i:03d}",
                x_correlation_id=f"corr-{i:03d}",
                turn_index=0,
                num_turns=1,
                issued_at_ns=1000000000 + i,
            )
            for i in range(3)
        ]

        credit_index = 0
        received = []
        received_event = asyncio.Event()

        async def mock_recv():
            nonlocal credit_index
            if credit_index < len(credits):
                result = msgspec.msgpack.encode(credits[credit_index])
                credit_index += 1
                return result
            await asyncio.Future()  # Block forever after all credits

        streaming_dealer_test_helper.setup_mock_socket(recv_side_effect=mock_recv)

        async def test_handler(message: RouterToWorkerMessage) -> None:
            received.append(message)
            if len(received) == len(credits):
                received_event.set()

        async with streaming_dealer_test_helper.create_client() as client:
            client.register_receiver(test_handler)
            await client.start()

            await asyncio.wait_for(received_event.wait(), timeout=2.0)

            assert len(received) == len(credits)
            for i, credit in enumerate(received):
                assert isinstance(credit, Credit)
                assert credit.id == i

    @pytest.mark.parametrize(
        "identity",
        ["worker-1", "worker_2", "worker.3", "worker:4", "worker@host"],
        ids=["dash", "underscore", "dot", "colon", "at-sign"],
    )  # fmt: skip
    def test_identity_with_special_characters(self, mock_zmq_context, identity):
        """Test creating client with various identity formats."""
        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555",
            identity=identity,
            bind=False,
        )
        assert client.identity == identity
        assert client.socket_ops[zmq.IDENTITY] == identity.encode()

    @pytest.mark.asyncio
    async def test_bind_mode(self, mock_zmq_socket, mock_zmq_context):
        """Test DEALER client in bind mode (unusual but supported)."""
        client = ZMQStreamingDealerClient(
            address="tcp://*:5555",
            identity="worker-1",
            bind=True,  # Bind instead of connect
        )
        await client.initialize()

        # Should bind, not connect
        mock_zmq_socket.bind.assert_called_once_with("tcp://*:5555")
        assert not mock_zmq_socket.connect.called

        await client.stop()
