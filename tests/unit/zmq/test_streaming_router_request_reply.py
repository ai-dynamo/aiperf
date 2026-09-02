# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-reply and configurable-decode tests for ZMQStreamingRouterClient.

Covers the surface the worker-pod lifecycle channel depends on:
``decode_type``, ``request_to`` correlation by ``cid``, the timeout path, and
reply-on-handler-return.
"""

import asyncio
from unittest.mock import AsyncMock

import msgspec.msgpack
import pytest

from aiperf.common.pod_lifecycle_structs import (
    GroupPeerAck,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupPeerHello,
    PeerToGroupManagerMessage,
)
from aiperf.credit.messages import WorkerToRouterMessage
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


def _make_client(mock_zmq_context, decode_type=PeerToGroupManagerMessage):
    return ZMQStreamingRouterClient(
        address="tcp://*:5555", bind=True, decode_type=decode_type
    )


class TestDecodeType:
    """The decoder is per-instance and defaults to the credit-plane union."""

    def test_default_decode_type_is_worker_to_router(self, mock_zmq_context) -> None:
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        assert client._decoder.type is WorkerToRouterMessage

    def test_explicit_decode_type_is_used(self, mock_zmq_context) -> None:
        client = _make_client(mock_zmq_context)
        assert client._decoder.type is PeerToGroupManagerMessage

    def test_two_clients_keep_independent_decoders(self, mock_zmq_context) -> None:
        """Regression guard: the decoder used to be a shared module global."""
        pod = _make_client(mock_zmq_context)
        credit = ZMQStreamingRouterClient(address="tcp://*:5556", bind=True)
        assert pod._decoder is not credit._decoder
        assert pod._decoder.type is not credit._decoder.type


class TestRequestTo:
    """``request_to`` pairs a request with its reply via ``cid``."""

    @pytest.mark.asyncio
    async def test_request_to_returns_reply_matched_by_cid(
        self, mock_zmq_context
    ) -> None:
        client = _make_client(mock_zmq_context)
        ack = GroupPeerCommandAck(cid="cid-1", service_id="worker_0")

        async def deliver_reply(identity: str, struct) -> None:
            # Stand in for the peer answering over the FD-reader drain path.
            client._dispatch_router((identity, ack))

        client.send_to = AsyncMock(side_effect=deliver_reply)

        response = await client.request_to(
            "worker_0",
            GroupPeerCommand(
                cid="cid-1", service_id="wgm", command="profile_configure"
            ),
            timeout=1.0,
        )

        assert response is ack
        # The future is cleaned up whether it resolved or not.
        assert client._pending_requests == {}

    @pytest.mark.asyncio
    async def test_request_to_times_out_when_no_reply_arrives(
        self, mock_zmq_context
    ) -> None:
        client = _make_client(mock_zmq_context)
        client.send_to = AsyncMock()

        with pytest.raises(TimeoutError):
            await client.request_to(
                "worker_0",
                GroupPeerCommand(cid="cid-1", service_id="wgm", command="shutdown"),
                timeout=0.01,
            )

        assert client._pending_requests == {}

    @pytest.mark.asyncio
    async def test_reply_with_mismatched_cid_does_not_resolve_request(
        self, mock_zmq_context
    ) -> None:
        """A stray ack for a different command must not satisfy this caller."""
        client = _make_client(mock_zmq_context)
        client.register_receiver(AsyncMock(return_value=None))

        async def deliver_wrong_reply(identity: str, struct) -> None:
            client._dispatch_router(
                (identity, GroupPeerCommandAck(cid="other", service_id="worker_1"))
            )

        client.send_to = AsyncMock(side_effect=deliver_wrong_reply)

        with pytest.raises(TimeoutError):
            await client.request_to(
                "worker_0",
                GroupPeerCommand(cid="cid-1", service_id="wgm", command="shutdown"),
                timeout=0.01,
            )

    @pytest.mark.asyncio
    async def test_request_to_rejects_struct_without_cid(
        self, mock_zmq_context
    ) -> None:
        client = _make_client(mock_zmq_context)
        with pytest.raises(ValueError, match="requires a struct with a 'cid'"):
            await client.request_to(
                "worker_0",
                GroupPeerHello(rid="r1", service_id="worker_0", service_type="worker"),
                timeout=1.0,
            )

    @pytest.mark.asyncio
    async def test_request_to_rejects_empty_cid_instead_of_hanging(
        self, mock_zmq_context
    ) -> None:
        """Both guards must use ``not cid``, not ``cid is None``.

        An empty-string cid passes an ``is None`` check but can never satisfy
        _try_resolve_pending_request's ``not cid`` guard, so the request would
        register a future nothing could ever resolve and hang until timeout.
        """
        client = _make_client(mock_zmq_context)
        client.send_to = AsyncMock()

        with pytest.raises(ValueError, match="requires a struct with a 'cid'"):
            await client.request_to(
                "worker_0",
                GroupPeerCommand(cid="", service_id="wgm", command="shutdown"),
                timeout=1.0,
            )

        client.send_to.assert_not_awaited()
        assert client._pending_requests == {}

    @pytest.mark.asyncio
    async def test_concurrent_requests_get_their_own_replies_out_of_order(
        self, mock_zmq_context
    ) -> None:
        """Two in-flight requests to different peers must not cross-deliver."""
        client = _make_client(mock_zmq_context)
        gate = asyncio.Event()

        async def hold_until_both_sent(identity: str, struct) -> None:  # noqa: ARG001
            # Return without replying, so both futures are pending at once.
            gate.set()

        client.send_to = AsyncMock(side_effect=hold_until_both_sent)

        first = asyncio.create_task(
            client.request_to(
                "id-w0",
                GroupPeerCommand(cid="cid-a", service_id="wgm", command="configure"),
                timeout=1.0,
            )
        )
        second = asyncio.create_task(
            client.request_to(
                "id-w1",
                GroupPeerCommand(cid="cid-b", service_id="wgm", command="configure"),
                timeout=1.0,
            )
        )
        await gate.wait()
        while len(client._pending_requests) < 2:
            await asyncio.sleep(0)

        # Replies arrive from the opposite peers, in reverse order.
        ack_b = GroupPeerCommandAck(cid="cid-b", service_id="worker_1")
        ack_a = GroupPeerCommandAck(cid="cid-a", service_id="worker_0")
        client._dispatch_router(("id-w1", ack_b))
        client._dispatch_router(("id-w0", ack_a))

        assert await first is ack_a
        assert await second is ack_b
        assert client._pending_requests == {}

    @pytest.mark.asyncio
    async def test_pending_replies_bypass_the_receiver_handler(
        self, mock_zmq_context
    ) -> None:
        """A correlated reply belongs to its awaiter, not the general handler."""
        client = _make_client(mock_zmq_context)
        handler = AsyncMock(return_value=None)
        client.register_receiver(handler)
        ack = GroupPeerCommandAck(cid="cid-1", service_id="worker_0")

        async def deliver_reply(identity: str, struct) -> None:
            client._dispatch_router((identity, ack))

        client.send_to = AsyncMock(side_effect=deliver_reply)

        await client.request_to(
            "worker_0",
            GroupPeerCommand(cid="cid-1", service_id="wgm", command="shutdown"),
            timeout=1.0,
        )

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_requests(self, mock_zmq_context) -> None:
        client = _make_client(mock_zmq_context)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        client._pending_requests["cid-1"] = future

        await client._clear_receiver()

        assert future.cancelled()
        assert client._pending_requests == {}


class TestReplyOnHandlerReturn:
    """A handler returning a Struct turns the exchange into request-reply."""

    @pytest.mark.asyncio
    async def test_handler_return_value_is_sent_back_to_sender(
        self, streaming_router_test_helper
    ) -> None:
        hello = GroupPeerHello(rid="r1", service_id="worker_0", service_type="worker")
        ack = GroupPeerAck(rid="r1", service_id="wgm")

        streaming_router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[
                [b"worker-identity", msgspec.msgpack.encode(hello)]
            ]
        )

        async with streaming_router_test_helper.create_client(
            client_kwargs={"decode_type": PeerToGroupManagerMessage}
        ) as client:
            sent: list[tuple[str, object]] = []
            client.send_to = AsyncMock(
                side_effect=lambda i, m: sent.append((i, m))  # noqa: ARG005
            )
            client.register_receiver(AsyncMock(return_value=ack))
            await client.start()

            for _ in range(50):
                if sent:
                    break
                await asyncio.sleep(0)

        assert sent == [("worker-identity", ack)]

    @pytest.mark.asyncio
    async def test_handler_exception_is_contained_not_propagated(
        self, mock_zmq_context
    ) -> None:
        """One raising handler must not tear down the shared dispatch loop."""
        client = _make_client(mock_zmq_context)
        client.register_receiver(AsyncMock(side_effect=RuntimeError("boom")))

        # Awaiting directly asserts the coroutine itself swallows the exception;
        # _dispatch_router fires it through execute_async in production.
        await client._dispatch_message(
            "worker-identity",
            GroupPeerHello(rid="r1", service_id="worker_0", service_type="worker"),
        )
