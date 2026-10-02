# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The streaming DEALER/ROUTER protocols must declare what callers actually use.

Two ways a Protocol goes wrong without any type checker complaining:

* it omits a method the implementation has and callers depend on, so a
  conforming implementation silently loses a feature (``request``); and
* it declares a return type the implementation contradicts, so a conforming
  handler cannot do the one thing the protocol's own ``request_to`` needs it
  to do (``register_receiver`` returning a reply ``Struct``).

Both were live here. These tests pin the declared surface against the ZMQ
implementations that back it.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.pod_lifecycle_structs import GroupDatasetStateSnapshot
from aiperf.common.protocols import (
    StreamingDealerClientProtocol,
    StreamingRouterClientProtocol,
)
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


class TestDealerProtocolSurface:
    """``request`` is part of the contract, not an implementation extra."""

    def test_protocol_declares_request(self) -> None:
        assert "request" in vars(StreamingDealerClientProtocol)

    def test_declared_surface_is_a_subset_of_the_zmq_implementation(self) -> None:
        """Every member declared here must exist on the only implementation.

        Guards the reverse drift: a protocol member no implementation provides
        is just as broken as an implementation member no protocol declares.
        Scoped to members declared on this Protocol itself -- the inherited
        lifecycle surface is instance state, not class attributes.
        """
        declared = {
            name
            for name, value in vars(StreamingDealerClientProtocol).items()
            if not name.startswith("_") and callable(value)
        }
        assert declared == {"register_receiver", "send", "request"}
        missing = {
            attr for attr in declared if not hasattr(ZMQStreamingDealerClient, attr)
        }
        assert not missing

    def test_request_using_methods_are_covered_by_the_protocol(self) -> None:
        """``request`` is reached through the declared type in two places.

        ``Worker._query_pod_dataset_state`` and
        ``pod_lifecycle_structs._send_group_peer_hello_with_retry`` both hold a
        ``StreamingDealerClientProtocol`` and call ``.request``.
        """
        assert callable(ZMQStreamingDealerClient.request)


class TestRouterProtocolSurface:
    """A ROUTER receiver handler must be allowed to return a reply struct."""

    def test_register_receiver_permits_a_reply_struct(self) -> None:
        handler = StreamingRouterClientProtocol.register_receiver.__annotations__[
            "handler"
        ]
        # protocols.py is `from __future__ import annotations`, so this is the
        # source text of the declaration.
        assert "Struct | None" in handler
        assert "Coroutine[Any, Any, None]" not in handler

    def test_dealer_register_receiver_stays_fire_and_forget(self) -> None:
        """The DEALER side genuinely has no reply path; it must not copy this."""
        handler = StreamingDealerClientProtocol.register_receiver.__annotations__[
            "handler"
        ]
        assert "Coroutine[Any, Any, None]" in handler

    @pytest.mark.asyncio
    async def test_returned_struct_is_sent_back_to_the_originating_dealer(
        self, mock_zmq_context
    ) -> None:
        """The behavior the declared return type has to describe."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.send_to = AsyncMock()
        reply = GroupDatasetStateSnapshot(rid="abc", service_id="wgm-1", ready=False)
        client.register_receiver(AsyncMock(return_value=reply))

        await client._dispatch_message("worker-1", MagicMock())

        client.send_to.assert_awaited_once_with("worker-1", reply)

    @pytest.mark.asyncio
    async def test_returning_none_stays_fire_and_forget(self, mock_zmq_context) -> None:
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.send_to = AsyncMock()
        client.register_receiver(AsyncMock(return_value=None))

        await client._dispatch_message("worker-1", MagicMock())

        client.send_to.assert_not_awaited()


class TestDatasetStateRecoveryDoesNotFeatureDetect:
    """``_query_pod_dataset_state`` must trust the protocol it is typed against."""

    @pytest.mark.asyncio
    async def test_snapshot_is_returned_and_cached(self) -> None:
        from aiperf.workers.worker import Worker

        snapshot = GroupDatasetStateSnapshot(
            rid=uuid.uuid4().hex, service_id="wgm-1", ready=True
        )
        worker = MagicMock(spec=Worker)
        worker.service_id = "worker-7f2a"
        worker.pod_lifecycle_dealer_client = MagicMock()
        worker.pod_lifecycle_dealer_client.request = AsyncMock(return_value=snapshot)

        result = await Worker._query_pod_dataset_state(worker)

        assert result is snapshot
        assert worker._latest_pod_dataset_state is snapshot

    @pytest.mark.asyncio
    async def test_a_client_without_request_fails_loudly(self) -> None:
        """Regression: this used to return None and hang startup with no signal.

        A transport that cannot round-trip ``GroupDatasetStateQuery`` cannot
        recover a missed dataset broadcast at all. Degrading the recovery path
        to a silent no-op turned a protocol violation into an unexplained
        never-becomes-dispatchable worker, so it must raise instead.
        """
        from aiperf.workers.worker import Worker

        class _NoRequest:
            async def send(self, message) -> None: ...
            def register_receiver(self, handler) -> None: ...

        worker = MagicMock(spec=Worker)
        worker.service_id = "worker-7f2a"
        worker.pod_lifecycle_dealer_client = _NoRequest()

        with pytest.raises(AttributeError):
            await Worker._query_pod_dataset_state(worker)

    @pytest.mark.asyncio
    async def test_no_client_at_all_is_still_a_quiet_none(self) -> None:
        """Not being in group-managed mode is a supported state, not an error."""
        from aiperf.workers.worker import Worker

        worker = MagicMock(spec=Worker)
        worker.pod_lifecycle_dealer_client = None

        assert await Worker._query_pod_dataset_state(worker) is None
