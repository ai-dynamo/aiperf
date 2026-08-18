# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record processors must answer the WorkerGroupManager's peer commands.

The WGM fans PROFILE_CONFIGURE and SHUTDOWN out to every registered
group-local peer and blocks on a GroupPeerCommandAck from each, bounded by
PROFILE_CONFIGURE_TIMEOUT (600 s). The record processor opened the dealer with
decode_type=GroupManagerToPeerMessage but never registered a receiver, so
nothing ever answered: every pod stalled 600 s at configure and again at
shutdown.

ABORT is the case that matters most: a clean stop() exits 0 and leaves the pod
Ready with no processors, so the container must hard-exit for kubelet to
restart it alongside the WGM.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CommandType
from aiperf.common.messages import FinalizeArtifactsCommand
from aiperf.common.pod_lifecycle_structs import GroupPeerCommand, GroupPeerCommandAck
from aiperf.records.record_processor_service import RecordProcessor


@pytest.fixture
def rp():
    svc = RecordProcessor.__new__(RecordProcessor)
    svc.pod_lifecycle_dealer_client = AsyncMock()
    svc.service_id = "record_processor_0"
    svc.stop = AsyncMock()
    svc.inference_result_parser = MagicMock(configure=AsyncMock())
    svc._children = []
    svc.error = MagicMock()
    svc.warning = MagicMock()
    svc.debug = MagicMock()
    return svc


def _cmd(command: CommandType) -> GroupPeerCommand:
    return GroupPeerCommand(cid="c-1", service_id="wgm", command=str(command))


class TestPeerCommandHandling:
    @pytest.mark.asyncio
    async def test_profile_configure_is_acked(self, rp):
        await rp._handle_pod_peer_command(_cmd(CommandType.PROFILE_CONFIGURE))

        sent = rp.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(sent, GroupPeerCommandAck)
        assert sent.cid == "c-1"
        assert sent.service_id == "record_processor_0"

    @pytest.mark.asyncio
    async def test_finalize_artifacts_flushes_every_writer_before_ack(self, rp):
        first = MagicMock(flush_buffer=AsyncMock())
        second = MagicMock(flush_buffer=AsyncMock())
        rp._children = [first, second]

        await rp._handle_pod_peer_command(_cmd(CommandType.FINALIZE_ARTIFACTS))

        first.flush_buffer.assert_awaited_once()
        second.flush_buffer.assert_awaited_once()
        assert isinstance(
            rp.pod_lifecycle_dealer_client.send.await_args.args[0], GroupPeerCommandAck
        )

    @pytest.mark.asyncio
    async def test_finalize_artifacts_failure_is_not_acked(self, rp):
        rp._children = [
            MagicMock(flush_buffer=AsyncMock(side_effect=OSError("disk full")))
        ]

        with pytest.raises(ExceptionGroup, match="Failed to finalize"):
            await rp._handle_pod_peer_command(_cmd(CommandType.FINALIZE_ARTIFACTS))

        rp.pod_lifecycle_dealer_client.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_global_finalize_command_flushes_artifacts(self, rp):
        rp._finalize_local_artifacts = AsyncMock()

        await rp._finalize_artifacts_command(
            FinalizeArtifactsCommand(service_id="records_manager")
        )

        rp._finalize_local_artifacts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_stops_and_acks(self, rp):
        await rp._handle_pod_peer_command(_cmd(CommandType.SHUTDOWN))

        rp.stop.assert_awaited_once()
        assert isinstance(
            rp.pod_lifecycle_dealer_client.send.await_args.args[0], GroupPeerCommandAck
        )

    @pytest.mark.asyncio
    async def test_shutdown_acks_before_stopping(self, rp):
        """stop() closes the comms children, dealer socket included.

        An ack sent after stop() never reaches the WGM, which then blocks the
        full PROFILE_CONFIGURE_TIMEOUT waiting for it.
        """
        order: list[str] = []
        rp.pod_lifecycle_dealer_client.send.side_effect = lambda *a, **k: order.append(
            "ack"
        )
        rp.stop.side_effect = lambda *a, **k: order.append("stop")

        await rp._handle_pod_peer_command(_cmd(CommandType.SHUTDOWN))

        assert order == ["ack", "stop"]

    @pytest.mark.asyncio
    async def test_shutdown_stops_even_if_the_ack_fails(self, rp):
        rp.pod_lifecycle_dealer_client.send.side_effect = RuntimeError("peer gone")

        await rp._handle_pod_peer_command(_cmd(CommandType.SHUTDOWN))

        rp.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_abort_hard_exits_after_best_effort_ack(self, rp, monkeypatch):
        """A clean stop() would exit 0 and leave the pod Ready with no workers."""
        exits: list[int] = []
        monkeypatch.setattr(
            "aiperf.records.record_processor_service.os._exit",
            lambda code: exits.append(code),
        )

        await rp._handle_pod_peer_command(_cmd(CommandType.ABORT))

        assert exits == [1], "ABORT did not hard-exit"
        rp.stop.assert_not_awaited()
        assert isinstance(
            rp.pod_lifecycle_dealer_client.send.await_args.args[0], GroupPeerCommandAck
        )

    @pytest.mark.asyncio
    async def test_abort_exits_even_if_the_ack_fails(self, rp, monkeypatch):
        """The WGM is already leaving; delivery is not required."""
        exits: list[int] = []
        monkeypatch.setattr(
            "aiperf.records.record_processor_service.os._exit",
            lambda code: exits.append(code),
        )
        rp.pod_lifecycle_dealer_client.send.side_effect = RuntimeError("peer gone")

        with contextlib.suppress(RuntimeError):
            await rp._handle_pod_peer_command(_cmd(CommandType.ABORT))

        assert exits == [1]

    @pytest.mark.asyncio
    async def test_unknown_command_is_not_acked(self, rp):
        """An ack would tell the WGM the command succeeded."""
        await rp._handle_pod_peer_command(
            GroupPeerCommand(cid="c-9", service_id="wgm", command="teleport")
        )
        rp.pod_lifecycle_dealer_client.send.assert_not_awaited()
        rp.warning.assert_called()

    @pytest.mark.asyncio
    async def test_receiver_is_registered_for_the_dealer(self):
        """Without register_receiver the handler is never reached at all."""
        import inspect

        src = inspect.getsource(RecordProcessor)
        assert "register_receiver" in src
