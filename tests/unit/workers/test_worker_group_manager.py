# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the worker-group state core."""

import time
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CommandType, WorkerStatus
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.common.pod_lifecycle_structs import GroupPeerCommandAck
from aiperf.workers.worker_group_manager import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)
from aiperf.workers.worker_pod_helpers import (
    configure_local_peers,
    shutdown_local_peers,
)

DEFAULT_MEMORY = 1024 * 1024 * 100


def _make_health(cpu_usage: float = 25.0) -> ProcessHealth:
    return ProcessHealth(
        create_time=time.time(),
        uptime=10.0,
        cpu_usage=cpu_usage,
        memory_usage=DEFAULT_MEMORY,
    )


def _make_worker_health_message(
    cpu_usage: float = 25.0,
    *,
    total: int = 4,
    completed: int = 3,
    failed: int = 0,
) -> WorkerHealthMessage:
    return WorkerHealthMessage(
        service_id="worker-0",
        health=_make_health(cpu_usage=cpu_usage),
        task_stats=WorkerTaskStats(
            total=total,
            completed=completed,
            failed=failed,
        ),
    )


def test_shared_worker_status_helpers_preserve_legacy_warning_and_summary_behavior() -> (
    None
):
    warnings: list[str] = []
    worker = WorkerStatusInfo(worker_id="worker-0")

    update_worker_status(
        worker,
        _make_worker_health_message(cpu_usage=97.0),
        warning=warnings.append,
    )

    assert worker.status == WorkerStatus.HIGH_LOAD
    assert warnings == [
        "CPU usage for worker-0 is 97%. AIPerf results may be inaccurate."
    ]
    assert build_worker_status_summary(
        service_id="group-manager",
        worker_infos={"worker-0": worker},
    ) == WorkerStatusSummaryMessage(
        service_id="group-manager",
        worker_statuses={"worker-0": WorkerStatus.HIGH_LOAD},
        worker_startup_states={},
    )


def test_mark_stale_workers_uses_shared_activity_window() -> None:
    worker = WorkerStatusInfo(worker_id="worker-0")
    worker.last_update_ns = 1

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("time.time_ns", lambda: int(1e12))
        mark_stale_workers({"worker-0": worker})

    assert worker.status == WorkerStatus.STALE



# =============================================================================
# Group-local lifecycle fanout (worker_pod_helpers)
# =============================================================================


class _StubRouter:
    """Minimal StreamingRouterClientProtocol stand-in for fanout tests."""

    def __init__(self, *, ack_for: set[str] | None = None) -> None:
        self.ack_for = ack_for
        self.requests: list[tuple[str, object]] = []

    async def request_to(self, identity: str, message, timeout: float):  # noqa: ARG002
        self.requests.append((identity, message))
        if self.ack_for is not None and identity not in self.ack_for:
            raise TimeoutError(f"no ack from {identity}")
        return GroupPeerCommandAck(cid=message.cid, service_id=identity)


@pytest.mark.asyncio
async def test_configure_local_peers_requests_an_ack_from_every_peer() -> None:
    router = _StubRouter()

    assert (
        await configure_local_peers(
            router=router,
            sender_service_id="wgm",
            peer_identities={"worker_0": "id-w0", "record_processor_0": "id-rp0"},
        )
        == []
    )

    assert sorted(identity for identity, _ in router.requests) == ["id-rp0", "id-w0"]
    assert all(
        message.command == str(CommandType.PROFILE_CONFIGURE)
        and message.service_id == "wgm"
        for _, message in router.requests
    )
    # Each command carries its own correlation id.
    assert len({message.cid for _, message in router.requests}) == 2


@pytest.mark.asyncio
async def test_configure_local_peers_is_a_noop_without_peers() -> None:
    router = _StubRouter()
    await configure_local_peers(
        router=router, sender_service_id="wgm", peer_identities={}
    )
    assert router.requests == []


@pytest.mark.asyncio
async def test_configure_local_peers_reports_a_peer_timeout() -> None:
    """The failure is returned, not raised.

    Raising out of the gather abandoned the sibling requests un-awaited and
    skipped the caller's _publish_worker_summary(), so a single wedged
    container in a 13-container pod took the pod's status report with it.
    """
    router = _StubRouter(ack_for=set())
    failures = await configure_local_peers(
        router=router,
        sender_service_id="wgm",
        peer_identities={"worker_0": "id-w0"},
    )
    assert [sid for sid, _ in failures] == ["worker_0"]
    assert isinstance(failures[0][1], TimeoutError)


@pytest.mark.asyncio
async def test_configure_local_peers_still_reaches_healthy_peers() -> None:
    """One dead peer must not stop the other twelve from being configured."""
    router = _StubRouter(ack_for={"id-w0", "id-rp0"})
    failures = await configure_local_peers(
        router=router,
        sender_service_id="wgm",
        peer_identities={
            "worker_0": "id-w0",
            "record_processor_0": "id-rp0",
            "worker_1": "id-w1",
        },
    )
    assert [sid for sid, _ in failures] == ["worker_1"]
    assert sorted(identity for identity, _ in router.requests) == [
        "id-rp0",
        "id-w0",
        "id-w1",
    ]


@pytest.mark.asyncio
async def test_shutdown_local_peers_warns_but_does_not_raise_on_peer_loss() -> None:
    """A peer that died must not stall teardown of the rest of the pod."""
    router = _StubRouter(ack_for={"id-w0"})
    logger = MagicMock()

    await shutdown_local_peers(
        router=router,
        sender_service_id="wgm",
        peer_identities={"worker_0": "id-w0", "worker_1": "id-w1"},
        command=CommandType.SHUTDOWN,
        logger=logger,
    )

    assert len(router.requests) == 2
    assert logger.warning.call_count == 1
    assert "worker_1" in str(logger.warning.call_args.args[0])
