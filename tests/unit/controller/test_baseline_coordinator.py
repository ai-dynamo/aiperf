# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from aiperf.common.enums import BaselineKind
from aiperf.common.messages import (
    PhaseBaselineAckMessage,
    PhaseBaselineRequestMessage,
)
from aiperf.controller.baseline_coordinator import BaselineCoordinator


class _Bus:
    def __init__(self) -> None:
        self.published: list[PhaseBaselineRequestMessage] = []

    async def publish(self, msg: PhaseBaselineRequestMessage) -> None:
        self.published.append(msg)


@pytest.fixture
def bus() -> _Bus:
    return _Bus()


@pytest.fixture
def coord(bus: _Bus) -> BaselineCoordinator:
    return BaselineCoordinator(publish=bus.publish, gate_timeout_s=0.05)


@pytest.mark.asyncio
async def test_empty_registered_returns_immediately(
    coord: BaselineCoordinator, bus: _Bus
) -> None:
    await coord.gate_phase("p1", "warmup", BaselineKind.START)
    assert bus.published == []


@pytest.mark.asyncio
async def test_happy_path_acks_release_gate(
    coord: BaselineCoordinator, bus: _Bus
) -> None:
    coord.register("svc-a")
    coord.register("svc-b")

    async def _drive() -> None:
        await asyncio.sleep(0)  # let gate publish first
        coord.handle_ack(
            PhaseBaselineAckMessage(
                service_id="svc-a", phase_id="p1", kind=BaselineKind.START, success=True
            )
        )
        coord.handle_ack(
            PhaseBaselineAckMessage(
                service_id="svc-b", phase_id="p1", kind=BaselineKind.START, success=True
            )
        )

    await asyncio.gather(coord.gate_phase("p1", "warmup", BaselineKind.START), _drive())
    assert len(bus.published) == 1
    assert bus.published[0].kind == BaselineKind.START


@pytest.mark.asyncio
async def test_timeout_with_unacked_logs_and_releases(
    coord: BaselineCoordinator, bus: _Bus, caplog: pytest.LogCaptureFixture
) -> None:
    coord.register("slow-svc")
    await coord.gate_phase("p1", "profiling", BaselineKind.START)
    assert len(bus.published) == 1
    assert any(
        "slow-svc" in rec.getMessage() and "timed out" in rec.getMessage()
        for rec in caplog.records
    )
    assert any(
        "AIPERF_BASELINE_GATE_TIMEOUT_S" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_error_ack_counts_as_ack(
    coord: BaselineCoordinator, bus: _Bus, caplog: pytest.LogCaptureFixture
) -> None:
    coord.register("svc-a")

    async def _drive() -> None:
        await asyncio.sleep(0)
        coord.handle_ack(
            PhaseBaselineAckMessage(
                service_id="svc-a",
                phase_id="p1",
                kind=BaselineKind.START,
                success=False,
                error="DCGM down",
            )
        )

    await asyncio.gather(coord.gate_phase("p1", "x", BaselineKind.START), _drive())
    assert any("DCGM down" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_late_ack_after_timeout_dropped_silently(
    coord: BaselineCoordinator,
) -> None:
    coord.register("svc-a")
    await coord.gate_phase("p1", "x", BaselineKind.START)  # times out, no acks
    coord.handle_ack(
        PhaseBaselineAckMessage(
            service_id="svc-a", phase_id="p1", kind=BaselineKind.START, success=True
        )
    )


def test_re_registration_idempotent(coord: BaselineCoordinator) -> None:
    coord.register("svc-a")
    coord.register("svc-a")
    assert coord.registered_count == 1


def test_unregister_removes(coord: BaselineCoordinator) -> None:
    coord.register("svc-a")
    coord.unregister("svc-a")
    assert coord.registered_count == 0
