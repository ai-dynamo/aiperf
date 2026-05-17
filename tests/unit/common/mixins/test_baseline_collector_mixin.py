# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import BaselineKind, ServiceCapability
from aiperf.common.messages import (
    PhaseBaselineAckMessage,
    PhaseBaselineRequestMessage,
)
from aiperf.common.mixins.baseline_collector_mixin import BaselineCollectorMixin


class _StubBus:
    def __init__(self) -> None:
        self.published: list[PhaseBaselineAckMessage] = []

    async def publish(self, msg: PhaseBaselineAckMessage) -> None:
        self.published.append(msg)


class _StubCollector(BaselineCollectorMixin):
    """Concrete subclass exercised in isolation (no BaseComponentService needed)."""

    def __init__(self, bus: _StubBus, *, fail: bool = False) -> None:
        self._bus = bus
        self.service_id = "svc-stub"
        self.calls: list[tuple[BaselineKind, str]] = []
        self._fail = fail

    async def publish(self, msg: PhaseBaselineAckMessage) -> None:
        await self._bus.publish(msg)

    async def collect_baseline(
        self, kind: BaselineKind, phase_id: str, phase_name: str
    ) -> None:
        if self._fail:
            raise RuntimeError("simulated DCGM failure")
        self.calls.append((kind, phase_name))


def test_extra_capabilities_includes_baseline_collector() -> None:
    assert ServiceCapability.BASELINE_COLLECTOR in _StubCollector.extra_capabilities


@pytest.mark.asyncio
async def test_handler_calls_collect_and_acks_success() -> None:
    bus = _StubBus()
    svc = _StubCollector(bus)
    await svc._on_phase_baseline_request(
        PhaseBaselineRequestMessage(
            phase_id="p1", phase_name="profiling", kind=BaselineKind.START
        )
    )
    assert svc.calls == [(BaselineKind.START, "profiling")]
    assert len(bus.published) == 1
    ack = bus.published[0]
    assert ack.success is True
    assert ack.service_id == "svc-stub"
    assert ack.phase_id == "p1"
    assert ack.kind == BaselineKind.START
    assert ack.error is None


@pytest.mark.asyncio
async def test_handler_acks_failure_when_collect_raises() -> None:
    bus = _StubBus()
    svc = _StubCollector(bus, fail=True)
    await svc._on_phase_baseline_request(
        PhaseBaselineRequestMessage(
            phase_id="p1", phase_name="profiling", kind=BaselineKind.END
        )
    )
    ack = bus.published[0]
    assert ack.success is False
    assert "simulated DCGM failure" in ack.error
