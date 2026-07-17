# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lane-level concurrency ramp on the graph replay path.

``--concurrency-ramp-duration`` cannot throttle graph lanes through session
slots (graph credits bypass them), so the strategy exposes
``LaneSettableProtocol.set_lane_limit`` and the runner's concurrency ramper
drives it: lanes above the live limit PARK before their first instance and
are admitted as the ramper raises the limit 1 -> ``--concurrency``.

Component-level: fake ``CreditIssuer`` echoing returns, no worker/ZMQ. The
decisive proofs: parked lanes dispatch NOTHING until admitted; raising the
limit releases exactly the newly-admitted lanes; the phase completes once the
limit reaches the target (the Ramper always reaches its target, and
duration-cancel cancels parked waiters cleanly through the TaskGroup).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.strategies.core import LaneSettableProtocol

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_MIN = _FIX_DIR / "weka_min.json"


@dataclass
class _EchoIssuer:
    """Fake CreditIssuer echoing each issued credit to the return observer."""

    observer: Any = None
    issued: int = 0
    returned: int = 0
    sending_complete_calls: int = 0
    sent: list[Any] = field(default_factory=list)

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued += 1
        self.sent.append(turn)
        asyncio.get_running_loop().call_soon(self._echo, turn)
        return True

    def _echo(self, turn: Any) -> None:
        self.returned += 1
        if self.observer is not None:
            self.observer(turn, None, False)

    def mark_graph_sending_complete(self) -> None:
        self.sending_complete_calls += 1

    def graph_all_returned(self) -> bool:
        return self.returned >= self.issued

    def set_graph_all_returned_event(self) -> None:
        return None


class _PhaseCfg:
    def __init__(self, *, concurrency: int | None = None) -> None:
        self.phase = CreditPhase.PROFILING
        self.concurrency = concurrency
        self.expected_num_sessions = None
        self.total_expected_requests = None
        self.expected_duration_sec = None
        self.num_dataset_entries = None
        self.max_context_length = None


def _corpus(n: int):
    """``n`` distinct-id clones of the gap-free ``weka_min`` template."""
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    base = from_weka_trace(str(_MIN))
    graph = base.graph
    zeroed = [
        msgspec.structs.replace(
            e,
            **{
                f: 0.0
                for f in ("delay_after_predecessor_us", "min_start_delay_us")
                if getattr(e, f, None) is not None
            },
        )
        for e in graph.edges
    ]
    base = msgspec.structs.replace(
        base, graph=msgspec.structs.replace(graph, edges=zeroed)
    )
    t0 = base.traces[0]
    clones = [t0]
    clones.extend(msgspec.structs.replace(t0, id=f"{t0.id}#{i}") for i in range(1, n))
    return msgspec.structs.replace(base, traces=clones)


def _make_strategy(parsed, issuer: _EchoIssuer, **overrides):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    overrides.setdefault("start_min_ratio", 0.0)
    overrides.setdefault("start_max_ratio", 0.0)
    return GraphIRReplayStrategy(
        config=overrides.pop("config", None),
        conversation_source=None,
        scheduler=None,
        stop_checker=None,
        credit_issuer=issuer,
        lifecycle=overrides.pop("lifecycle", None),
        parsed_graph=parsed,
        register_observer=lambda obs: setattr(issuer, "observer", obs),
        **overrides,
    )


async def test_strategy_satisfies_lane_settable_protocol():
    """The runner's ramper fan-out gates on this isinstance check."""
    strategy = _make_strategy(_corpus(1), _EchoIssuer(), max_concurrent_traces=2)
    assert isinstance(strategy, LaneSettableProtocol)


async def test_lane_admission_mechanics():
    """Admitted lanes pass immediately; parked lanes release on a raise;
    the limit clamps to [1, concurrency]."""
    strategy = _make_strategy(_corpus(4), _EchoIssuer(), max_concurrent_traces=4)

    strategy.set_lane_limit(2)
    # Lanes 0 and 1 are admitted: the waits complete without a raise.
    await asyncio.wait_for(strategy._wait_for_lane_admission(0), timeout=1.0)
    await asyncio.wait_for(strategy._wait_for_lane_admission(1), timeout=1.0)

    parked = asyncio.create_task(strategy._wait_for_lane_admission(3))
    await asyncio.sleep(0)
    assert not parked.done(), "lane 3 must park while the limit is 2"

    strategy.set_lane_limit(3)
    await asyncio.sleep(0)
    assert not parked.done(), "lane 3 must stay parked at limit 3"

    strategy.set_lane_limit(4)
    await asyncio.wait_for(parked, timeout=1.0)

    # Clamps: never below 1, never above the resolved concurrency.
    strategy.set_lane_limit(0)
    assert strategy._lane_limit == 1
    strategy.set_lane_limit(99)
    assert strategy._lane_limit == 4


async def test_parked_lanes_dispatch_nothing_until_admitted():
    """E2E through execute_phase: with the limit held at 1, only lane 0's
    instance dispatches; raising the limit to the target releases the parked
    lanes and the phase completes covering the whole corpus."""
    parsed = _corpus(3)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_PhaseCfg(concurrency=3),
        max_concurrent_traces=3,
        allow_dataset_wrap=False,
    )
    # Simulate the ramper's initial setter(1), which lands before execution.
    strategy.set_lane_limit(1)

    await strategy.setup_phase()
    run = asyncio.create_task(strategy.execute_phase())

    # Let lane 0 finish its single-pass instance; lanes 1-2 stay parked.
    for _ in range(200):
        await asyncio.sleep(0)
    assert not run.done(), "phase must not complete while lanes are parked"
    dispatched_traces = {t.trace_id.split("::", 1)[0] for t in issuer.sent}
    assert len(dispatched_traces) == 1, (
        f"only lane 0's template may dispatch under limit 1, got "
        f"{sorted(dispatched_traces)}"
    )

    strategy.set_lane_limit(3)
    await asyncio.wait_for(run, timeout=15.0)
    assert strategy.completed_traces == 3
    assert issuer.sending_complete_calls >= 1


async def test_no_ramp_admits_all_lanes_immediately():
    """Without a ramp the limit is born at the resolved concurrency and the
    run is byte-identical to before the feature."""
    parsed = _corpus(3)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_PhaseCfg(concurrency=3),
        max_concurrent_traces=3,
        allow_dataset_wrap=False,
    )
    assert strategy._lane_limit == 3
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=15.0)
    assert strategy.completed_traces == 3
