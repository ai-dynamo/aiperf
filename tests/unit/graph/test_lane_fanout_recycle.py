# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lane fan-out + recycle + marker rotation locking tests.

The load-generation model -- ``concurrency`` lanes wrapping the corpus
and recycling on every root final turn, gated by
``stop_checker.can_start_new_session`` -- on the dataflow
``GraphIRReplayStrategy``. These tests pin the three coupled behaviors a
single-pass model would miss:

* **Lane fan-out**: ``concurrency`` lanes are built EVEN WHEN concurrency
  exceeds the corpus size (lane ``i`` wraps onto ``traces[i % N]``), instead of
  dispatching only ``N`` traces.
* **Recycle**: a freed lane re-dispatches a fresh root until the
  stop-condition gate refuses, so a duration / request-count / session-count run
  sustains load instead of stopping after one corpus pass.
* **Marker rotation**: each recycle pass mints a fresh ``{trace_id}#{pass}``
  instance id, so the per-instance cache-bust marker rotates across passes while
  staying constant within one instance.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.timing.strategies.cache_bust import build_trace_instance_marker
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def _corpus(n_traces: int) -> Any:
    """Build a ``ParsedGraph`` whose ``traces`` list has ``n_traces`` distinct ids.

    The single-trace ``weka_min`` fixture is replicated under fresh trace ids so
    the strategy sees an ``N``-trace corpus to wrap lanes over. Every replica
    shares the one graph topology (single-graph workload), which is all the
    lane/recycle logic needs.
    """
    parsed = from_weka_trace(str(_FIX))
    base = parsed.traces[0]
    traces = [msgspec.structs.replace(base, id=f"t-{i}") for i in range(n_traces)]
    return msgspec.structs.replace(parsed, traces=traces)


class _Config:
    """Minimal per-phase config the strategy reads stop thresholds from."""

    timing_mode = None
    phase = None

    def __init__(
        self,
        *,
        concurrency: int | None = None,
        expected_num_sessions: int | None = None,
        total_expected_requests: int | None = None,
        expected_duration_sec: float | None = None,
    ) -> None:
        self.concurrency = concurrency
        self.expected_num_sessions = expected_num_sessions
        self.total_expected_requests = total_expected_requests
        self.expected_duration_sec = expected_duration_sec


class _StubIssuer:
    """Issuer whose graph credits resolve immediately via the return observer.

    Recycle is driven by the executor completing, which only happens once every
    parked dispatch Future resolves. We schedule the strategy's own
    ``_on_graph_return`` on the loop so each issued credit returns successfully,
    letting the per-lane recycle loop iterate.
    """

    def __init__(self) -> None:
        self.observer = None
        self.issued: list[Any] = []

    def bind(self, strategy: GraphIRReplayStrategy) -> None:
        self.observer = strategy._on_graph_return

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued.append(turn)
        observer = self.observer
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda: observer(turn, None, False))
        return True

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


def _strategy(
    parsed: Any, config: _Config, stop_checker: Any = None
) -> tuple[GraphIRReplayStrategy, _StubIssuer]:
    issuer = _StubIssuer()
    strategy = GraphIRReplayStrategy(
        config=config,
        credit_issuer=issuer,
        stop_checker=stop_checker,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
    )
    issuer.bind(strategy)
    return strategy, issuer


# ---------------------------------------------------------------------------
# C2 -- lane fan-out exceeds corpus size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lane_count_is_concurrency_not_corpus_size():
    """concurrency=8 over a 2-trace corpus builds 8 lanes (AgentX _target_size).

    The prior model ran only ``N=2`` traces; the port wraps ``concurrency=8``
    lanes onto ``traces[i % 2]``, so the strategy sustains 8 in-flight instances.
    A large session cap (>= concurrency) lets all 8 lanes start (the lane count is
    ``min(concurrency, cap)``); the 8 INITIAL lanes are what we assert here.
    """
    parsed = _corpus(2)
    config = _Config(concurrency=8, expected_num_sessions=8)

    class _AllowAll:
        def can_send_dag_child_turn(self) -> bool:
            return True

    strategy, _issuer = _strategy(parsed, config, stop_checker=_AllowAll())

    runs: list[tuple[str, int, int]] = []
    original = strategy._run_instance

    async def _spy(trace, lane_index, recycle_pass, **kwargs):
        runs.append((trace.id, lane_index, recycle_pass))
        await original(trace, lane_index, recycle_pass, **kwargs)

    strategy._run_instance = _spy
    await strategy.execute_phase()

    # 8 lanes, cap=8 -> exactly 8 initial instances (lanes 0..7), no recycle.
    assert len(runs) == 8, f"expected 8 lanes fanned out, got {len(runs)}"
    initial = [r for r in runs if r[2] == 0]
    lanes = sorted(r[1] for r in initial)
    assert lanes == list(range(8)), f"lanes not 0..7: {lanes}"
    # Lane i wraps onto traces[i % 2].
    for trace_id, lane, _pass in initial:
        assert trace_id == f"t-{lane % 2}", (
            f"lane {lane} wrapped onto {trace_id}, expected t-{lane % 2}"
        )


@pytest.mark.asyncio
async def test_no_stop_condition_runs_single_corpus_pass():
    """A bare run (no cap) covers the WHOLE corpus exactly once, then stops.

    corpus=3, concurrency=2: two lanes start on t-0/t-1 and the first freed
    lane draws the last unclaimed template t-2 -- a SINGLE corpus pass (every
    template claimed exactly once), not one-instance-per-lane truncation and
    not unbounded recycle.
    """
    parsed = _corpus(3)
    config = _Config(concurrency=2)  # concurrency < corpus, no cap
    strategy, issuer = _strategy(parsed, config)

    await strategy.execute_phase()
    assert strategy.admitted_traces == 3
    assert strategy.completed_traces == 3
    # Every template of the corpus ran exactly once (instance ids are
    # "{template}::{nonce}"; the base template is everything before the '::').
    templates = [t.trace_id.split("::", 1)[0] for t in issuer.issued]
    assert set(templates) == {"t-0", "t-1", "t-2"}


# ---------------------------------------------------------------------------
# C1 -- recycle until the stop condition fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recycle_until_session_count_cap():
    """concurrency=2, --num-conversations=6 dispatches exactly 6 root instances.

    The v1 ``CreditCounter`` never bumps ``sent_sessions`` for graph credits, so
    the strategy itself counts admitted roots and stops recycling at the cap --
    AgentX's ``can_start_new_session`` gate semantics, enforced strategy-side.
    """
    parsed = _corpus(2)
    config = _Config(concurrency=2, expected_num_sessions=6)

    class _AllowAll:
        def can_send_dag_child_turn(self) -> bool:
            return True

    strategy, _issuer = _strategy(parsed, config, stop_checker=_AllowAll())
    await strategy.execute_phase()

    assert strategy.admitted_traces == 6, (
        f"expected 6 root instances (2 lanes recycling to the cap), got "
        f"{strategy.admitted_traces}"
    )
    assert strategy.completed_traces == 6


@pytest.mark.asyncio
async def test_recycle_stops_when_request_count_gate_closes():
    """Recycle halts once ``can_send_dag_child_turn`` (request-count) refuses.

    The gate flips False after a budget of new-root admissions; the lanes must
    finish their in-flight instance and then stop recycling -- no unbounded spin.
    """
    parsed = _corpus(2)
    # No session cap: rely purely on the request-count-style gate to stop.
    config = _Config(concurrency=2, total_expected_requests=999)

    class _BudgetGate:
        def __init__(self, budget: int) -> None:
            self.calls = 0
            self.budget = budget

        def can_send_dag_child_turn(self) -> bool:
            self.calls += 1
            return self.calls <= self.budget

    gate = _BudgetGate(budget=3)
    strategy, _issuer = _strategy(parsed, config, stop_checker=gate)
    await strategy.execute_phase()

    # 2 initial lanes + recycles permitted while the gate is open (3 opens),
    # then both lanes see it closed and stop. Bounded, never infinite.
    assert 2 <= strategy.admitted_traces <= 2 + 3
    assert strategy.completed_traces == strategy.admitted_traces


# ---------------------------------------------------------------------------
# C3 -- cache-bust marker rotates across recycle passes
# ---------------------------------------------------------------------------


def test_marker_rotates_across_recycle_passes():
    """Distinct recycle-pass instance ids mint DISTINCT markers; same id shares one.

    The strategy stamps ``{trace_id}#{recycle_pass}`` on ``credit.trace_id``; the
    worker digests that instance id (``build_trace_instance_marker``). Rotating
    the pass rotates the digest, so a recycled instance cannot warm the prior
    pass's prefix -- AgentX's per-recycle ``recycle_pass`` bump.
    """
    bench = "seed-123"
    ftp = CacheBustTarget.FIRST_TURN_PREFIX
    # Instance id is ``{trace}#{lane}.{pass}``; rotate the pass on one lane.
    pass0 = build_trace_instance_marker(bench, "t-0#0.0", target=ftp)
    pass1 = build_trace_instance_marker(bench, "t-0#0.1", target=ftp)
    pass2 = build_trace_instance_marker(bench, "t-0#0.2", target=ftp)
    assert pass0 != pass1 != pass2 and pass0 != pass2, (
        "marker must rotate per recycle pass"
    )
    # Two concurrent lanes wrapping the SAME template also decorrelate (AgentX's
    # per-lane trajectory_index in the digest): t-0 on lane 0 vs lane 2.
    lane0 = build_trace_instance_marker(bench, "t-0#0.0", target=ftp)
    lane2 = build_trace_instance_marker(bench, "t-0#2.0", target=ftp)
    assert lane0 != lane2, "same template on distinct lanes must mint distinct markers"
    # Within one instance the marker is stable (every turn shares it).
    again = build_trace_instance_marker(bench, "t-0#0.1", target=ftp)
    assert again == pass1


@pytest.mark.asyncio
async def test_adapter_stamps_instance_id_on_credit_trace_id():
    """The per-recycle adapter stamps a nonce-bearing INSTANCE id on
    ``credit.trace_id``.

    Catalog/envelope lookups key on the BASE template id; the marker + return
    de-mux key on the instance id (``{template}::{nonce}``, minted fresh per
    recycle). We assert two DISTINCT instance ids (pass 0 + pass 1) are stamped,
    both stripping to the template ``t-0``, while the credit's ``conversation_id``
    is the stable TEMPLATE trajectory id (nonce-free, deliberately shared across
    recycles) -- NOT the per-instance trace_id.
    """
    parsed = _corpus(1)
    config = _Config(concurrency=1, expected_num_sessions=2)

    class _AllowAll:
        def can_send_dag_child_turn(self) -> bool:
            return True

    strategy, issuer = _strategy(parsed, config, stop_checker=_AllowAll())
    await strategy.execute_phase()

    instance_ids = {t.trace_id for t in issuer.issued}
    # Two distinct nonce-bearing instances (pass 0 + pass 1 on lane 0).
    assert len(instance_ids) == 2, f"expected 2 recycle instances: {instance_ids}"
    assert all("::" in tid for tid in instance_ids), instance_ids
    # The worker strips ``::{nonce}`` back to the base template id.
    assert all(t.trace_id.split("::", 1)[0] == "t-0" for t in issuer.issued)
    # conversation_id is the stable TEMPLATE trajectory id: nonce-free and
    # identical across both recycle instances (instance identity rides trace_id).
    conversation_ids = {t.conversation_id for t in issuer.issued}
    assert len(conversation_ids) == 1, conversation_ids
    assert conversation_ids.pop().split("::", 1)[0] == "t-0"
    for turn in issuer.issued:
        assert turn.conversation_id != turn.trace_id
