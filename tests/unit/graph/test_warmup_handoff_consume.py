# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PROFILING consumes the extended-warmup handoff -- frontier resume tests.

Pins the re-cut contract: a profiling pass-0 lane with a handoff entry resumes
the handed-off template at its execution frontier (executed nodes never
re-dispatch; residual re-roots), recycle passes are untouched, and the full
warmup -> teardown -> profiling transition works end to end on real strategy
objects sharing one conversation source (regression guard for the
adapter-tests-skip-validator gap: this test path exercises the REAL phase
transition, not just node-level fields).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.timing.graph_warmup_handoff import GraphWarmupHandoff, LaneHandoff
from aiperf.timing.strategies.cache_bust import build_trace_instance_marker
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def _corpus(n_traces: int) -> Any:
    parsed = from_weka_trace(str(_FIX))
    base = parsed.traces[0]
    traces = [msgspec.structs.replace(base, id=f"t-{i}") for i in range(n_traces)]
    return msgspec.structs.replace(parsed, traces=traces)


class _Config:
    timing_mode = None
    phase = CreditPhase.WARMUP

    def __init__(self, *, concurrency: int | None = None) -> None:
        self.concurrency = concurrency
        self.expected_num_sessions = None
        self.total_expected_requests = None
        self.expected_duration_sec = None


class _StubIssuer:
    """Resolve every graph credit instantly on the next loop tick."""

    def __init__(self) -> None:
        self.observer = None
        self.issued: list[Any] = []

    def bind(self, strategy: GraphIRReplayStrategy) -> None:
        self.observer = strategy._on_graph_return

    async def issue_graph_credit(self, credit: Any) -> bool:
        self.issued.append(credit)
        observer = self.observer
        asyncio.get_running_loop().call_soon(lambda: observer(credit, None, False))
        return True

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


class _ParkAfterIssuer:
    """Resolve the first ``park_after`` graph credits instantly, park the rest.

    Parking stalls the pressure lanes so the ``wait_for`` duration timer fires
    deterministically instead of recycling unboundedly fast under the
    instant-sleep test fixtures.
    """

    def __init__(self, park_after: int | None = None) -> None:
        self.observer = None
        self.issued: list[Any] = []
        self._park_after = park_after

    def bind(self, strategy: GraphIRReplayStrategy) -> None:
        self.observer = strategy._on_graph_return

    async def issue_graph_credit(self, credit: Any) -> bool:
        self.issued.append(credit)
        if self._park_after is not None and len(self.issued) > self._park_after:
            return True  # parked: never resolved
        observer = self.observer
        asyncio.get_running_loop().call_soon(lambda: observer(credit, None, False))
        return True

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


class _FakeSource:
    """Duck-typed graph channel (cross-phase warmup-handoff slot)."""

    def __init__(self) -> None:
        self.warmup_handoff = None


def _warmup_strategy(
    parsed: Any,
    *,
    duration: float | None,
    park_after: int | None = None,
    graph_channel: Any = None,
    concurrency: int = 1,
) -> tuple[GraphIRReplayStrategy, _ParkAfterIssuer]:
    issuer = _ParkAfterIssuer(park_after=park_after)
    strategy = GraphIRReplayStrategy(
        config=_Config(concurrency=concurrency),
        graph_channel=graph_channel,
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.5,
        start_max_ratio=0.5,
        t_star_random_seed=1234,
        cache_pressure_duration_s=duration,
        dispatch_timeout_s=2.0,
    )
    issuer.bind(strategy)
    return strategy, issuer


def _profiling_strategy(parsed, *, handoff, session_cap=1, concurrency=1):
    issuer = _StubIssuer()
    config = _Config(concurrency=concurrency)
    config.phase = CreditPhase.PROFILING
    config.expected_num_sessions = session_cap
    strategy = GraphIRReplayStrategy(
        config=config,
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.5,
        start_max_ratio=0.5,
        t_star_random_seed=1234,
        warmup_handoff=handoff,
    )
    issuer.bind(strategy)
    return strategy, issuer


@pytest.mark.asyncio
async def test_profiling_pass0_skips_executed_nodes():
    parsed = _corpus(1)
    all_nodes = set(parsed.graph.nodes)
    executed = {sorted(all_nodes)[0]}  # first node id, deterministic
    handoff = GraphWarmupHandoff(
        lanes={
            0: LaneHandoff(
                template_trace_id="t-0",
                instance_id="t-0#0.p0",
                t_star_us=0.0,
                executed_node_ids=frozenset(executed),
                return_wall_us={nid: 0.0 for nid in executed},
            )
        },
        drain_end_wall_us=1.0,
        corpus_cursor=1,
        pressure_lane_count=1,
    )
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff)
    await strategy.execute_phase()

    pass0 = [c for c in issuer.issued if c.trace_id == handoff.lanes[0].instance_id]
    dispatched = {getattr(c, "node_ordinal", None) for c in pass0}
    executed_ordinals = {strategy._catalog.catalog["t-0"][nid] for nid in executed}
    assert pass0, "pass-0 must dispatch the surviving frontier"
    assert not (dispatched & executed_ordinals), (
        "executed nodes must NOT re-dispatch in the handoff resume"
    )


@pytest.mark.asyncio
async def test_profiling_lane_runs_handoff_template_not_lane_assignment():
    """Pressure recycled the lane onto t-1; profiling pass-0 must resume t-1."""
    parsed = _corpus(2)  # lane 0's pass-0 assignment would be t-0
    handoff = GraphWarmupHandoff(
        lanes={
            0: LaneHandoff(
                template_trace_id="t-1",
                instance_id="t-1#0.p0",
                t_star_us=0.0,
                executed_node_ids=frozenset(),
                return_wall_us={},
            )
        },
        drain_end_wall_us=1.0,
        corpus_cursor=1,
        pressure_lane_count=1,
    )
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff, session_cap=1)
    await strategy.execute_phase()
    assert any(c.trace_id.startswith("t-1#0.") for c in issuer.issued)


@pytest.mark.asyncio
async def test_profiling_without_handoff_is_unchanged():
    parsed = _corpus(1)
    strategy, issuer = _profiling_strategy(parsed, handoff=None)
    await strategy.execute_phase()
    # One single-pass instance, ``t-0::{nonce}`` (nonce minted fresh per instance).
    ids = {c.trace_id for c in issuer.issued}
    assert len(ids) == 1
    assert ids.pop().split("::", 1)[0] == "t-0"


@pytest.mark.asyncio
async def test_warmup_pressure_to_profiling_end_to_end_no_refire():
    """Full transition on real strategies sharing one source.

    Warmup (priming + pressure, park-stalled) -> teardown stashes handoff ->
    profiling built with the popped handoff -> pass 0 refires NO node the
    pressure stage executed on the live instance.

    ``park_after=2`` is tuned to keep the pass-0 pressure instance LIVE at drain
    with one PRESSURE credit resolved: priming issues one credit (resolves), the
    first pressure credit resolves, the second parks.
    """
    parsed = _corpus(1)
    source = _FakeSource()
    warmup, warmup_issuer = _warmup_strategy(
        parsed, duration=0.2, park_after=2, graph_channel=source
    )
    await warmup.execute_phase()
    await warmup.teardown_phase()
    handoff = source.warmup_handoff
    assert handoff is not None and handoff.lanes

    entry = handoff.lanes[0]
    assert entry.executed_node_ids, "E2E must exercise a non-empty executed set"
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff)
    await strategy.execute_phase()

    executed_ordinals = {
        strategy._catalog.catalog[entry.template_trace_id][nid]
        for nid in entry.executed_node_ids
    }
    # Resumed pass-0 credits carry the stashed live instance id (marker
    # continuity), not a freshly minted ``.0`` id.
    pass0 = [c for c in issuer.issued if c.trace_id == entry.instance_id]
    assert pass0, "profiling pass-0 must dispatch the surviving frontier"
    assert not ({getattr(c, "node_ordinal", None) for c in pass0} & executed_ordinals)


@pytest.mark.asyncio
async def test_resumed_pass0_reuses_pressure_instance_id_for_marker_continuity():
    """Resumed pass-0 credits reuse the stashed live instance id verbatim.

    The per-instance cache-bust marker digests ``credit.trace_id``
    (``build_trace_instance_marker``). Reusing the pressure instance's id at the
    handoff resume keeps that marker byte-identical across the WARMUP -> PROFILING
    boundary, so the KV the pressure stage built at the id transfers instead of
    cold-prefilling behind a fresh ``.0`` marker.
    """
    parsed = _corpus(1)
    source = _FakeSource()
    warmup, _ = _warmup_strategy(
        parsed, duration=0.2, park_after=2, graph_channel=source
    )
    await warmup.execute_phase()
    await warmup.teardown_phase()
    handoff = source.warmup_handoff
    assert handoff is not None and handoff.lanes

    entry = handoff.lanes[0]
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff)
    await strategy.execute_phase()

    resumed = [c for c in issuer.issued if c.trace_id == entry.instance_id]
    assert resumed, "resumed pass-0 credits must carry the stashed live instance id"

    bid = "seed-1234"
    assert build_trace_instance_marker(
        bid, resumed[0].trace_id, target=CacheBustTarget.FIRST_TURN_PREFIX
    ) == build_trace_instance_marker(
        bid, entry.instance_id, target=CacheBustTarget.FIRST_TURN_PREFIX
    )


@pytest.mark.asyncio
async def test_bounded_profiling_recycle_continues_from_handoff_cursor():
    """A bounded profiling recycle resumes the corpus draw at the handoff cursor.

    Corpus t-0,t-1,t-2; lane 0 resumes the handoff template t-0, then recycles.
    ``corpus_cursor=2`` means the pressure stage last drew position 1, so the
    freed lane continues at position 2 (t-2) -- NOT the pass-0 default cursor of
    1 (t-1) a fresh profiling run would use (agentx shared-sampler parity).
    """
    parsed = _corpus(3)
    handoff = GraphWarmupHandoff(
        lanes={
            0: LaneHandoff(
                template_trace_id="t-0",
                instance_id="t-0::press0",
                t_star_us=0.0,
                executed_node_ids=frozenset(),
                return_wall_us={},
            )
        },
        drain_end_wall_us=1.0,
        corpus_cursor=2,
        pressure_lane_count=1,
    )
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff, session_cap=2)
    await strategy.execute_phase()

    templates = {c.trace_id.split("::", 1)[0] for c in issuer.issued}
    assert "t-2" in templates, (
        "the freed lane must recycle from the handoff cursor onto t-2"
    )
    assert "t-1" not in templates, (
        "t-1 (the pass-0 default cursor draw) must NOT be served"
    )


@pytest.mark.asyncio
async def test_single_pass_profiling_ignores_handoff_cursor():
    """Single-pass profiling covers the whole corpus, ignoring the handoff cursor.

    With no stop conditions the lanes do ONE corpus pass whose termination check
    (``next_index >= len(traces)``) encodes cover-the-corpus-once. Carrying the
    handoff's ``corpus_cursor=999`` would skip past the corpus and drop templates,
    so single-pass deliberately keeps its own cursor.
    """
    parsed = _corpus(2)
    handoff = GraphWarmupHandoff(
        lanes={}, drain_end_wall_us=0.0, corpus_cursor=999, pressure_lane_count=0
    )
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff, session_cap=None)
    await strategy.execute_phase()

    templates = {c.trace_id.split("::", 1)[0] for c in issuer.issued}
    assert {"t-0", "t-1"} <= templates, (
        f"single-pass must claim every corpus position once; saw {templates}"
    )


@pytest.mark.asyncio
async def test_single_pass_lane_keeps_pass0_plan_despite_pressure_lane_count():
    """Single-pass mode NEVER fresh-starts: cover-the-corpus-once wins.

    A drained-empty pressure lane (pressure_lane_count=1, no entry) in a
    single-pass profiling run (no stop conditions) keeps its normal pass-0
    assignment -- no .f0 instance, no cursor draw.
    """
    parsed = _corpus(2)
    handoff = GraphWarmupHandoff(
        lanes={},  # lane 0 was a pressure lane but drained empty
        drain_end_wall_us=1.0,
        corpus_cursor=999,  # a fresh-start draw would jump here; single-pass ignores it
        pressure_lane_count=1,
    )
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff, session_cap=None)
    await strategy.execute_phase()

    ids = {c.trace_id for c in issuer.issued}
    templates = {trace_id.split("::", 1)[0] for trace_id in ids}
    # The fresh-start gate is BOUNDED-mode-only: single-pass keeps each lane's
    # normal pass-0 assignment (lane 0 -> t-0), ignoring the handoff cursor.
    assert "t-0" in templates, (
        f"single-pass lane must resume its plain pass-0 assignment; saw {ids}"
    )
    # cover-the-corpus-once still holds despite the nonzero pressure lane count
    assert {"t-0", "t-1"} <= templates, (
        f"single-pass must claim every corpus position once; saw {templates}"
    )


@pytest.mark.asyncio
async def test_fresh_start_lane_runs_full_replay_of_cursor_template():
    """A pressure lane with no live instance at drain fresh-starts in profiling.

    AgentX parity (_build_handoff_trajectories: empty lanes get a fresh
    recycle conversation at turn 0): the lane must NOT re-run its t* resume
    (pressure already replayed it -- measuring it again inflates cache-hit
    stats); it draws the next cursor template and replays it in full.
    """
    parsed = _corpus(3)
    handoff = GraphWarmupHandoff(
        lanes={},  # lane 0 was a pressure lane but completed at drain
        drain_end_wall_us=1.0,
        corpus_cursor=2,
        pressure_lane_count=1,
    )
    strategy, issuer = _profiling_strategy(parsed, handoff=handoff, session_cap=1)
    await strategy.execute_phase()
    # Fresh template drawn from the cursor (t-2), full t*=0 replay under a fresh
    # ``t-2::{nonce}`` instance (the fresh-start flavor is logged, not id-encoded;
    # the per-instance nonce already precludes any boundary-priming id collision).
    pass0 = [c for c in issuer.issued if c.trace_id.split("::", 1)[0] == "t-2"]
    all_ordinals = set(strategy._catalog.catalog["t-2"].values())
    assert pass0, "the fresh-started lane must dispatch the cursor template t-2"
    assert {getattr(c, "node_ordinal", None) for c in pass0} == all_ordinals
    # and the lane-assigned template's t* resume did NOT run
    assert not any(c.trace_id.split("::", 1)[0] == "t-0" for c in issuer.issued)


@pytest.mark.asyncio
async def test_profiling_honors_handoff_lanes_past_session_clamp():
    """--num-conversations < concurrency must not drop drained lanes.

    ALL handoff trajectories dispatch; the session cap gates only
    recycles.
    """
    parsed = _corpus(2)
    handoff = GraphWarmupHandoff(
        lanes={
            0: LaneHandoff(
                template_trace_id="t-0",
                instance_id="t-0#0.p0",
                t_star_us=0.0,
                executed_node_ids=frozenset(),
                return_wall_us={},
            ),
            1: LaneHandoff(
                template_trace_id="t-1",
                instance_id="t-1#1.p0",
                t_star_us=0.0,
                executed_node_ids=frozenset(),
                return_wall_us={},
            ),
        },
        drain_end_wall_us=1.0,
        corpus_cursor=2,
        pressure_lane_count=2,
    )
    # session_cap=1 would normally clamp profiling to ONE lane
    strategy, issuer = _profiling_strategy(
        parsed, handoff=handoff, session_cap=1, concurrency=2
    )
    await strategy.execute_phase()
    ids = {c.trace_id for c in issuer.issued}
    assert "t-0#0.p0" in ids and "t-1#1.p0" in ids
