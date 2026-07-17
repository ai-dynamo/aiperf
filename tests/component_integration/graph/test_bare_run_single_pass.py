# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A BARE graph run does a SINGLE PASS OVER N SESSIONS: each loaded trace once.

With no explicit stop condition (``--request-count`` / ``--num-conversations`` /
``--benchmark-duration`` all unset) the run stays in SINGLE-PASS mode:
``GraphIRReplayStrategy._recycle_has_stop_condition()`` is ``False`` (it reads only
EXPLICIT stop conditions), so the lane fan-out clamps to the corpus size and each
lane does exactly ONE corpus pass -- clean pass-0 plans, no recycle, no
fresh-start, no auto-10 truncation.

Separately, ``_resolved_num_sessions()`` derives the reported session TARGET
``N = len(self._parsed.traces)`` (the loaded trace count -- mirroring dag_jsonl's
roots->sessions convention). That target drives lane clamping and progress
reporting ONLY; it is NOT a recycle bound (routing it into the recycle gate would
flip the bare run out of single-pass mode). An explicit ``--num-conversations``
sets a real stop condition and takes the bounded/recycle path instead.

Component-level (no worker / ZMQ / mock server): a fake ``CreditIssuer`` echoes
each issued credit back to the strategy's return observer, so ``execute_phase``
drives the REAL dispatch path over REAL weka traces.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase

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

    def set_graph_all_returned_event(self) -> None: ...


class _BarePhaseCfg:
    """Per-phase config stub with NO stop condition (bare graph run)."""

    def __init__(self, *, concurrency: int | None = None) -> None:
        self.phase = CreditPhase.PROFILING
        self.concurrency = concurrency
        self.expected_num_sessions = None
        self.total_expected_requests = None
        self.expected_duration_sec = None
        self.num_dataset_entries = None
        self.max_context_length = None


def _corpus_distinct(n: int):
    """Return a gap-free ParsedGraph whose ``traces`` holds ``n`` clean-id traces.

    The single ``weka_min`` template is cloned into ``n`` traces with ids
    ``t-0``..``t-{n-1}`` (no ``#`` so the instance-id split is unambiguous), and
    edge delays are zeroed so a dispatching run replays instantly.
    """
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
    clones = [msgspec.structs.replace(t0, id=f"t-{i}") for i in range(n)]
    return msgspec.structs.replace(base, traces=clones)


def _make_strategy(parsed, issuer: _EchoIssuer, **overrides):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    overrides.setdefault("start_min_ratio", 0.0)
    overrides.setdefault("start_max_ratio", 0.0)
    overrides.setdefault("cache_bust", CacheBustTarget.NONE)
    return GraphIRReplayStrategy(
        config=overrides.pop("config", None),
        credit_issuer=issuer,
        lifecycle=overrides.pop("lifecycle", None),
        parsed_graph=parsed,
        register_observer=lambda obs: setattr(issuer, "observer", obs),
        **overrides,
    )


def _distinct_templates(issuer: _EchoIssuer) -> set[str]:
    """Distinct template trace ids (instance ``t-i#p`` -> ``t-i``) among sent turns."""
    templates: set[str] = set()
    for turn in issuer.sent:
        trace_id = getattr(turn, "trace_id", None)
        if trace_id is not None:
            templates.add(str(trace_id).split("::", 1)[0])
    return templates


@pytest.mark.parametrize("n", [3, 7])
async def test_bare_run_dispatches_each_trace_exactly_once(n: int) -> None:
    """No explicit stop + concurrency==corpus -> single pass over N sessions."""
    parsed = _corpus_distinct(n)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_BarePhaseCfg(concurrency=n),
        max_concurrent_traces=n,
    )

    # Reported session target == loaded trace count (N), but the run stays in
    # SINGLE-PASS mode: no explicit stop condition, so recycle is OFF (each trace
    # once via one corpus pass, clean pass-0 plans, no fresh-start).
    assert strategy._resolved_num_sessions() == n
    assert strategy._recycle_has_stop_condition() is False

    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=30.0)

    # Single pass over N sessions: N instances, each admitted + completed once.
    assert strategy._instances_started == n
    assert strategy.admitted_traces == n
    assert strategy.completed_traces == n
    # And every distinct loaded template ran exactly once (no recycle clones).
    assert _distinct_templates(issuer) == {f"t-{i}" for i in range(n)}


async def test_bare_run_single_lane_covers_full_corpus() -> None:
    """Default concurrency 1 (no explicit stop) covers the whole N-session corpus once."""
    n = 5
    parsed = _corpus_distinct(n)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_BarePhaseCfg(concurrency=None),
        max_concurrent_traces=None,  # resolves to the aiperf default of 1
    )
    assert strategy._max_concurrent == 1
    # Reported session target = loaded trace count, and single-pass mode (no
    # explicit stop -> recycle OFF), even at concurrency 1.
    assert strategy._resolved_num_sessions() == n
    assert strategy._recycle_has_stop_condition() is False

    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=30.0)

    assert strategy._instances_started == n
    assert strategy.completed_traces == n
    assert _distinct_templates(issuer) == {f"t-{i}" for i in range(n)}


async def test_explicit_num_conversations_overrides_derived_sessions() -> None:
    """An explicit --num-conversations wins over the derived corpus-size target."""
    n = 6
    parsed = _corpus_distinct(n)
    issuer = _EchoIssuer()
    cfg = _BarePhaseCfg(concurrency=2)
    cfg.expected_num_sessions = 4  # explicit bound below the corpus size
    strategy = _make_strategy(parsed, issuer, config=cfg, max_concurrent_traces=2)

    assert strategy._resolved_num_sessions() == 4

    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=30.0)

    # Explicit cap bounds the run at 4 sessions, not the 6-trace corpus.
    assert strategy._instances_started == 4
    assert strategy.completed_traces == 4
