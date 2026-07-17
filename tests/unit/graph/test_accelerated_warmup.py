# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accelerated cache-pressure warmup -- knob-gated pacing.

These tests pin the zero-idle warmup replay (live trajectories replay with
ZERO idle delay to drive the server KV cache to pressure):
the WARMUP-phase ``TraceExecutor`` is built with ``compress_edge_delays=True``
ONLY when a cache-pressure duration is configured, collapsing
every captured inter-node edge delay. Default (knob OFF) honors every edge delay
exactly. The 1-token warmup output
cap (``WARMUP_MAX_OUTPUT_TOKENS``) is independent of pacing and untouched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.graph.executor import TraceExecutor

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


class _RecordingCreditIssuer:
    """Stub issuer recording dispatch order; returns a placeholder string."""

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> str:
        self.dispatched.append(request.node_id)
        return f"placeholder::{request.node_id}"


def _count_real_sleeps(monkeypatch) -> list[float]:
    """Patch ``asyncio.sleep`` in the executor module to record positive waits.

    ``_apply_firing_delay`` calls ``await asyncio.sleep(wait_us / 1e6)`` only
    when a non-zero gate remains after honoring edge delays. Recording the
    POSITIVE-duration sleeps it issues is a direct, virtual-time-independent
    probe of whether the executor honored (>=1 positive sleep on a delayed
    chain) or collapsed (zero positive sleeps) the edge delays.
    """
    sleeps: list[float] = []
    import aiperf.graph.executor as executor_mod

    real_sleep = asyncio.sleep

    async def _spy(delay: float, *args, **kwargs):
        if delay > 0:
            sleeps.append(delay)
        return await real_sleep(0)

    monkeypatch.setattr(executor_mod.asyncio, "sleep", _spy)
    return sleeps


@pytest.mark.asyncio
async def test_executor_honors_edge_delays_by_default(monkeypatch):
    """Default executor (compress_edge_delays=False) honors the F1 edge delays.

    The trie IR always stamps end-to-start inter-turn delays, so ``weka_min``
    carries them (700 ms, 500 ms) on its main chain and the default executor must
    issue at least one positive sleep in ``_apply_firing_delay``.
    """
    monkeypatch.setattr(Environment.GRAPH, "IGNORE_EDGE_DELAYS", False)
    sleeps = _count_real_sleeps(monkeypatch)

    parsed = from_weka_trace(str(FIX))
    issuer = _RecordingCreditIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer)

    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            await executor.run(trace)

    assert sleeps, (
        "default executor must honor F1 end-to-start edge delays (expected at "
        "least one positive firing-delay sleep on weka_min's 700/500 ms chain)"
    )


@pytest.mark.asyncio
async def test_executor_collapses_edge_delays_when_compressed(monkeypatch):
    """compress_edge_delays=True collapses ALL firing delays (zero-idle / burst).

    The live trajectories replay with zero idle delay. No positive
    firing-delay sleep
    may be issued, yet every node still dispatches exactly once.
    """
    monkeypatch.setattr(Environment.GRAPH, "IGNORE_EDGE_DELAYS", False)
    sleeps = _count_real_sleeps(monkeypatch)

    parsed = from_weka_trace(str(FIX))
    from aiperf.dataset.graph.models import LlmNode

    graph = parsed.graph if not parsed.graphs else next(iter(parsed.graphs.values()))
    expected = {nid for nid, n in graph.nodes.items() if isinstance(n, LlmNode)}

    issuer = _RecordingCreditIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer, compress_edge_delays=True)

    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            await executor.run(trace)

    assert sleeps == [], (
        f"compressed executor must collapse all firing delays (zero idle), got "
        f"positive sleeps {sleeps}"
    )
    assert set(issuer.dispatched) == expected, (
        "compressed warmup must still dispatch every node exactly once"
    )


class _StubIssuer:
    def mark_graph_sending_complete(self):
        pass

    def graph_all_returned(self):
        return True

    def set_graph_all_returned_event(self):
        pass


def _strategy(
    parsed, *, phase: CreditPhase | None, cache_pressure_duration_s: float | None = None
):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.phase = phase
    return GraphIRReplayStrategy(
        config=cfg,
        credit_issuer=_StubIssuer(),
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.25,
        start_max_ratio=0.75,
        t_star_random_seed=1234,
        cache_pressure_duration_s=cache_pressure_duration_s,
    )


@pytest.mark.parametrize(
    "phase,duration,expected",
    [
        # A WARMUP phase with no pressure duration honors recorded pacing.
        (CreditPhase.WARMUP, None, False),
        (CreditPhase.PROFILING, None, False),
        (None, None, False),
        # The configured duration (--agentic-cache-warmup-duration) alone
        # activates compressed pacing -- WARMUP phase only.
        (CreditPhase.WARMUP, 30.0, True),
        (CreditPhase.PROFILING, 30.0, False),
        (None, 30.0, False),
    ],
)
def test_strategy_accelerated_warmup_gate(phase, duration, expected):
    """Accelerated warmup is active iff WARMUP phase + a pressure duration."""
    parsed = from_weka_trace(str(FIX))
    strategy = _strategy(parsed, phase=phase, cache_pressure_duration_s=duration)
    assert strategy.accelerated_warmup is expected
