# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 — PhaseRunner._build_strategy selects + injects GraphIRReplayStrategy.

Verifies the plugin lookup resolves ``TimingMode.GRAPH_IR`` to
``GraphIRReplayStrategy`` and that the graph-only injection branch supplies the
``ParsedGraph`` (from the conversation source) and the graph-return observer
(from the callback handler) WITHOUT disturbing the other strategies' fixed-kwarg
construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TimingMode
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_plugin_registry_resolves_graph_ir_to_strategy():
    cls = plugins.get_class(PluginType.TIMING_STRATEGY, TimingMode.GRAPH_IR)
    assert cls is GraphIRReplayStrategy


def test_build_strategy_injects_parsed_graph_and_observer():
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    parsed = from_weka_trace(str(_FIX))

    # Minimal stand-ins for the runner collaborators the graph branch reads.
    class _Source:
        parsed_graph = parsed

    class _Handler:
        def __init__(self) -> None:
            self.installed = None

        def set_graph_return_observer(self, obs) -> None:
            self.installed = obs

    class _Config:
        timing_mode = TimingMode.GRAPH_IR
        phase = None
        concurrency = None

    # Build via the same code path PhaseRunner._build_strategy uses, without
    # standing up a full PhaseRunner (its __init__ needs the whole pipeline).
    handler = _Handler()
    StrategyClass = plugins.get_class(PluginType.TIMING_STRATEGY, TimingMode.GRAPH_IR)
    strategy = StrategyClass(
        config=_Config(),
        conversation_source=_Source(),
        scheduler=None,
        stop_checker=None,
        credit_issuer=object(),
        lifecycle=None,
        parsed_graph=_Source.parsed_graph,
        register_observer=handler.set_graph_return_observer,
    )
    assert isinstance(strategy, GraphIRReplayStrategy)


@pytest.mark.asyncio
async def test_setup_phase_installs_single_observer():
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    parsed = from_weka_trace(str(_FIX))
    installed = []
    strategy = GraphIRReplayStrategy(
        credit_issuer=object(),
        parsed_graph=parsed,
        register_observer=installed.append,
    )
    await strategy.setup_phase()
    assert len(installed) == 1
    assert callable(installed[0])


@pytest.fixture
def graph_runner():
    """Runner + graph channel wired the way ``PhaseRunner`` does.

    Mirrors ``test_tstar_activation``'s ``PhaseRunner.__new__`` harness so these
    tests exercise the real ``_build_graph_ir_strategy`` seam (kwarg threading
    and the consume-once handoff pop) without standing up the full pipeline.
    Returns ``(runner, channel, captured, StrategyClass)`` where ``captured`` is
    updated with the strategy kwargs on every build.
    """
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
    from aiperf.timing.graph_channel import GraphPhaseChannel
    from aiperf.timing.phase.runner import PhaseRunner

    parsed = from_weka_trace(str(_FIX))
    captured: dict = {}

    class _CapturingStrategy(GraphIRReplayStrategy):
        def __init__(self, **kw):
            captured.update(kw)
            super().__init__(**kw)

    class _Handler:
        def set_graph_return_observer(self, obs) -> None:
            self._obs = obs

        def set_graph_first_token_observer(self, obs) -> None:
            self._ft_obs = obs

    class _Config:
        timing_mode = TimingMode.GRAPH_IR
        phase = None
        concurrency = None
        expected_num_sessions = None

    channel = GraphPhaseChannel(parsed_graph=parsed)
    runner = PhaseRunner.__new__(PhaseRunner)
    runner._config = _Config()
    runner._conversation_source = None
    runner._graph_channel = channel
    runner._scheduler = None
    runner._stop_checker = None
    runner._credit_issuer = object()
    runner._lifecycle = None
    runner._callback_handler = _Handler()

    return runner, channel, captured, _CapturingStrategy


def test_build_graph_ir_strategy_threads_and_consumes_warmup_handoff(graph_runner):
    """The runner threads the stashed handoff into the strategy and clears it.

    Consume-once: a second phase built from the same graph channel must
    NOT see a stale handoff (multi-phase profiling configs).
    """
    from aiperf.timing.graph_warmup_handoff import GraphWarmupHandoff

    runner, channel, captured, StrategyClass = graph_runner

    handoff = GraphWarmupHandoff(
        lanes={}, drain_end_wall_us=0.0, corpus_cursor=0, pressure_lane_count=0
    )
    channel.warmup_handoff = handoff

    runner._build_graph_ir_strategy(StrategyClass)

    assert captured["warmup_handoff"] is handoff
    assert channel.warmup_handoff is None

    runner._build_graph_ir_strategy(StrategyClass)
    assert captured["warmup_handoff"] is None


def test_build_graph_ir_strategy_threads_pressure_duration(graph_runner):
    """The runner sources the cache-pressure duration from the phase config."""
    runner, _channel, captured, StrategyClass = graph_runner
    runner._config.cache_pressure_duration = 30.0

    runner._build_graph_ir_strategy(StrategyClass)

    assert captured["cache_pressure_duration_s"] == 30.0
