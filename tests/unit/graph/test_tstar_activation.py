# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""t* activation: the config window + env seed thread through to GraphIRReplayStrategy.

The prior unit built the t* machinery (``GraphIRConversationSource`` sampling,
warmup variants, snapshot rewrite) but left it INERT: nothing passed real ratios
into the strategy, so every run sampled t*=0 (identity replay). These tests pin
the activation seam:

* ``BenchmarkConfig`` exposes ``trajectory_start_min/max_ratio``
  (``--trajectory-start-min/max-ratio``); the sampling seed is the run's
  resolved ``--random-seed`` carried on the phase config,
* ``PhaseRunner._build_graph_ir_strategy`` threads the phase config's window
  + the env seed into the strategy, so a graph run with a positive window
  samples a non-trivial t* (NOT t*=0 identity).
"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.plugin.enums import TimingMode
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_config_tstar_window_defaults_off() -> None:
    # Bare default is full replay (t* window OFF); the AgentX 0.0..1.0
    # window is scenario-applied (--scenario inferencex-agentx-mvp).
    from aiperf.config import BenchmarkConfig

    assert BenchmarkConfig.model_fields["trajectory_start_min_ratio"].default is None
    assert BenchmarkConfig.model_fields["trajectory_start_max_ratio"].default is None


def test_strategy_with_positive_window_samples_nonzero_tstar() -> None:
    """A deterministic 0.5..0.5 window engages t* (non-identity partition)."""
    parsed = from_weka_trace(str(_FIX))
    strategy = GraphIRReplayStrategy(
        credit_issuer=object(),
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.5,
        start_max_ratio=0.5,
        t_star_random_seed=42,
    )
    plans = strategy._plans
    assert plans, "expected a per-trace t* plan"
    gt = next(iter(plans.values()))
    assert gt.t_star_us > 0, "positive window must engage t* (not the inert t*=0)"


def test_strategy_zero_window_is_inert_identity() -> None:
    """Explicit [0, 0] strategy kwargs keep the byte-identical t*=0 path."""
    parsed = from_weka_trace(str(_FIX))
    strategy = GraphIRReplayStrategy(
        credit_issuer=object(),
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
    )
    gt = next(iter(strategy._plans.values()))
    assert gt.t_star_us == 0


def test_strategy_default_window_is_inert() -> None:
    """No ratio kwargs => the 0.0..0.0 constructor default keeps t*=0 (full replay)."""
    parsed = from_weka_trace(str(_FIX))
    strategy = GraphIRReplayStrategy(
        credit_issuer=object(),
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        t_star_random_seed=42,
    )
    gt = next(iter(strategy._plans.values()))
    assert gt.t_star_us == 0


def test_build_graph_ir_strategy_threads_config_ratios() -> None:
    """``PhaseRunner._build_graph_ir_strategy`` sources the phase-config window
    plus the env seed.

    Build the strategy via the same code path the runner uses (without standing
    up a full PhaseRunner) and confirm the config-driven positive window engages t*.
    """
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
        trajectory_start_min_ratio = 0.5
        trajectory_start_max_ratio = 0.5
        burst_phase_starts = None
        random_seed = 7

    from aiperf.timing.graph_channel import GraphPhaseChannel
    from aiperf.timing.phase.runner import PhaseRunner

    runner = PhaseRunner.__new__(PhaseRunner)
    runner._config = _Config()
    runner._conversation_source = None
    runner._graph_channel = GraphPhaseChannel(parsed_graph=parsed)
    runner._scheduler = None
    runner._stop_checker = None
    runner._credit_issuer = object()
    runner._lifecycle = None
    runner._callback_handler = _Handler()

    strategy = runner._build_graph_ir_strategy(_CapturingStrategy)

    assert captured["start_min_ratio"] == 0.5
    assert captured["start_max_ratio"] == 0.5
    assert captured["t_star_random_seed"] == 7
    gt = next(iter(strategy._plans.values()))
    assert gt.t_star_us > 0, "config window must engage t* through the runner seam"
