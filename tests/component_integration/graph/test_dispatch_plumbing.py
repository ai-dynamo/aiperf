# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolved dataset-selection values reach ``GraphIRReplayStrategy``.

Drives the REAL ``TimingConfig.from_run`` -> ``PhaseRunner._build_graph_ir_strategy``
path for a weka graph workload and asserts the constructed strategy stores the
resolved ``dataset_sampling_strategy`` and ``allow_dataset_wrap`` on its
attributes. This is interface plumbing only: consumption of the values is
covered by ``test_wrap_guard.py`` and ``test_graph_sampling.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import TimingConfig
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy
from tests.unit.conftest import make_run_from_cli

pytestmark = pytest.mark.component_integration

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_WEKA_MIN = _FIX_DIR / "weka_min.json"


def _graph_run_with_resolved_selection():
    """Real weka graph run with the resolved selection values ``GraphDispatchResolver`` publishes.

    ``make_run_from_cli`` leaves ``run.resolved`` at its defaults (the resolver
    chain isn't run here), so we set the two fields explicitly to stand in for
    what the post-scenario graph-dispatch resolver derives.
    """
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(_WEKA_MIN),
        request_count=3,
    )
    run = make_run_from_cli(cfg)
    run.resolved.dataset_sampling_strategy = DatasetSamplingStrategy.SHUFFLE
    run.resolved.allow_dataset_wrap = True
    return run


def _graph_runner_for(profiling_config):
    """A ``PhaseRunner`` wired for the real ``_build_graph_ir_strategy`` seam.

    Mirrors ``tests/unit/graph/test_build_strategy_graph_ir.py``'s harness:
    ``PhaseRunner.__new__`` with the collaborators the graph branch reads, and a
    capturing ``GraphIRReplayStrategy`` subclass so the forwarded kwargs are
    observable.
    """
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
    from aiperf.timing.graph_channel import GraphPhaseChannel
    from aiperf.timing.phase.runner import PhaseRunner

    parsed = from_weka_trace(str(_WEKA_MIN))
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

    channel = GraphPhaseChannel(parsed_graph=parsed)
    runner = PhaseRunner.__new__(PhaseRunner)
    runner._config = profiling_config
    runner._conversation_source = None
    runner._graph_channel = channel
    runner._scheduler = None
    runner._stop_checker = None
    runner._credit_issuer = object()
    runner._lifecycle = None
    runner._callback_handler = _Handler()
    return runner, captured, _CapturingStrategy


def test_resolved_selection_reaches_graph_strategy() -> None:
    run = _graph_run_with_resolved_selection()

    tc = TimingConfig.from_run(run)
    profiling = [p for p in tc.phase_configs if p.phase == CreditPhase.PROFILING]
    assert profiling, "expected a graph profiling phase"
    cfg = profiling[0]
    assert cfg.timing_mode == TimingMode.GRAPH_IR

    # from_run copies the resolved selection onto the CreditPhaseConfig.
    assert cfg.dataset_sampling_strategy == DatasetSamplingStrategy.SHUFFLE
    assert cfg.allow_dataset_wrap is True

    runner, captured, StrategyClass = _graph_runner_for(cfg)
    strategy = runner._build_graph_ir_strategy(StrategyClass)

    assert isinstance(strategy, GraphIRReplayStrategy)
    # Stored on the strategy for Tasks 11/12 to consume.
    assert strategy._dataset_sampling_strategy == DatasetSamplingStrategy.SHUFFLE
    assert strategy._allow_dataset_wrap is True
    # Forwarded as explicit kwargs by the runner, not left to defaults.
    assert captured["dataset_sampling_strategy"] == DatasetSamplingStrategy.SHUFFLE
    assert captured["allow_dataset_wrap"] is True
