# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterization of the graph replay schedule anchor.

Linear replay picks schedule zero three ways
(``FixedScheduleStrategy.setup_phase``): ``auto_offset_timestamps`` -> first
timestamp; else ``fixed_schedule_start_offset`` -> that value; else ``0.0``.
Graph always anchors on the earliest recorded start in the corpus, and these
tests pin that as CORRECT rather than as an oversight:

* ``auto_offset`` / ``start_offset`` exist only on ``FixedSchedulePhase``, and
  a graph workload with a fixed-schedule phase is rejected outright
  (``_reject_graph_incompatible_phases``), so on the graph path those
  ``CreditPhaseConfig`` fields can only ever hold their not-applicable
  defaults (``False`` / ``None``).
* Graph nodes carry ABSOLUTE unix epoch milliseconds
  (``recorded_start_unix_ms``, stamped by the dynamo trie lowering), so the
  ``auto_offset=False`` branch -- treat timestamps as offsets from benchmark
  start -- would schedule every trace decades out: an epoch-ms ``trace_start``
  of ~1.7e12 divided by 1000 is ~1.7e9 SECONDS, i.e. ~54 years.

Tests 3-5 pin the gates that make the fields unreachable; if any is ever
relaxed, the "graph may ignore these knobs" argument has to be revisited.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.config.flags import CLIConfig
from aiperf.config.flags._converter_profiling import _apply_phase_specific_routes
from aiperf.config.phases import FixedSchedulePhase, PhaseType
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig, _reject_graph_incompatible_phases
from aiperf.timing.strategies.agent_graph_replay import AgentGraphReplayStrategy


class _Issuer:
    async def issue_graph_credit(self, turn: Any) -> bool:
        return True

    def mark_graph_sending_complete(self) -> None: ...
    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...
    async def end_graph_trace(self, trace_id: str) -> None: ...


def _parsed(starts: dict[str, int]) -> ParsedGraph:
    """Multi-graph corpus: one trace per id, each with a recorded start."""
    graphs = {
        trace_id: GraphRecord(
            nodes={
                "n": LlmNode(prompt=["hi"], output="out", recorded_start_unix_ms=start)
            },
            edges=[],
            state={},
        )
        for trace_id, start in starts.items()
    }
    return ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}),
        graphs=graphs,
        traces=[TraceRecord(id=trace_id, graph_ref=trace_id) for trace_id in starts],
    )


def _strategy(parsed: ParsedGraph, **phase_kwargs: Any) -> AgentGraphReplayStrategy:
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENT_GRAPH,
        **phase_kwargs,
    )
    return AgentGraphReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )


# Plausible absolute epoch milliseconds -- the scale the anchor actually sees.
_EPOCH_MS = 1_700_000_000_000


@pytest.mark.parametrize(
    "auto_offset",
    [param(False, id="auto_offset_false"), param(True, id="auto_offset_true")],
)  # fmt: skip
def test_anchor_is_corpus_minimum_regardless_of_auto_offset(
    auto_offset: bool,
) -> None:
    """``auto_offset_timestamps`` does not move the graph anchor either way.

    The graph anchor IS the ``auto_offset=True`` semantic already; the
    ``False`` branch (anchor 0, timestamps as offsets from benchmark start) is
    incoherent against absolute epoch milliseconds.
    """
    strategy = _strategy(
        _parsed({"t-1": _EPOCH_MS + 5_000, "t-2": _EPOCH_MS}),
        auto_offset_timestamps=auto_offset,
    )

    assert strategy._schedule_zero_unix_ms == _EPOCH_MS


def test_anchor_ignores_fixed_schedule_start_offset() -> None:
    """A fixed-schedule start offset is not adopted as the graph anchor."""
    strategy = _strategy(_parsed({"t-1": _EPOCH_MS}), fixed_schedule_start_offset=1_234)

    assert strategy._schedule_zero_unix_ms == _EPOCH_MS


def test_graph_workload_rejects_a_fixed_schedule_phase() -> None:
    """The gate that keeps ``auto_offset``/``start_offset`` off graph runs.

    Those fields live only on ``FixedSchedulePhase``; a graph workload cannot
    have one, so ``CreditPhaseConfig`` can only ever see the defaults.
    """
    phase = FixedSchedulePhase(
        name="profiling",
        type=PhaseType.FIXED_SCHEDULE,
        auto_offset=False,
        start_offset=1_000,
    )

    with pytest.raises(ValueError, match="not supported for graph workloads"):
        _reject_graph_incompatible_phases([], [phase])


def test_fixed_schedule_offset_flags_require_fixed_schedule() -> None:
    """The CLI gate: the offset flags are rejected outside --fixed-schedule.

    A graph run resolves to a CONCURRENCY phase, so passing the offsets
    alongside a graph input fails here rather than silently no-opping.
    """
    cli = CLIConfig(fixed_schedule_start_offset=1_000)
    prof: dict[str, Any] = {"type": PhaseType.CONCURRENCY}

    with pytest.raises(ValueError, match="requires --fixed-schedule"):
        _apply_phase_specific_routes(prof, cli)


def test_graph_run_with_fixed_schedule_is_rejected_end_to_end() -> None:
    """Both gates, exercised through their real CALL SITES.

    The two tests above invoke the gate functions directly, so deleting or
    conditionalizing the ``_reject_graph_incompatible_phases`` call at
    ``timing/config.py:248`` would leave them green while invalidating the
    argument. This drives a real graph trace through ``TimingConfig.from_run``.
    """
    from aiperf.config.dataset.resolver import DatasetResolver
    from aiperf.timing.config import TimingConfig
    from tests.unit.conftest import make_run_from_cli

    trace = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "graph"
        / "adapters"
        / "fixtures"
        / "dynamo_nested"
        / "nested_2_level.jsonl.gz"
    )
    cli = CLIConfig(
        model_names=["m"],
        tokenizer="builtin",
        input_file=str(trace),
        fixed_schedule=True,
        fixed_schedule_start_offset=1_000,
    )
    run = make_run_from_cli(cli)
    DatasetResolver().resolve(run)

    with pytest.raises(ValueError, match="not supported for graph workloads"):
        TimingConfig.from_run(run)


def test_reanchor_is_the_only_post_init_anchor_writer() -> None:
    """Corpus selection is the sole thing that moves the anchor after init.

    No configured anchor competes with it, so no precedence rule is needed:
    re-anchoring moves the anchor FORWARD to the selected corpus minimum and
    shifts every selected trace by the same constant.
    """
    parsed = _parsed(
        {"t-1": _EPOCH_MS, "t-2": _EPOCH_MS + 10_000, "t-3": _EPOCH_MS + 20_000}
    )
    strategy = _strategy(parsed)
    assert strategy._schedule_zero_unix_ms == _EPOCH_MS

    strategy._reanchor_schedule_zero(parsed.traces[1:])

    assert strategy._schedule_zero_unix_ms == _EPOCH_MS + 10_000
