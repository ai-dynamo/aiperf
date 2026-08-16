# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace timing artifact: _write_trace_summary_artifact + strategy accumulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiperf.common.enums import CreditPhase
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.graph.executor import TraceResult
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.agent_graph_replay import (
    AgentGraphReplayStrategy,
    _write_trace_summary_artifact,
)

# ---------------------------------------------------------------------------
# _write_trace_summary_artifact — pure writer
# ---------------------------------------------------------------------------


def test_write_trace_summary_artifact_creates_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "profile_export_graph_trace_summary.json"
    summaries = [
        {
            "trace_id": "t-1",
            "total_s": 10.0,
            "model_s": 8.0,
            "tool_s": 1.0,
            "model_time_fraction": 0.8,
            "tool_time_fraction": 0.1,
            "model_calls": 3,
            "tool_calls": 2,
        },
        {
            "trace_id": "t-2",
            "total_s": 20.0,
            "model_s": 15.0,
            "tool_s": 3.0,
            "model_time_fraction": 0.75,
            "tool_time_fraction": 0.15,
            "model_calls": 5,
            "tool_calls": 4,
        },
    ]
    _write_trace_summary_artifact(path, summaries=summaries)
    assert path.exists()
    data = json.loads(path.read_bytes())
    assert data["trace_count"] == 2
    assert data["traces"] == summaries
    agg = data["aggregate"]
    assert abs(agg["total_s"] - 30.0) < 1e-9
    assert abs(agg["model_s"] - 23.0) < 1e-9
    assert abs(agg["tool_s"] - 4.0) < 1e-9


def test_write_trace_summary_aggregate_fractions_are_in_0_1_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    summaries = [
        {
            "trace_id": "t-1",
            "total_s": 5.0,
            "model_s": 4.0,
            "tool_s": 0.5,
            "model_time_fraction": 0.8,
            "tool_time_fraction": 0.1,
            "model_calls": 2,
            "tool_calls": 1,
        }
    ]
    _write_trace_summary_artifact(path, summaries=summaries)
    data = json.loads(path.read_bytes())
    agg = data["aggregate"]
    assert 0.0 <= agg["model_time_fraction"] <= 1.0
    assert 0.0 <= agg["tool_time_fraction"] <= 1.0


def test_write_trace_summary_aggregate_fractions_match_math(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summaries = [
        {
            "trace_id": "t-1",
            "total_s": 10.0,
            "model_s": 6.0,
            "tool_s": 2.0,
            "model_time_fraction": 0.6,
            "tool_time_fraction": 0.2,
            "model_calls": 3,
            "tool_calls": 1,
        }
    ]
    _write_trace_summary_artifact(path, summaries=summaries)
    data = json.loads(path.read_bytes())
    agg = data["aggregate"]
    assert abs(agg["model_time_fraction"] - 0.6) < 1e-9
    assert abs(agg["tool_time_fraction"] - 0.2) < 1e-9


def test_write_trace_summary_empty_list_no_file(tmp_path: Path) -> None:
    """Empty summaries list must not write anything (guard against crash on missing output)."""
    path = tmp_path / "summary.json"
    _write_trace_summary_artifact(path, summaries=[])
    assert not path.exists()


def test_write_trace_summary_zero_total_wall_does_not_divide_by_zero(
    tmp_path: Path,
) -> None:
    """Aggregate fractions when total_s sums to zero must be 0.0, not NaN/inf."""
    path = tmp_path / "summary.json"
    summaries = [
        {
            "trace_id": "t-1",
            "total_s": 0.0,
            "model_s": 0.0,
            "tool_s": 0.0,
            "model_time_fraction": 0.0,
            "tool_time_fraction": 0.0,
            "model_calls": 0,
            "tool_calls": 0,
        }
    ]
    _write_trace_summary_artifact(path, summaries=summaries)
    data = json.loads(path.read_bytes())
    agg = data["aggregate"]
    assert agg["model_time_fraction"] == 0.0
    assert agg["tool_time_fraction"] == 0.0


def test_write_trace_summary_aggregate_call_counts(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summaries = [
        {
            "trace_id": "t-1",
            "total_s": 5.0,
            "model_s": 4.0,
            "tool_s": 0.5,
            "model_time_fraction": 0.8,
            "tool_time_fraction": 0.1,
            "model_calls": 2,
            "tool_calls": 3,
        },
        {
            "trace_id": "t-2",
            "total_s": 3.0,
            "model_s": 1.0,
            "tool_s": 1.0,
            "model_time_fraction": 0.33,
            "tool_time_fraction": 0.33,
            "model_calls": 1,
            "tool_calls": 2,
        },
    ]
    _write_trace_summary_artifact(path, summaries=summaries)
    data = json.loads(path.read_bytes())
    agg = data["aggregate"]
    assert agg["model_calls"] == 3
    assert agg["tool_calls"] == 5


# ---------------------------------------------------------------------------
# Strategy accumulation via _record_trace_timing
# ---------------------------------------------------------------------------


class _Issuer:
    async def issue_graph_credit(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("no credit should be issued by these tests")


def _parsed_graph() -> ParsedGraph:
    return ParsedGraph(
        graph=GraphRecord(
            nodes={"n0": LlmNode(prompt=["hi"], output="n0_out")}, edges=[], state={}
        ),
        traces=[TraceRecord(id="t-1")],
    )


def _strategy(
    *,
    artifact_dir: Path | None = None,
) -> AgentGraphReplayStrategy:
    return AgentGraphReplayStrategy(
        config=CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
            artifact_dir=artifact_dir,
        ),
        credit_issuer=_Issuer(),
        parsed_graph=_parsed_graph(),
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )


def _result(
    *,
    trace_id: str = "t-1",
    llm_durations_s: list[float] | None = None,
    tool_durations_s: list[float] | None = None,
    trace_wall_s: float = 10.0,
) -> TraceResult:
    return TraceResult(
        trace_id=trace_id,
        channels={},
        llm_durations_s=llm_durations_s or [3.0, 2.0],
        tool_durations_s=tool_durations_s or [],
        trace_wall_s=trace_wall_s,
    )


def test_record_trace_timing_accumulates_summaries() -> None:
    strat = _strategy()
    assert strat._trace_summaries == []
    strat._record_trace_timing(
        _result(trace_id="t-1", llm_durations_s=[3.0, 2.0], trace_wall_s=10.0)
    )
    strat._record_trace_timing(
        _result(trace_id="t-2", llm_durations_s=[1.5], trace_wall_s=5.0)
    )
    assert len(strat._trace_summaries) == 2
    assert strat._trace_summaries[0]["trace_id"] == "t-1"
    assert strat._trace_summaries[1]["trace_id"] == "t-2"


def test_record_trace_timing_computes_correct_per_trace_model_s() -> None:
    strat = _strategy()
    strat._record_trace_timing(_result(llm_durations_s=[3.0, 2.0], trace_wall_s=10.0))
    s = strat._trace_summaries[0]
    assert abs(s["model_s"] - 5.0) < 1e-9
    assert abs(s["total_s"] - 10.0) < 1e-9
    assert abs(s["model_time_fraction"] - 0.5) < 1e-9


def test_record_trace_timing_uses_local_duration_when_worker_latency_is_missing() -> (
    None
):
    """Missing worker timing falls back to the measured dispatch duration."""
    strat = _strategy()
    strat._record_trace_timing(
        TraceResult(
            trace_id="t-1",
            channels={},
            llm_durations_s=[2.0, 4.0],
            llm_request_latency_s=[None, 4.0],
            llm_ttft_s=[None, 1.0],
            llm_target_osl=[None, 5],
            llm_observed_osl=[None, 3],
            trace_wall_s=6.0,
        )
    )

    normalized = strat._trace_summaries[0]["normalized_model_s"]
    assert normalized is not None
    assert abs(normalized - 9.0) < 1e-9


def test_record_trace_timing_includes_tool_durations() -> None:
    strat = _strategy()
    strat._record_trace_timing(
        _result(llm_durations_s=[2.0], tool_durations_s=[1.0, 0.5], trace_wall_s=8.0)
    )
    s = strat._trace_summaries[0]
    assert abs(s["tool_s"] - 1.5) < 1e-9
    assert s["tool_calls"] == 2
    assert abs(s["tool_time_fraction"] - 1.5 / 8.0) < 1e-9


def test_record_trace_timing_zero_wall_fractions_are_zero() -> None:
    strat = _strategy()
    strat._record_trace_timing(_result(llm_durations_s=[0.0], trace_wall_s=0.0))
    s = strat._trace_summaries[0]
    assert s["model_time_fraction"] == 0.0
    assert s["tool_time_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Warmup phase skips report_trace_summary
# ---------------------------------------------------------------------------


def test_report_trace_summary_skips_on_warmup_phase(tmp_path: Path) -> None:
    """report_trace_summary must be a no-op when _is_warmup_phase is True."""
    strat = _strategy(artifact_dir=tmp_path)
    # _is_warmup_phase is set by start_phase(); set directly to simulate warmup teardown.
    strat._is_warmup_phase = True
    # Simulate accumulation of a summary (should not happen in warmup, but guard
    # against the teardown race where warmup fires after profiling's teardown).
    strat._trace_summaries.append(
        {
            "trace_id": "warmup-t-1",
            "total_s": 1.0,
            "model_s": 0.8,
            "tool_s": 0.0,
            "model_time_fraction": 0.8,
            "tool_time_fraction": 0.0,
            "model_calls": 1,
            "tool_calls": 0,
        }
    )
    strat.report_trace_summary()
    # No artifact must be written by the warmup phase.
    artifacts = list(tmp_path.glob("profile_export_graph_trace_summary.json"))
    assert not artifacts


def test_report_trace_summary_writes_artifact_on_profiling_phase(
    tmp_path: Path,
) -> None:
    strat = _strategy(artifact_dir=tmp_path)
    strat._record_trace_timing(_result(llm_durations_s=[4.0], trace_wall_s=5.0))
    strat.report_trace_summary()
    artifact = tmp_path / "profile_export_graph_trace_summary.json"
    assert artifact.exists()
    data = json.loads(artifact.read_bytes())
    assert data["trace_count"] == 1


def test_report_trace_summary_skips_when_no_summaries(tmp_path: Path) -> None:
    strat = _strategy(artifact_dir=tmp_path)
    strat.report_trace_summary()
    artifact = tmp_path / "profile_export_graph_trace_summary.json"
    assert not artifact.exists()
