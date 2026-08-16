# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ConsoleGraphMetricsExporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from aiperf.exporters.console_graph_metrics_exporter import ConsoleGraphMetricsExporter


def _make_exporter(artifact_dir: Path) -> ConsoleGraphMetricsExporter:
    cfg = MagicMock()
    cfg.artifacts.artifact_directory = artifact_dir
    ec = MagicMock()
    ec.cfg = cfg
    return ConsoleGraphMetricsExporter(ec)


def _write_summary(
    path: Path,
    *,
    traces: list[dict[str, Any]],
) -> None:
    total_s = sum(t["total_s"] for t in traces)
    model_s = sum(t["model_s"] for t in traces)
    tool_s = sum(t["tool_s"] for t in traces)
    norm_values = [
        t["normalized_model_s"]
        for t in traces
        if t.get("normalized_model_s") is not None
    ]
    agg_normalized = sum(norm_values) if norm_values else None
    agg_osl_warnings = sum(t.get("low_osl_model_calls", 0) for t in traces)
    payload = {
        "trace_count": len(traces),
        "aggregate": {
            "total_s": total_s,
            "model_s": model_s,
            "tool_s": tool_s,
            "model_time_fraction": model_s / total_s if total_s else 0.0,
            "tool_time_fraction": tool_s / total_s if total_s else 0.0,
            "model_calls": sum(t["model_calls"] for t in traces),
            "tool_calls": sum(t["tool_calls"] for t in traces),
            "normalized_model_s": agg_normalized,
            "total_osl_warnings": agg_osl_warnings,
        },
        "traces": traces,
    }
    path.write_text(json.dumps(payload))


def _write_tool(path: Path, *, durations: list[float], backend: str) -> None:
    import statistics

    path.write_text(
        json.dumps(
            {
                "command_count": len(durations),
                "trace_count": 1,
                "backend": backend,
                "total_s": sum(durations),
                "mean_s": statistics.mean(durations),
                "median_s": statistics.median(sorted(durations)),
                "max_s": max(durations),
                "durations_s": durations,
            }
        )
    )


@pytest.mark.asyncio
async def test_exporter_silently_skips_when_no_artifact(tmp_path: Path) -> None:
    """No graph artifact → no output, no error."""
    exporter = _make_exporter(tmp_path)
    captured: list[str] = []
    console = Console(
        file=type(
            "_F",
            (),
            {"write": lambda s, t: captured.append(t), "flush": lambda s: None},
        )()
    )  # noqa: E501
    # Should not raise.
    await exporter.export(console)
    assert captured == []


@pytest.mark.asyncio
async def test_exporter_renders_aggregate_table(tmp_path: Path) -> None:
    """Aggregate table appears with correct rows."""
    _write_summary(
        tmp_path / "profile_export_graph_trace_summary.json",
        traces=[
            {
                "trace_id": "t-1::abc",
                "total_s": 2.0,
                "model_s": 1.5,
                "tool_s": 0.5,
                "sandbox_setup_s": 0.0,
                "model_time_fraction": 0.75,
                "tool_time_fraction": 0.25,
                "model_calls": 3,
                "tool_calls": 6,
                "normalized_model_s": None,
                "low_osl_model_calls": 0,
            }
        ],
    )

    exporter = _make_exporter(tmp_path)
    console = Console(record=True, width=120)
    await exporter.export(console)
    text = console.export_text()

    assert "Agent Graph Replay" in text
    assert "1" in text  # trace count
    assert "2.000s" in text  # total
    assert "1.500s" in text  # model
    assert "0.500s" in text  # tool
    assert "75.0%" in text or "75.0" in text
    assert "3" in text  # llm calls
    assert "6" in text  # tool calls


@pytest.mark.asyncio
async def test_exporter_renders_tool_backend_and_latency(tmp_path: Path) -> None:
    """Tool artifact rows appear when profile_export_graph_tool_time.json is present."""
    _write_summary(
        tmp_path / "profile_export_graph_trace_summary.json",
        traces=[
            {
                "trace_id": "t-1::abc",
                "total_s": 1.0,
                "model_s": 0.8,
                "tool_s": 0.2,
                "sandbox_setup_s": 0.0,
                "model_time_fraction": 0.8,
                "tool_time_fraction": 0.2,
                "model_calls": 2,
                "tool_calls": 3,
                "normalized_model_s": None,
                "low_osl_model_calls": 0,
            }
        ],
    )
    _write_tool(
        tmp_path / "profile_export_graph_tool_time.json",
        durations=[0.06, 0.07, 0.07],
        backend="docker:my-task:latest",
    )

    exporter = _make_exporter(tmp_path)
    console = Console(record=True, width=120)
    await exporter.export(console)
    text = console.export_text()

    assert "docker:my-task:latest" in text
    assert "Tool cmd latency" in text


@pytest.mark.asyncio
async def test_exporter_omits_per_trace_table_for_single_trace(tmp_path: Path) -> None:
    """Per-trace breakdown is suppressed when there is only one trace."""
    _write_summary(
        tmp_path / "profile_export_graph_trace_summary.json",
        traces=[
            {
                "trace_id": "t-1::abc",
                "total_s": 1.0,
                "model_s": 0.9,
                "tool_s": 0.1,
                "sandbox_setup_s": 0.0,
                "model_time_fraction": 0.9,
                "tool_time_fraction": 0.1,
                "model_calls": 2,
                "tool_calls": 1,
                "normalized_model_s": None,
                "low_osl_model_calls": 0,
            }
        ],
    )
    exporter = _make_exporter(tmp_path)
    console = Console(record=True, width=120)
    await exporter.export(console)
    text = console.export_text()

    assert "Per-Trace Breakdown" not in text


@pytest.mark.asyncio
async def test_exporter_shows_per_trace_table_for_multiple_traces(
    tmp_path: Path,
) -> None:
    """Per-trace breakdown appears when there are 2+ traces."""
    traces = [
        {
            "trace_id": f"t-{i}::abc",
            "total_s": 1.0,
            "model_s": 0.8,
            "tool_s": 0.2,
            "sandbox_setup_s": 0.0,
            "model_time_fraction": 0.8,
            "tool_time_fraction": 0.2,
            "model_calls": 2,
            "tool_calls": 3,
            "normalized_model_s": None,
            "low_osl_model_calls": 0,
        }
        for i in range(3)
    ]
    _write_summary(tmp_path / "profile_export_graph_trace_summary.json", traces=traces)
    exporter = _make_exporter(tmp_path)
    console = Console(record=True, width=160)
    await exporter.export(console)
    text = console.export_text()

    assert "Per-Trace Breakdown" in text
    assert "t-0::abc" in text
    assert "t-2::abc" in text


@pytest.mark.asyncio
async def test_exporter_renders_normalized_model_time(tmp_path: Path) -> None:
    """Norm. model time row appears when normalized_model_s is present."""
    _write_summary(
        tmp_path / "profile_export_graph_trace_summary.json",
        traces=[
            {
                "trace_id": "t-1::abc",
                "total_s": 10.0,
                "model_s": 4.0,
                "tool_s": 6.0,
                "sandbox_setup_s": 0.0,
                "model_time_fraction": 0.4,
                "tool_time_fraction": 0.6,
                "model_calls": 2,
                "tool_calls": 4,
                # observed 50 tokens, target 200 → ratio 4.0 → normalized = 4.0 * 4.0 = 16.0
                "normalized_model_s": 16.0,
                "low_osl_model_calls": 2,
            }
        ],
    )
    exporter = _make_exporter(tmp_path)
    console = Console(record=True, width=120)
    await exporter.export(console)
    text = console.export_text()

    assert "Norm. model time" in text
    assert "16.000s" in text
    assert "OSL warnings" in text
    assert "2" in text
