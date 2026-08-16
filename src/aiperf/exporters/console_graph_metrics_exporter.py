# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Console exporter for agent-graph-replay trace timing breakdown.

Reads ``profile_export_graph_trace_summary.json`` (and optionally
``profile_export_graph_tool_time.json``) from the artifact directory.
Self-disables silently when those files are absent (non-graph run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import rich.box
from rich.console import Console, Group, RenderableType
from rich.table import Table

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleGraphMetricsExporter(AIPerfLoggerMixin):
    """Console exporter for agent-graph-replay trace timing breakdown.

    Rendered only when the run produced ``profile_export_graph_trace_summary.json``,
    which the strategy writes after every profiling phase that executed at least
    one graph trace. Non-graph runs produce no file and the exporter silently
    returns without output.

    Two Rich tables are printed:

    * **Aggregate** — headline totals: trace count, wall / model / tool / sandbox
      time, LLM and tool call counts, and the tool backend.
    * **Per-trace** — one row per trace instance, shown when the run has at
      least two traces.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._artifact_dir: Path = exporter_config.cfg.artifacts.artifact_directory

    async def export(self, console: Console) -> None:
        summary_path = self._artifact_dir / "profile_export_graph_trace_summary.json"
        if not summary_path.exists():
            return

        data: dict[str, Any] = orjson.loads(summary_path.read_bytes())
        tool_path = self._artifact_dir / "profile_export_graph_tool_time.json"
        tool_data: dict[str, Any] | None = (
            orjson.loads(tool_path.read_bytes()) if tool_path.exists() else None
        )

        renderables: list[RenderableType] = [self._aggregate_table(data, tool_data)]

        traces: list[dict[str, Any]] = data.get("traces", [])
        if len(traces) > 1:
            renderables.append(Table.grid())  # blank spacer
            renderables.append(self._per_trace_table(traces))

        console.print("\n")
        console.print(Group(*renderables))
        console.file.flush()

    # ------------------------------------------------------------------
    # Table builders
    # ------------------------------------------------------------------

    def _aggregate_table(
        self,
        data: dict[str, Any],
        tool_data: dict[str, Any] | None,
    ) -> Table:
        agg = data.get("aggregate", {})
        traces: list[dict[str, Any]] = data.get("traces", [])
        trace_count: int = data.get("trace_count", len(traces))

        total_s: float = agg.get("total_s", 0.0)
        model_s: float = agg.get("model_s", 0.0)
        tool_s: float = agg.get("tool_s", 0.0)
        model_calls: int = agg.get("model_calls", 0)
        tool_calls: int = agg.get("tool_calls", 0)
        model_frac: float = model_s / total_s if total_s > 0 else 0.0
        tool_frac: float = tool_s / total_s if total_s > 0 else 0.0

        backend: str = tool_data.get("backend", "local") if tool_data else "local"

        tbl = Table(
            title="NVIDIA AIPerf | Agent Graph Replay",
            show_header=False,
            box=rich.box.HEAVY_EDGE,
        )
        tbl.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        tbl.add_column("Value", justify="left", style="green")

        tbl.add_row("Traces", str(trace_count))
        tbl.add_row("Total wall time", _fmt_s(total_s))
        tbl.add_row("Model time", f"{_fmt_s(model_s)}  ({model_frac * 100:.1f}%)")
        norm_model_s: float | None = agg.get("normalized_model_s")
        if norm_model_s is not None:
            norm_frac = norm_model_s / total_s if total_s > 0 else 0.0
            tbl.add_row(
                "Norm. model time", f"{_fmt_s(norm_model_s)}  ({norm_frac * 100:.1f}%)"
            )
        tbl.add_row("Tool time", f"{_fmt_s(tool_s)}  ({tool_frac * 100:.1f}%)")
        tbl.add_row("LLM calls", str(model_calls))
        tbl.add_row("Tool calls", str(tool_calls))
        osl_warnings: int = agg.get("total_osl_warnings", 0)
        if osl_warnings:
            tbl.add_row("OSL warnings", str(osl_warnings))
        if tool_data:
            tbl.add_row("Tool backend", backend)
            cmd_mean = tool_data.get("mean_s")
            cmd_max = tool_data.get("max_s")
            if cmd_mean is not None:
                tbl.add_row(
                    "Tool cmd latency (mean / max)",
                    f"{_fmt_s(cmd_mean)} / {_fmt_s(cmd_max)}",
                )

        return tbl

    def _per_trace_table(self, traces: list[dict[str, Any]]) -> Table:
        tbl = Table(
            title="NVIDIA AIPerf | Agent Graph Replay — Per-Trace Breakdown",
        )
        tbl.add_column("Trace", justify="left", style="cyan", no_wrap=True)
        tbl.add_column("Total (s)", justify="right", style="green")
        tbl.add_column("Model (s)", justify="right", style="green")
        tbl.add_column("Norm. Model (s)", justify="right", style="yellow")
        tbl.add_column("Model%", justify="right", style="green")
        tbl.add_column("Tool (s)", justify="right", style="green")
        tbl.add_column("Tool%", justify="right", style="green")
        tbl.add_column("LLM", justify="right", style="green")
        tbl.add_column("Tools", justify="right", style="green")

        for t in traces:
            total = t.get("total_s", 0.0)
            model = t.get("model_s", 0.0)
            tool = t.get("tool_s", 0.0)
            model_pct = (model / total * 100) if total > 0 else 0.0
            tool_pct = (tool / total * 100) if total > 0 else 0.0
            norm = t.get("normalized_model_s")
            norm_str = f"{norm:.3f}" if norm is not None else "N/A"

            tbl.add_row(
                t.get("trace_id", ""),
                f"{total:.3f}",
                f"{model:.3f}",
                norm_str,
                f"{model_pct:.1f}",
                f"{tool:.3f}",
                f"{tool_pct:.1f}",
                str(t.get("model_calls", 0)),
                str(t.get("tool_calls", 0)),
            )

        return tbl


def _fmt_s(seconds: float | None) -> str:
    """Format a duration in seconds with 3 decimal places."""
    if seconds is None:
        return "N/A"
    return f"{seconds:.3f}s"
