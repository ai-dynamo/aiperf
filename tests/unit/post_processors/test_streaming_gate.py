# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-level STREAMING_ONLY gate relaxation for graph-IR workloads.

``BaseMetricsProcessor.get_filters`` disables every ``STREAMING_ONLY`` metric
(TTFT / TTST / ICL / TTFOT / stream-setup) when the global ``endpoint.streaming``
flag is off. Graph-IR replays (weka / dynamo / native graph) stream per-request
from recorded node modes even without the global flag, so the run-level gate
must NOT fire for them -- per-record applicability is enforced by the
``streamed_request`` predicate metric instead.

Every run here is a REAL resolved ``BenchmarkRun`` (native ``BenchmarkConfig`` /
the production ``resolve_config`` path), never MagicMock, so a wrong attribute
path in the graph-workload predicate cannot be silently auto-created (the
repo's MagicMock-path-drift trap).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


def _write_minimal_native_graph(tmp_path: Path) -> Path:
    """Write a minimal valid native graph JSONL (one LLM node + one trace).

    Native is auto-detect-EXCLUDED, so this file is only treated as a graph
    workload when ``--graph-format native`` forces it -- exactly the resolved
    ``FileDataset.graph_format`` the predicate reads.
    """
    f = tmp_path / "native_min.jsonl"
    f.write_text(
        '{"kind": "graph", "nodes": '
        '{"a": {"node_type": "llm", '
        '"prompt": [{"role": "user", "content": "hi"}], "output": "out"}}}\n'
        '{"kind": "trace", "id": "t1"}\n'
    )
    return f


def _graph_run(tmp_path: Path):
    """Real resolved graph-workload ``BenchmarkRun`` (native --graph-format)."""
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(_write_minimal_native_graph(tmp_path)),
        graph_format="native",
    )
    return make_run_from_cli(cfg)


class TestStreamingOnlyGate:
    """Gate matrix for MetricFlags.STREAMING_ONLY over (global flag, workload)."""

    def test_global_on_non_graph_allows_streaming(self, mock_run) -> None:
        # Synthetic (non-graph) dataset with the global flag on: unchanged.
        mock_run.cfg.endpoint.streaming = True
        _, disallowed = BaseMetricsProcessor(mock_run).get_filters()
        assert not (disallowed & MetricFlags.STREAMING_ONLY)

    def test_global_off_non_graph_disallows_streaming(self, mock_run) -> None:
        # Synthetic (non-graph) dataset with the global flag off: the gate still
        # fires (nothing in the run can stream). Unchanged behavior.
        mock_run.cfg.endpoint.streaming = False
        _, disallowed = BaseMetricsProcessor(mock_run).get_filters()
        assert disallowed & MetricFlags.STREAMING_ONLY

    def test_global_on_graph_allows_streaming(self, tmp_path) -> None:
        run = _graph_run(tmp_path)
        run.cfg.endpoint.streaming = True
        _, disallowed = BaseMetricsProcessor(run).get_filters()
        assert not (disallowed & MetricFlags.STREAMING_ONLY)

    def test_global_off_graph_allows_streaming(self, tmp_path) -> None:
        # NEW: graph-IR workloads stream per-request, so the run-level gate must
        # not fire even with the global flag off.
        run = _graph_run(tmp_path)
        run.cfg.endpoint.streaming = False
        _, disallowed = BaseMetricsProcessor(run).get_filters()
        assert not (disallowed & MetricFlags.STREAMING_ONLY)

    def test_predicate_true_for_graph_false_for_synthetic(
        self, mock_run, tmp_path
    ) -> None:
        # Guards against MagicMock-path drift: the predicate reads real resolved
        # config, so it must distinguish an actual graph workload from a
        # synthetic one rather than auto-creating a truthy attribute path.
        assert BaseMetricsProcessor(mock_run)._is_graph_workload() is False
        assert BaseMetricsProcessor(_graph_run(tmp_path))._is_graph_workload() is True


class TestEmptyStreamedSubset:
    """A graph run whose records are all non-streamed leaves TTFT absent."""

    @pytest.mark.asyncio
    async def test_all_non_streamed_records_leave_ttft_absent(self, tmp_path) -> None:
        """With the gate relaxed, TTFT is applicable but the per-record streamed
        predicate never fires, so no record carries the TTFT tag; the accumulator
        must leave TTFT absent without raising or warning."""
        from aiperf.metrics.accumulator import MetricsAccumulator
        from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
        from aiperf.metrics.types.ttft_metric import TTFTMetric
        from tests.unit.post_processors.conftest import create_metric_records_data

        accumulator = MetricsAccumulator(_graph_run(tmp_path))

        for i in range(3):
            msg = create_metric_records_data(
                x_request_id=f"r-{i}",
                request_start_ns=1_000_000_000 + i,
                results=[{RequestLatencyMetric.tag: 42.0}],
            )
            await accumulator.process_record(msg)

        summary = await accumulator.summarize()
        assert RequestLatencyMetric.tag in summary.results
        assert TTFTMetric.tag not in summary.results
