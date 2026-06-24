# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static edge tests for job-detail metric and artifact data shaping."""

from __future__ import annotations

from pathlib import Path

JOB_DETAIL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "job-detail.js"
)


def _source() -> str:
    return JOB_DETAIL_PATH.read_text()


def _assert_in_order(source: str, *needles: str) -> None:
    cursor = -1
    for needle in needles:
        pos = source.find(needle, cursor + 1)
        assert pos >= 0, needle
        cursor = pos


def test_polling_throughput_summary_prefers_live_summary_then_final_summary_then_live_metrics() -> (
    None
):
    source = _source()

    _assert_in_order(
        source,
        "function extractSummary(data)",
        "return status.liveSummary ?? status.summary ?? null;",
        "const summary = extractSummary(data);",
        "summary?.output_token_throughput?.avg ??",
        "data?.status?.liveMetrics?.metrics?.output_token_throughput?.avg ??",
    )


def test_rendered_metric_summary_prefers_results_before_live_metrics_before_summary_snapshots() -> (
    None
):
    source = _source()

    _assert_in_order(
        source,
        "const restSummary =",
        "status.results?.metrics ??",
        "status.liveMetrics?.metrics ??",
        "status.summary ??",
        "status.liveSummary ??",
        "const summary = (liveData.connected && Object.keys(liveData.summary).length > 0)",
        "? { ...restSummary, ...liveData.summary }",
        "const results = summary;",
    )


def test_archived_epoch_views_do_not_mix_current_live_run_data_into_artifact_view() -> (
    None
):
    source = _source()

    _assert_in_order(
        source,
        "const viewingCurrentRun = liveRunState.viewingCurrentRun;",
        "const wsActive = liveRunState.isRunning && viewingCurrentRun;",
        "const liveServerMetricsBase = viewingCurrentRun ? status.serverMetrics : null;",
        "${viewingCurrentRun && pods.length > 0 && html`",
        "mode=${viewingCurrentRun ? (isRunning ? 'live' : 'completed') : 'archived'}",
        "archived=${!viewingCurrentRun}",
    )


def test_latency_histogram_uses_second_buckets_and_formats_millisecond_labels_below_one_second() -> (
    None
):
    source = _source()

    _assert_in_order(
        source,
        "job?.status?.results?.latency_histogram",
        "job?.status?.results?.histograms?.request_latency",
        "Bucket upper bound ``le`` is in seconds.",
        "if (le >= 1) return le.toFixed(le >= 10 ? 0 : 1) + 's';",
        "return (le * 1000).toFixed(0) + 'ms';",
        "labels: buckets.map((b) => fmtBucket(b.le)),",
        "data: buckets.map((b) => b.count ?? b.value ?? 0),",
    )


def test_profile_jsonl_loading_is_artifact_gated_text_line_oriented_and_best_effort() -> (
    None
):
    source = _source()

    _assert_in_order(
        source,
        "if (fileList.some(f => f.name === 'profile_export.jsonl')) {",
        "fetch(`${resultsBase}/profile_export.jsonl`, { signal: ac.signal })",
        ".then(r => r.ok ? r.text() : null)",
        "const lines = text.split('\\n');",
        "const CHUNK = 2000;",
        "if (!line || line.trim() === '') continue;",
        "try { recs.push(JSON.parse(line)); } catch { /* skip bad line */ }",
        "setJsonlRecords(recs.length > 0 ? recs : null);",
        "setJsonlLoaded(true);",
    )


def test_server_metrics_fetch_and_loading_state_are_gated_by_artifact_listing() -> None:
    source = _source()

    _assert_in_order(
        source,
        "if (fileList.some(f => f.name === 'server_metrics_export.json')) {",
        "fetch(`${resultsBase}/server_metrics_export.json`, { signal: ac.signal })",
        "setServerMetrics(sm);",
        "setServerMetricsLoaded(true);",
        "Server metrics artifact could not be read.",
        ": (isTerminal && files.some(f => f.name === 'server_metrics_export.json') && !serverMetricsLoaded && html`",
        "Loading server metrics…",
    )


def test_live_server_metrics_fall_back_to_rest_snapshot_unless_websocket_summary_is_connected() -> (
    None
):
    source = _source()

    _assert_in_order(
        source,
        "const liveServerMetrics = (liveData.connected && liveData.serverSummary)",
        "? liveData.serverSummary",
        ": liveServerMetricsBase;",
        "const displayedServerMetrics = serverMetrics || liveServerMetrics;",
        "const serverMetricsSource = serverMetrics ? 'final' : 'live';",
    )
