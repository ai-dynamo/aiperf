# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static coverage for operator UI loading, error, and empty states."""

from __future__ import annotations

from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
PAGES = UI_ROOT / "pages"


def _source(page: str) -> str:
    return (PAGES / page).read_text()


def test_dashboard_blocks_cold_empty_cluster_with_loading_and_error_states() -> None:
    source = _source("dashboard.js")

    assert "firstJobsLoad" in source
    assert "Loading dashboard…" in source
    assert 'testid="dashboard-loading"' in source
    assert "jobsError" in source
    assert "Failed to load jobs" in source
    assert "dashboard-jobs-error" in source
    assert "No benchmarks yet" in source
    assert "Submit your first benchmark" in source
    assert "No active jobs." in source
    assert "completed run" in source


def test_jobs_page_distinguishes_loading_error_real_empty_and_filtered_empty() -> None:
    source = _source("jobs.js")

    assert "firstLoad" in source
    assert "Loading jobs…" in source
    assert "jobs-loading" in source
    assert "loadError" in source
    assert "Failed to load jobs:" in source
    assert "jobs-error" in source
    assert "filtered.length === 0 && localJobs.length === 0" in source
    assert "jobs-empty-real" in source
    assert "No jobs yet." in source
    assert "filtered.length === 0 && localJobs.length > 0" in source
    assert "jobs-empty-filtered" in source
    assert "No jobs match these filters." in source
    assert "Clear filters" in source


def test_sweeps_page_distinguishes_loading_error_real_empty_and_filtered_empty() -> (
    None
):
    source = _source("sweeps.js")

    assert "firstLoad" in source
    assert "Loading sweeps…" in source
    assert "sweeps-loading" in source
    assert "loadError" in source
    assert "Failed to load sweeps:" in source
    assert "sweeps-error" in source
    assert "hasFilters" in source
    assert "sweeps-empty-filtered" in source
    assert "No sweeps match these filters." in source
    assert "sweeps-empty-real" in source
    assert "No sweeps yet." in source


def test_leaderboard_page_has_metric_loading_error_real_empty_and_filtered_empty_copy() -> (
    None
):
    source = _source("leaderboard.js")

    assert "setLoading(true)" in source
    assert "Loading leaderboard…" in source
    assert "leaderboard-loading" in source
    assert "Failed to load leaderboard:" in source
    assert "entries.length === 0" in source
    assert "No completed benchmarks yet." in source
    assert "No results match the current filters." in source
    assert "Clear filters" in source


def test_history_page_has_metric_loading_error_real_empty_and_filtered_empty_copy() -> (
    None
):
    source = _source("history.js")

    assert "setLoading(true)" in source
    assert "Loading history…" in source
    assert "history-loading" in source
    assert "Failed to load history:" in source
    assert "entries.length === 0" in source
    assert "No completed benchmarks yet." in source
    assert "No data points match the current filters." in source
    assert "clear the model/endpoint" in source


def test_compare_page_covers_selector_and_results_empty_loading_error_states() -> None:
    source = _source("compare.js")

    assert "jobsLoading" in source
    assert "Loading jobs…" in source
    assert "compare-jobs-loading" in source
    assert "jobsError" in source
    assert "No completed jobs found." in source
    assert (
        "No completed jobs yet. Run an AIPerfJob to populate the comparison list."
        in source
    )
    assert "Select 2 or more jobs from the list to compare them." in source
    assert "Select at least one more job" in source
    assert "Running comparison…" in source
    assert "compare-running" in source
    assert "Compare failed:" in source
    assert "No comparable metrics returned for the selected runs." in source
    assert "compare-no-entries" in source
    assert "Cluster has no stored runs" in source
    assert (
        "No clusters visible. Toggle a series above to bring runs back into the chart."
        in source
    )


def test_compare_selector_distinguishes_real_empty_from_filtered_empty_copy() -> None:
    source = _source("compare.js")

    assert "storedJobs.length === 0" in source
    assert "filtered.length === 0 && storedJobs.length > 0" in source
    assert "No completed jobs match these filters." in source


def test_job_detail_has_initial_error_loading_artifact_loading_and_stale_data_resets() -> (
    None
):
    source = _source("job-detail.js")

    assert "!job && !error" in source
    assert "Loading " in source
    assert "job-detail-loading" in source
    assert "Failed to load job" in source
    assert "job-detail-error" in source
    assert "setFiles([]);" in source
    assert "setFilesLoaded(false);" in source
    assert "setServerMetrics(null);" in source
    assert "setServerMetricsError(null);" in source
    assert "setJsonlRecords(null);" in source
    assert "setJsonlError(null);" in source
    assert "setJobConfig(null);" in source
    assert "Clear stale live state" in source
    assert "Loading server metrics…" in source
    assert "Loading job configuration…" in source
    assert "Loading per-request records…" in source
    assert "Parsing per-request records" in source
    assert "Per-request records could not be read." in source


def test_sweep_detail_has_initial_error_loading_pending_child_and_artifact_states() -> (
    None
):
    source = _source("sweep-detail.js")

    assert "if (error)" in source
    assert "<strong>Error:</strong>" in source
    assert "if (!detail)" in source
    assert "Loading sweep " in source
    assert "setArtifactFiles([]);" in source
    assert "setArtifactFilesLoaded(false);" in source
    assert "sweep-detail-live-stale" in source
    assert "Live updates paused" in source
    assert (
        "waiting: 'Waiting for a sweep epoch before showing aggregate artifacts.'"
        in source
    )
    assert (
        "completed: 'No aggregate artifacts available for this sweep epoch.'" in source
    )
    assert "running: 'No aggregate artifacts yet.'" in source
    assert "Sweep is being initialized — children will appear here shortly." in source
    assert "No children persisted for this epoch yet." in source


def test_launch_page_has_no_cold_loading_state_but_covers_submit_parse_error_http_error_and_success() -> (
    None
):
    source = _source("launch.js")

    assert "useState(() => buildTemplates())" in source
    assert "sessionStorage.removeItem('aiperf.launch.prefill')" in source
    assert "setState({ kind: 'submitting' })" in source
    assert "Launching…" in source
    assert "Created <" in source
    assert "launch-success" in source
    assert "launch-parse-err" in source
    assert "YAML · ${peek.parseError}" in source
    assert "launch-err" in source
    assert "HTTP ${state.status}" in source
