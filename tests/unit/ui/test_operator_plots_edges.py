# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for operator UI plot/chart and artifact surfaces."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_APP_PATH = _UI_ROOT / "app.js"
_TOP_NAV_PATH = _UI_ROOT / "components" / "top-nav.js"
_ARTIFACTS_CARD_PATH = _UI_ROOT / "components" / "artifacts-card.js"
_VARIATIONS_CHART_PATH = _UI_ROOT / "components" / "variations-chart.js"
_METRIC_SELECTOR_PATH = _UI_ROOT / "components" / "metric-selector.js"
_API_PATH = _UI_ROOT / "lib" / "api.js"
_JOB_DETAIL_STATE_PATH = _UI_ROOT / "pages" / "job-detail-state.js"
_JOB_DETAIL_PATH = _UI_ROOT / "pages" / "job-detail.js"
_SWEEP_DETAIL_PATH = _UI_ROOT / "pages" / "sweep-detail.js"
_PLOTS_PAGE_PATH = _UI_ROOT / "pages" / "plots.js"


def _source(path: Path) -> str:
    return path.read_text()


def _artifacts_helper_script(body: str) -> str:
    return f"""
        import fs from 'node:fs';
        const palette = {{}};
        const source = fs.readFileSync({json.dumps(str(_ARTIFACTS_CARD_PATH))}, 'utf8');
        // Anchored on the first helper the body actually calls. The old anchor
        // (`function fileColor`) was renamed away, and `indexOf` returning -1
        // silently sliced the wrong window instead of failing.
        const start = source.indexOf('function resultFileUrl');
        if (start < 0) throw new Error('artifacts-card.js no longer defines resultFileUrl');
        const helpers = source.slice(start, source.indexOf('export function ArtifactsCard'));
        eval(helpers + '\\n' + {json.dumps(body)});
    """


def test_plots_navigation_is_feature_gated_external_dashboard_link() -> None:
    """The Plots entry should not be routed as an in-app page."""
    app_source = _source(_APP_PATH)
    top_nav_source = _source(_TOP_NAV_PATH)

    assert not _PLOTS_PAGE_PATH.exists()
    assert "currentRoute === '/plots'" not in app_source
    assert "path: '/dashboard/'" in top_nav_source
    assert "features && features.dashboard_enabled" in top_nav_source
    assert "label: 'Plots ↗'" in top_nav_source
    assert 'target="_blank"' in top_nav_source
    assert 'rel="noopener noreferrer"' in top_nav_source
    assert (
        "data-testid=${item.testId || ('nav-link-' + routeSlug(item.path))}"
        in top_nav_source
    )


def test_plot_artifact_file_urls_encode_archive_inputs_and_nested_paths() -> None:
    """Artifact links must preserve run epoch and nested plot filenames safely."""
    script = _artifacts_helper_script(
        """
        console.log(JSON.stringify({
          jobPlot: resultFileUrl('ns a', 'job/one', '1700000000', 'plots/chart 1#p.png'),
          nestedJson: resultFileUrl('team', 'bench', '42', 'reports/plot config.json'),
          waiting: selectedEmptyKey({ resolvedEpoch: null, isCompleted: true, isRunning: false }),
          completed: selectedEmptyKey({ resolvedEpoch: '42', isCompleted: true, isRunning: false }),
          running: selectedEmptyKey({ resolvedEpoch: '42', isCompleted: false, isRunning: true }),
          unavailable: selectedEmptyKey({ resolvedEpoch: '42', isCompleted: false, isRunning: false }),
        }));
        """
    )

    assert json.loads(run_node(script)) == {
        "jobPlot": "/api/v1/results/ns%20a/job%2Fone/runs/1700000000/plots/chart%201%23p.png",
        "nestedJson": "/api/v1/results/team/bench/runs/42/reports/plot%20config.json",
        "waiting": "waiting",
        "completed": "completed",
        "running": "running",
        "unavailable": "unavailable",
    }


def test_sweep_aggregate_artifact_urls_are_epoch_scoped_and_encoded() -> None:
    """Sweep plot exports should never fall back to unpinned latest artifacts."""
    source = _source(_API_PATH)

    assert "sweepArtifactListUrl(ns, sweepName, epoch)" in source
    assert "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts`" in source
    assert "sweepArtifactBundleUrl(ns, sweepName, epoch)" in source
    assert (
        "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts.zip`" in source
    )
    assert "sweepArtifactFileUrl(ns, sweepName, epoch, filename)" in source
    assert (
        "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts/${fileSeg}`"
        in source
    )
    assert "sweepProfileExportUrl(ns, sweepName, epoch, format = 'json')" in source
    assert "profile_export?format=${formatSeg}" in source
    # Nested artifact paths (e.g. ``plots/foo.png``) keep their ``/`` separators
    # but each segment is encoded individually so unsafe chars never break out.
    assert "filename.split('/').map(encodeURIComponent).join('/')" in source


def test_metric_selection_defaults_fallbacks_and_url_elision_are_explicit() -> None:
    """Chart metric selectors should survive unknown metric params and clean defaults."""
    metric_selector_source = _source(_METRIC_SELECTOR_PATH)
    sweep_source = _source(_SWEEP_DETAIL_PATH)

    assert (
        "const metric = value?.metric ?? 'request_throughput'" in metric_selector_source
    )
    assert "const stat = value?.stat ?? 'avg'" in metric_selector_source
    # The selector validates the chosen option against its known METRICS/STATS
    # lists before emitting, so unknown ``e.target.value`` params can't leak out.
    assert "onSelect({ metric: selectedMetric.value, stat })" in metric_selector_source
    assert "onSelect({ metric, stat: selectedStat.value })" in metric_selector_source

    assert (
        "const DEFAULT_CHART_METRIC_KEY = 'output_token_throughput.avg'" in sweep_source
    )
    assert (
        "const urlMetric = query.value.metric ?? DEFAULT_CHART_METRIC_KEY"
        in sweep_source
    )
    assert (
        "HEADLINE_METRICS.find(x => x.key + '.' + x.stat === chartMetricKey)"
        in sweep_source
    )
    assert "?? HEADLINE_METRICS[0]" in sweep_source
    assert (
        "setQuery({ metric: e.target.value === DEFAULT_CHART_METRIC_KEY ? undefined : e.target.value })"
        in sweep_source
    )


def test_archived_run_state_disables_live_plot_inputs() -> None:
    """Pinned archived runs must not consume the current run websocket stream."""
    script = f"""
        import {{ deriveJobRunState }} from {_JOB_DETAIL_STATE_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          archivedPast: deriveJobRunState({{ phase: 'Running', epoch: '100', runEpoch: '200' }}),
          archivedTerminal: deriveJobRunState({{ phase: 'Archived', epoch: '100', runEpoch: '200' }}),
          liveCurrent: deriveJobRunState({{ phase: 'Running', epoch: '200', runEpoch: '200' }}),
        }}));
    """

    result = json.loads(run_node(script))

    assert result["archivedPast"]["viewingCurrentRun"] is False
    assert result["archivedPast"]["isRunning"] is True
    assert result["archivedPast"]["showLiveRunPanels"] is True
    assert result["archivedTerminal"]["isArchived"] is True
    assert result["archivedTerminal"]["pollingDone"] is True
    assert result["archivedTerminal"]["showLiveRunPanels"] is False
    assert result["liveCurrent"]["viewingCurrentRun"] is True


def test_detail_pages_pass_archived_epoch_inputs_to_artifact_and_plot_fetches() -> None:
    """Archived pages should fetch data for the selected epoch, not latest."""
    job_source = _source(_JOB_DETAIL_PATH)
    sweep_source = _source(_SWEEP_DETAIL_PATH)

    assert "const resultsBase = epoch\n    ? `/api/v1/results/" in job_source
    assert "fetch(resultsBase, { signal: ac.signal })" in job_source
    assert (
        "api.fetchRunRequests(namespace, name, epoch, perRecordFilename)" in job_source
    )
    assert (
        "<${LatencyTimelineChart} records=${jsonlRecords} loading=${!jsonlLoaded} skipped=${jsonlError} />"
        in job_source
    )
    assert "resolvedEpoch=${resolvedEpoch}" in job_source

    assert "api.getSweepCells(namespace, name, epoch)" in sweep_source
    assert "api.getSweepChildren(namespace, name, epoch)" in sweep_source
    assert (
        "fetch(api.sweepArtifactListUrl(namespace, name, resolvedEpoch), { signal: ac.signal })"
        in sweep_source
    )
    assert (
        "bundleUrl=${resolvedEpoch != null ? api.sweepArtifactBundleUrl(namespace, name, resolvedEpoch) : null}"
        in sweep_source
    )
    assert (
        "quickExportUrl=${resolvedEpoch != null ? api.sweepProfileExportUrl(namespace, name, resolvedEpoch, 'json') : null}"
        in sweep_source
    )
    assert (
        "? api.sweepArtifactFileUrl(namespace, name, resolvedEpoch, fileName)"
        in sweep_source
    )


def test_empty_and_error_states_for_missing_plot_artifacts_are_visible() -> None:
    """Missing plot/result artifacts and preview failures should render explicit states."""
    artifacts_source = _source(_ARTIFACTS_CARD_PATH)
    variations_source = _source(_VARIATIONS_CHART_PATH)
    job_source = _source(_JOB_DETAIL_PATH)
    sweep_source = _source(_SWEEP_DETAIL_PATH)

    assert "Waiting for a run epoch before showing result files." in artifacts_source
    assert "No result files persisted for this run." in artifacts_source
    assert "No result files yet." in artifacts_source
    assert "Looking up result files…" in artifacts_source
    assert "Unable to load preview for ${filename}: ${details}" in artifacts_source
    assert 'data-testid="artifacts-empty"' in artifacts_source

    assert "if (means.every(m => m == null)) return null" in variations_source
    assert 'data-testid="variations-chart-empty"' in variations_source
    assert (
        "No ${metricLabel || 'variation'} data available for any variation yet."
        in variations_source
    )

    assert 'data-testid="job-detail-error"' in job_source
    assert "Failed to load job" in job_source
    assert 'data-testid="page-sweep-detail"' in sweep_source
    assert "<strong>Error:</strong> ${error}" in sweep_source
