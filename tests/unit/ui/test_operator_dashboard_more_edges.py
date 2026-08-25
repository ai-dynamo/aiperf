# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for dashboard pure helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

DASHBOARD_HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "dashboard-helpers.js"
)


def test_recent_jobs_filters_terminal_error_phases_and_sorts_newest_first() -> None:
    script = f"""
        import {{ recentJobs }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        const jobs = [
          {{ name: 'running-newest', phase: 'Running', created: '2026-05-18T13:00:00Z' }},
          {{ name: 'completed-mid', phase: 'Completed', created: '2026-05-18T11:00:00Z' }},
          {{ name: 'failed-newest', phase: 'Failed', created: '2026-05-18T12:00:00Z' }},
          {{ name: 'pending-newer', phase: 'Pending', created: '2026-05-18T12:30:00Z' }},
          {{ name: 'error-oldest', phase: 'Error', created: '2026-05-18T10:00:00Z' }},
          {{ name: 'succeeded-old', phase: 'Succeeded', created: '2026-05-18T09:00:00Z' }},
        ];
        console.log(JSON.stringify(recentJobs(jobs).map(j => j.name)));
    """

    assert json.loads(run_node(script)) == [
        "failed-newest",
        "completed-mid",
        "error-oldest",
        "succeeded-old",
    ]


def test_recent_jobs_honors_limit_after_sorting() -> None:
    script = f"""
        import {{ recentJobs }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        const jobs = [
          {{ name: 'old-1', phase: 'Completed', created: '2026-05-18T09:00:00Z' }},
          {{ name: 'new-1', phase: 'Completed', created: '2026-05-18T13:00:00Z' }},
          {{ name: 'old-2', phase: 'Failed', created: '2026-05-18T08:00:00Z' }},
          {{ name: 'new-2', phase: 'Error', created: '2026-05-18T12:00:00Z' }},
        ];
        console.log(JSON.stringify(recentJobs(jobs, 2).map(j => j.name)));
    """

    assert json.loads(run_node(script)) == ["new-1", "new-2"]


def test_recent_jobs_uses_start_and_completion_fallback_timestamps() -> None:
    script = f"""
        import {{ recentJobs }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        const jobs = [
          {{ name: 'completion-fallback', phase: 'Completed', completionTime: '2026-05-18T12:00:00Z' }},
          {{ name: 'start-fallback', phase: 'Failed', startTime: '2026-05-18T13:00:00Z' }},
          {{ name: 'created-wins', phase: 'Error', created: '2026-05-18T11:00:00Z', startTime: '2026-05-18T14:00:00Z' }},
        ];
        console.log(JSON.stringify(recentJobs(jobs).map(j => j.name)));
    """

    assert json.loads(run_node(script)) == [
        "start-fallback",
        "completion-fallback",
        "created-wins",
    ]


def test_recent_jobs_keeps_missing_and_invalid_timestamps_at_end() -> None:
    script = f"""
        import {{ recentJobs, jobCreatedTs }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        const jobs = [
          {{ name: 'missing-created', phase: 'Completed' }},
          {{ name: 'valid-created', phase: 'Failed', created: '2026-05-18T12:00:00Z' }},
          {{ name: 'invalid-created', phase: 'Error', created: 'not-a-date' }},
        ];
        console.log(JSON.stringify({{
          names: recentJobs(jobs).map(j => j.name),
          missingTs: jobCreatedTs(jobs[0]),
          invalidTs: jobCreatedTs(jobs[2]),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "names": ["valid-created", "missing-created", "invalid-created"],
        "missingTs": 0,
        "invalidTs": 0,
    }


def test_recent_jobs_accepts_nullish_and_sparse_inputs() -> None:
    script = f"""
        import {{ recentJobs, isRecentJob }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        const sparseJobs = [
          null,
          undefined,
          {{ name: 'terminal', phase: 'completed', created: '2026-05-18T12:00:00Z' }},
          {{ name: 'no-phase', created: '2026-05-18T13:00:00Z' }},
        ];
        console.log(JSON.stringify({{
          nullList: recentJobs(null),
          undefinedList: recentJobs(undefined),
          sparseNames: recentJobs(sparseJobs).map(j => j.name),
          nullIsRecent: isRecentJob(null),
          undefinedIsRecent: isRecentJob(undefined),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "nullList": [],
        "undefinedList": [],
        "sparseNames": ["terminal"],
        "nullIsRecent": False,
        "undefinedIsRecent": False,
    }


def test_dashboard_helpers_do_not_currently_export_namespace_or_model_filters() -> None:
    script = f"""
        import * as helpers from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify(Object.keys(helpers).sort()));
    """

    assert json.loads(run_node(script)) == ["isRecentJob", "jobCreatedTs", "recentJobs"]


def test_every_dashboard_poller_is_named() -> None:
    """`poll()` only records freshness for a poller that passes a `source`, so
    an unnamed poller silently drops out of the staleness banner. Counted
    rather than listed by name: the set of pollers changes (the leaderboard
    fanout became a single `scatter` call), the naming requirement does not."""
    source = (DASHBOARD_HELPERS_PATH.parent / "dashboard.js").read_text(
        encoding="utf-8"
    )

    assert source.count("poll(async () =>") == source.count("source: '")


def test_dashboard_pollers_that_handle_an_error_still_rethrow_it() -> None:
    """A poller that catches its own error to paint a local empty state must
    rethrow, or `poll()` never counts the failure and the app-level
    "Operator API unreachable" banner never appears."""
    source = (DASHBOARD_HELPERS_PATH.parent / "dashboard.js").read_text(
        encoding="utf-8"
    )

    cluster_block = source.split("const data = await api.getCluster();", 1)[1].split(
        "}, 10000", 1
    )[0]
    scatter_block = source.split("const data = await api.getScatterData();", 1)[
        1
    ].split("}, 30000", 1)[0]

    assert "setClusterError(true);" in cluster_block
    assert "throw err;" in cluster_block
    # The scatter poller has no local error state, so it must not catch at all.
    assert "catch" not in scatter_block
