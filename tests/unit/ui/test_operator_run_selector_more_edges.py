# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for operator UI run selector helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

RUN_SELECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "run-selector.js"
)


def _run_selector_rows(options: dict) -> list[dict]:
    script = f"""
        import {{ buildRunSelectorRows }} from {RUN_SELECTOR_PATH.as_uri()!r};
        const rows = buildRunSelectorRows({json.dumps(options)});
        console.log(JSON.stringify(rows));
    """
    return json.loads(run_node(script))


def test_run_selector_sorts_numeric_epochs_newest_first() -> None:
    rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [
                {"epoch": "9"},
                {"epoch": "10"},
                {"epoch": 2},
            ],
            "current": "10",
            "hasLive": False,
            "isRunning": False,
        }
    )

    assert [row["epoch"] for row in rows] == ["10", "9", "2"]


def test_run_selector_keeps_latest_pill_for_completed_jobs() -> None:
    rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [
                {"epoch": "100", "isLatest": False},
                {"epoch": "200", "isLatest": True},
            ],
            "current": "100",
            "hasLive": True,
            "isRunning": False,
        }
    )

    assert [row["isLatest"] for row in rows] == [True, False]


def test_run_selector_live_row_only_when_running_and_selected_for_live_url() -> None:
    rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [{"epoch": "200", "isLatest": True}],
            "current": None,
            "hasLive": True,
            "isRunning": True,
        }
    )

    assert rows[0] == {
        "kind": "live",
        "epoch": None,
        "label": "Live",
        "selected": True,
        "href": "#/jobs/bench/job",
        "fileCount": None,
        "mtimeEpoch": None,
        "isLatest": False,
    }
    assert [row["kind"] for row in rows] == ["live", "epoch"]

    completed_rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [{"epoch": "200", "isLatest": True}],
            "current": "200",
            "hasLive": True,
            "isRunning": False,
        }
    )
    assert [row["kind"] for row in completed_rows] == ["epoch"]


def test_run_selector_encodes_namespace_name_and_epoch_in_hrefs() -> None:
    rows = _run_selector_rows(
        {
            "namespace": "ns with/slash",
            "name": "job #1/blue",
            "epochs": [{"epoch": "2026/05/18 12:34"}],
            "current": "2026/05/18 12:34",
            "hasLive": True,
            "isRunning": True,
        }
    )

    assert rows[0]["href"] == "#/jobs/ns%20with%2Fslash/job%20%231%2Fblue"
    assert rows[1]["href"] == (
        "#/jobs/ns%20with%2Fslash/job%20%231%2Fblue/runs/2026%2F05%2F18%2012%3A34"
    )


def test_run_selector_normalizes_missing_file_count_and_mtime() -> None:
    rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [
                {"epoch": "200"},
                {"epoch": "100", "fileCount": 0, "mtimeEpoch": 0},
            ],
            "current": "200",
            "hasLive": False,
            "isRunning": False,
        }
    )

    assert rows[0]["fileCount"] is None
    assert rows[0]["mtimeEpoch"] is None
    assert rows[1]["fileCount"] == 0
    assert rows[1]["mtimeEpoch"] == 0


def test_run_selector_selects_latest_epoch_when_current_is_unpinned() -> None:
    rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [
                {"epoch": "100", "isLatest": False},
                {"epoch": "200", "isLatest": True},
            ],
            "current": None,
            "hasLive": True,
            "isRunning": False,
        }
    )

    assert [row["selected"] for row in rows] == [True, False]


def test_run_selector_orphan_pinned_epoch_marks_no_epoch_selected_and_keeps_latest_link() -> (
    None
):
    rows = _run_selector_rows(
        {
            "namespace": "bench",
            "name": "job",
            "epochs": [
                {"epoch": "100", "isLatest": False},
                {"epoch": "200", "isLatest": True},
            ],
            "current": "999",
            "hasLive": True,
            "isRunning": False,
        }
    )

    assert [row["epoch"] for row in rows] == ["200", "100"]
    assert all(row["selected"] is False for row in rows)
    assert rows[0]["href"] == "#/jobs/bench/job/runs/200"
    assert rows[0]["isLatest"] is True
