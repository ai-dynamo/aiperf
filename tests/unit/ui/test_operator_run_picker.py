# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ui RunPicker (component + helpers).

The component is exercised by importing pure helpers via ``node`` and
asserting the JSON-serialized output. Render assertions live in the
integration smoke tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Tests import the pure helpers from the sibling module — `run-picker.js`
# itself imports htm/preact (browser-only via importmap) which raw Node
# cannot resolve. The component re-exports the helpers so the runtime
# surface is unchanged.
RUN_PICKER_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "run-picker-helpers.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _epochs_fixture() -> list[dict]:
    return [
        {
            "epoch": "1000",
            "isLatest": False,
            "mtimeEpoch": 1000,
            "fileCount": 3,
            "status": "succeeded",
            "startedAt": 999,
            "endedAt": 1001,
        },
        {
            "epoch": "2000",
            "isLatest": False,
            "mtimeEpoch": 2000,
            "fileCount": 5,
            "status": "failed",
            "startedAt": 1999,
            "endedAt": 2002,
        },
        {
            "epoch": "3000",
            "isLatest": True,
            "mtimeEpoch": 3000,
            "fileCount": 7,
            "status": "running",
            "startedAt": 2999,
            "endedAt": None,
        },
    ]


def test_build_picker_rows_orders_newest_first_with_ordinal_labels() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildPickerRows }} from {RUN_PICKER_PATH.as_uri()!r};
        const rows = buildPickerRows({{
          namespace: 'bench',
          name: 'j1',
          epochs: {fixture},
          current: undefined,
        }});
        console.log(JSON.stringify(rows));
    """
    rows = json.loads(_run_node(script))
    assert [r["label"] for r in rows] == ["Run 3", "Run 2", "Run 1"]
    assert [r["status"] for r in rows] == ["running", "failed", "succeeded"]
    assert [r["isLatest"] for r in rows] == [True, False, False]
    assert [r["selected"] for r in rows] == [True, False, False]
    assert rows[0]["href"] == "#/jobs/bench/j1"
    assert rows[1]["href"] == "#/jobs/bench/j1/runs/2000"


def test_build_picker_rows_marks_pinned_older_run_selected() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildPickerRows }} from {RUN_PICKER_PATH.as_uri()!r};
        const rows = buildPickerRows({{
          namespace: 'bench',
          name: 'j1',
          epochs: {fixture},
          current: '2000',
        }});
        console.log(JSON.stringify(rows.map(r => ({{
          label: r.label, selected: r.selected, isLatest: r.isLatest,
        }}))));
    """
    rows = json.loads(_run_node(script))
    assert rows == [
        {"label": "Run 3", "selected": False, "isLatest": True},
        {"label": "Run 2", "selected": True, "isLatest": False},
        {"label": "Run 1", "selected": False, "isLatest": False},
    ]


def test_build_button_label_for_each_state() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const epochs = {fixture};
        const cases = [
          // viewing latest, running
          {{ current: undefined, now: 3060 }},
          // viewing older
          {{ current: '2000', now: 5602 }},
          // viewing latest after completion (mock different epochs)
          {{
            current: undefined, now: 3700,
            epochs: [{{ ...epochs[2], status: 'succeeded', endedAt: 3000 }}, ...epochs.slice(0, 2)],
          }},
        ];
        const out = cases.map(c => buildButtonLabel({{
          epochs: c.epochs ?? epochs,
          current: c.current, now: c.now,
        }}));
        console.log(JSON.stringify(out));
    """
    out = json.loads(_run_node(script))
    # Running latest: numeric "Run 3 · running"
    assert out[0]["text"].startswith("Run 3")
    assert "running" in out[0]["text"]
    assert out[0]["status"] == "running"
    assert out[0]["isLatest"] is True
    # Viewing older: includes "not latest"
    assert out[1]["isLatest"] is False
    assert out[1]["notLatest"] is True
    assert out[1]["text"].startswith("Run 2")
    # Latest completed: relative-time format
    assert out[2]["status"] == "succeeded"
    assert out[2]["isLatest"] is True


def test_build_button_label_single_epoch_renders_inert() -> None:
    fixture = json.dumps([_epochs_fixture()[2]])
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const out = buildButtonLabel({{
          epochs: {fixture}, current: undefined, now: 3060,
        }});
        console.log(JSON.stringify(out));
    """
    out = json.loads(_run_node(script))
    assert out["inert"] is True


def test_build_button_label_zero_epochs_returns_null() -> None:
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const out = buildButtonLabel({{ epochs: [], current: undefined, now: 0 }});
        console.log(JSON.stringify(out));
    """
    assert _run_node(script) == "null"


def test_build_picker_rows_handles_stale_pinned_epoch() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildPickerRows, buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const rows = buildPickerRows({{
          namespace: 'bench', name: 'j1',
          epochs: {fixture}, current: '9999',
        }});
        const label = buildButtonLabel({{
          epochs: {fixture}, current: '9999', now: 5000,
        }});
        console.log(JSON.stringify({{
          rowCount: rows.length, anySelected: rows.some(r => r.selected),
          label: label,
        }}));
    """
    out = json.loads(_run_node(script))
    # Orphan epochs are not added as menu rows.
    assert out["rowCount"] == 3
    assert out["anySelected"] is False
    assert out["label"]["text"].startswith("Run ?")
    assert out["label"]["status"] == "unknown"
    assert out["label"]["notLatest"] is True


def test_orphan_pinned_epoch_keeps_dropdown_openable() -> None:
    """Stale URL on a job with one epoch must still allow Jump-to-latest."""
    fixture = json.dumps([_epochs_fixture()[2]])  # one epoch only
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const out = buildButtonLabel({{
          epochs: {fixture}, current: '9999', now: 5000,
        }});
        console.log(JSON.stringify(out));
    """
    out = json.loads(_run_node(script))
    assert out["text"].startswith("Run ?")
    assert out["notLatest"] is True
    assert out["inert"] is False
