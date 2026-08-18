# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf.cli_commands.kube._runs_render`.

Two pure-rendering helpers:
- `annotate_preview` mirrors server-side enforce_retention dry-run semantics:
  marks each run's `would_delete` based on retain_runs (count keepers),
  retain_days (age cutoff), and protects `latest_epoch`. Also stamps the
  raw retention dict on the payload.
- `print_runs_table` renders the RunHistoryListResponse payload as a Rich
  table or an empty-state message; preview mode adds a WOULD DELETE column.

Both functions take dicts and call console helpers - no I/O, no external deps.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
from pytest import param


def _run(
    epoch: int,
    *,
    mtime_epoch: int = 0,
    file_count: int = 0,
    total_size_bytes: int = 0,
    is_latest: bool = False,
) -> dict[str, Any]:
    """Build a minimal run dict matching the RunHistoryListResponse shape."""
    return {
        "epoch": epoch,
        "mtime_epoch": mtime_epoch,
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "is_latest": is_latest,
    }


# ---------------------------------------------------------------------------
# annotate_preview
# ---------------------------------------------------------------------------


class TestAnnotatePreview:
    """Mirrors server-side retention-policy dry-run."""

    def test_empty_runs_stamps_retention_only(self) -> None:
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        payload: dict[str, Any] = {"runs": []}
        annotate_preview(payload, {"retain_runs": 5, "retain_days": 7})

        assert payload["retention"] == {"retain_runs": 5, "retain_days": 7}
        assert payload["runs"] == []

    def test_count_keepers_protect_top_n_by_mtime(self) -> None:
        """retain_runs=2 keeps the two most-recent runs by mtime_epoch."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        runs = [
            _run(1, mtime_epoch=100),
            _run(2, mtime_epoch=300),
            _run(3, mtime_epoch=200),
            _run(4, mtime_epoch=50),
        ]
        payload = {"runs": runs, "latest_epoch": None}
        annotate_preview(payload, {"retain_runs": 2, "retain_days": 0})

        by_epoch = {r["epoch"]: r["would_delete"] for r in runs}
        # Two newest by mtime_epoch are 2 (mtime=300) and 3 (mtime=200) - kept.
        # Older runs 1 (100) and 4 (50) should be marked for deletion.
        assert by_epoch == {1: True, 2: False, 3: False, 4: True}

    def test_latest_epoch_always_protected(self) -> None:
        """latest_epoch is forced to would_delete=False even outside count keepers."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        runs = [
            _run(1, mtime_epoch=10),
            _run(2, mtime_epoch=20),
            _run(3, mtime_epoch=30),
        ]
        payload = {"runs": runs, "latest_epoch": 1}
        annotate_preview(payload, {"retain_runs": 1, "retain_days": 0})

        by_epoch = {r["epoch"]: r["would_delete"] for r in runs}
        # retain_runs=1 keeps only epoch 3 (highest mtime).
        # But latest_epoch=1 must also be protected.
        assert by_epoch[1] is False
        assert by_epoch[3] is False
        assert by_epoch[2] is True

    def test_age_cutoff_does_not_override_count_keepers(self) -> None:
        """Age alone cannot delete runs retained by the count policy."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        now = int(time.time())
        day = 86400
        runs = [
            _run(1, mtime_epoch=now - 1 * day),  # 1 day old: kept by age
            _run(2, mtime_epoch=now - 5 * day),  # 5 days old: too old
            _run(3, mtime_epoch=now - 10 * day),  # 10 days old: too old
        ]
        payload = {"runs": runs, "latest_epoch": None}
        # retain_runs is large enough that every run is retained by count.
        annotate_preview(payload, {"retain_runs": 10, "retain_days": 3})

        by_epoch = {r["epoch"]: r["would_delete"] for r in runs}
        assert by_epoch[1] is False
        assert by_epoch[2] is False
        assert by_epoch[3] is False

    def test_zero_retention_marks_everything_for_deletion(self) -> None:
        """retain_runs=0, retain_days=0 -> every non-latest run is would_delete=True."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        runs = [_run(1, mtime_epoch=10), _run(2, mtime_epoch=20)]
        payload = {"runs": runs, "latest_epoch": None}
        annotate_preview(payload, {"retain_runs": 0, "retain_days": 0})

        assert all(r["would_delete"] for r in runs)

    def test_missing_runs_key_treated_as_empty(self) -> None:
        """Payload without 'runs' key is tolerated (returns no error)."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        payload: dict[str, Any] = {}
        annotate_preview(payload, {"retain_runs": 1, "retain_days": 1})
        assert payload["retention"] == {"retain_runs": 1, "retain_days": 1}

    def test_runs_none_treated_as_empty(self) -> None:
        """payload['runs']=None must not raise."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        payload: dict[str, Any] = {"runs": None}
        annotate_preview(payload, {"retain_runs": 1, "retain_days": 0})
        # No exception, retention still stamped
        assert "retention" in payload

    def test_missing_mtime_epoch_treated_as_zero(self) -> None:
        """Runs without mtime_epoch sort to the bottom (effectively 0)."""
        from aiperf.cli_commands.kube._runs_render import annotate_preview

        runs = [
            {"epoch": 1, "mtime_epoch": 100},  # newer
            {"epoch": 2},  # treated as mtime=0, oldest
        ]
        payload = {"runs": runs, "latest_epoch": None}
        annotate_preview(payload, {"retain_runs": 1, "retain_days": 0})

        # epoch 1 is the only count keeper; epoch 2 is dropped
        by_epoch = {r["epoch"]: r["would_delete"] for r in runs}
        assert by_epoch[1] is False
        assert by_epoch[2] is True


# ---------------------------------------------------------------------------
# print_runs_table
# ---------------------------------------------------------------------------


class TestPrintRunsTable:
    """Render path: empty state vs table; preview adds extra column + footer."""

    def test_empty_runs_calls_print_info_with_namespace_and_job(self) -> None:
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        payload = {"runs": [], "namespace": "ns-a", "job_id": "job-1"}
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.console.print") as mock_print,
        ):
            print_runs_table(payload)

        mock_info.assert_called_once()
        msg = mock_info.call_args.args[0]
        assert "ns-a" in msg
        assert "job-1" in msg
        # No table is printed when runs is empty
        mock_print.assert_not_called()

    def test_missing_namespace_and_job_id_default_to_empty(self) -> None:
        """payload without namespace/job_id uses empty strings in the message."""
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        payload: dict[str, Any] = {"runs": []}
        with patch("aiperf.kubernetes.console.print_info") as mock_info:
            print_runs_table(payload)

        # The format is "No runs found for {namespace}/{job_id}"
        assert mock_info.call_args.args[0].endswith("/")

    def test_renders_table_with_rows_and_hint(self) -> None:
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        runs = [
            _run(
                1, mtime_epoch=100, file_count=3, total_size_bytes=1024, is_latest=True
            ),
            _run(2, mtime_epoch=200, file_count=7, total_size_bytes=2048),
        ]
        payload = {"runs": runs, "namespace": "ns", "job_id": "job"}
        with patch("aiperf.kubernetes.console.console.print") as mock_print:
            print_runs_table(payload)

        # console.print is called twice: once for the table, once for the hint line
        assert mock_print.call_count == 2
        # The second call is the hint line as a string
        hint = mock_print.call_args_list[1].args[0]
        assert "aiperf kube results" in hint

    def test_preview_mode_adds_would_delete_column_and_footer(self) -> None:
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        runs = [
            _run(1, mtime_epoch=10, file_count=1, total_size_bytes=10),
        ]
        runs[0]["would_delete"] = True
        payload = {
            "runs": runs,
            "namespace": "ns",
            "job_id": "job",
            "retention": {"retain_runs": 5, "retain_days": 30},
        }
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.console.print") as mock_print,
        ):
            print_runs_table(payload, preview=True)

        # Table + hint = 2 print calls, plus the retention-policy footer goes
        # through print_info.
        assert mock_print.call_count == 2
        mock_info.assert_called_once()
        footer = mock_info.call_args.args[0]
        assert "RETAIN_RUNS=5" in footer
        assert "RETAIN_DAYS=30" in footer

    def test_preview_mode_zero_days_uses_disabled_label(self) -> None:
        """retain_days=0 footer reads 'age policy disabled', not '0'."""
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        payload = {
            "runs": [_run(1, mtime_epoch=10)],
            "namespace": "ns",
            "job_id": "job",
            "retention": {"retain_runs": 3, "retain_days": 0},
        }
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.console.print"),
        ):
            print_runs_table(payload, preview=True)

        footer = mock_info.call_args.args[0]
        assert "age policy disabled" in footer

    def test_preview_mode_missing_retention_still_renders(self) -> None:
        """When retention is absent on the payload, both values fall back to 0."""
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        payload = {
            "runs": [_run(1, mtime_epoch=10)],
            "namespace": "ns",
            "job_id": "job",
        }
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.console.print"),
        ):
            print_runs_table(payload, preview=True)

        footer = mock_info.call_args.args[0]
        assert "RETAIN_RUNS=0" in footer

    @pytest.mark.parametrize(
        "is_latest,expected_marker",
        [
            param(True, True, id="latest-shows-checkmark"),
            param(False, False, id="non-latest-empty"),
        ],
    )  # fmt: skip
    def test_latest_marker_in_table(
        self, is_latest: bool, expected_marker: bool
    ) -> None:
        """is_latest=True renders a green checkmark cell; False renders an empty cell."""
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        # We patch Table.add_row to capture the row tuple and inspect the
        # latest-cell value.
        captured_rows: list[tuple[Any, ...]] = []

        def _capture_add_row(self: Any, *cells: Any, **_: Any) -> None:
            captured_rows.append(cells)

        runs = [_run(1, mtime_epoch=10, is_latest=is_latest)]
        payload = {"runs": runs, "namespace": "ns", "job_id": "job"}
        with (
            patch("aiperf.kubernetes.console.console.print"),
            patch("rich.table.Table.add_row", _capture_add_row),
        ):
            print_runs_table(payload)

        assert len(captured_rows) == 1
        # Index 4 is the LATEST column (epoch, ts, files, size, latest)
        latest_cell = captured_rows[0][4]
        if expected_marker:
            assert "✓" in latest_cell
        else:
            assert latest_cell == ""

    def test_unicode_namespace_and_job_id_in_empty_message(self) -> None:
        """Non-ASCII namespace/job_id passes through unchanged."""
        from aiperf.cli_commands.kube._runs_render import print_runs_table

        payload = {"runs": [], "namespace": "ns-π", "job_id": "job-✓"}
        with patch("aiperf.kubernetes.console.print_info") as mock_info:
            print_runs_table(payload)

        msg = mock_info.call_args.args[0]
        assert "ns-π" in msg
        assert "job-✓" in msg
