# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial checkpoints must actually be written during a cluster run.

Two consumers depend on them and both survived the port while the writer did
not: the results sidecar serves ``checkpoints/`` *before* the ready marker so a
run can be inspected while it is still going, and the operator's completion
fetch uses "the checkpoint on disk is still growing" as its liveness heuristic.
With no writer, ``checkpoints/`` is always empty, live inspection returns
nothing and the heuristic degenerates.
"""

import orjson

from aiperf.records.records_manager_export import (
    build_checkpoint_snapshot,
    write_json_file_atomic,
)


class TestAtomicWrite:
    def test_writes_and_creates_parents(self, tmp_path):
        dest = tmp_path / "checkpoints" / "partial.json"
        write_json_file_atomic(dest, b'{"a": 1}')
        assert orjson.loads(dest.read_bytes()) == {"a": 1}

    def test_leaves_no_temp_file_behind(self, tmp_path):
        dest = tmp_path / "partial.json"
        write_json_file_atomic(dest, b"{}")
        assert [p.name for p in tmp_path.iterdir()] == ["partial.json"]

    def test_replaces_existing_content(self, tmp_path):
        dest = tmp_path / "partial.json"
        write_json_file_atomic(dest, b'{"n": 1}')
        write_json_file_atomic(dest, b'{"n": 2}')
        assert orjson.loads(dest.read_bytes()) == {"n": 2}


class _Stats:
    def __init__(self, total=0, success=0, error=0, start=None, end=None):
        self.total_records = total
        self.success_records = success
        self.error_records = error
        self.start_ns = start
        self.requests_end_ns = end
        self.phase_name = "profiling_0"
        self.phase_kind = "profiling"


class _Tracker:
    def __init__(self, stats_by_key):
        self._phase_trackers = dict.fromkeys(stats_by_key)
        self._stats = stats_by_key

    def create_stats_for_phase(self, phase, phase_index=None):
        return self._stats[(phase, phase_index)]


class TestSnapshot:
    def test_counts_records_across_phases(self):
        tracker = _Tracker(
            {
                ("profiling", 0): _Stats(total=10, success=9, error=1, start=5, end=15),
                ("profiling", 1): _Stats(total=4, success=4, error=0, start=20, end=30),
            }
        )
        snap = build_checkpoint_snapshot(tracker)

        assert snap["total_records"] == 14
        assert snap["successful_request_count"] == 13
        assert snap["error_request_count"] == 1
        assert snap["start_ns"] == 5
        assert snap["end_ns"] == 30
        assert len(snap["phases"]) == 2

    def test_empty_tracker_is_still_valid(self):
        snap = build_checkpoint_snapshot(_Tracker({}))
        assert snap["total_records"] == 0
        assert snap["phases"] == []

    def test_snapshot_is_json_serialisable(self):
        tracker = _Tracker({("profiling", 0): _Stats(total=1, success=1)})
        orjson.dumps(build_checkpoint_snapshot(tracker))

    def test_record_count_grows_with_the_run(self):
        """The operator's liveness heuristic reads exactly this movement."""
        stats = _Stats(total=1)
        tracker = _Tracker({("profiling", 0): stats})
        first = build_checkpoint_snapshot(tracker)["total_records"]
        stats.total_records = 7
        second = build_checkpoint_snapshot(tracker)["total_records"]
        assert (first, second) == (1, 7)
