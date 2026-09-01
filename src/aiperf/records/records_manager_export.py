# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-checkpoint export for the RecordsManager.

A cluster run is opaque until it finishes unless something writes progress to
disk as it goes. Two consumers depend on that:

* the results sidecar serves ``checkpoints/`` *before* the results-ready marker
  exists, so a run can be inspected while it is still going;
* the operator's completion fetch treats a checkpoint that is still growing as
  evidence the controller is alive rather than wedged.

Both survived the port; the writer did not, so ``checkpoints/`` was always
empty and the liveness heuristic had nothing to read.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.records.records_tracker import RecordsTracker


def write_json_file_atomic(path: Path, content: bytes) -> None:
    """Write a JSON file atomically so readers never observe a partial write.

    The sidecar serves this directory while it is being written, so a reader
    can arrive mid-write at any moment.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def build_checkpoint_snapshot(tracker: RecordsTracker) -> dict[str, Any]:
    """Summarise in-flight progress across every phase instance.

    Deliberately cheap: this runs on an interval during profiling, so it reads
    counters rather than building a full ``ProfileResults``, which would mean
    summarising every accumulator mid-run.
    """
    phases: list[dict[str, Any]] = []
    total = success = errors = 0
    start_ns: int | None = None
    end_ns: int | None = None

    for phase, phase_index in tracker._phase_trackers:
        stats = tracker.create_stats_for_phase(phase, phase_index)
        total += stats.total_records
        success += stats.success_records
        errors += stats.error_records
        if stats.start_ns is not None:
            start_ns = min(start_ns, stats.start_ns) if start_ns else stats.start_ns
        if stats.requests_end_ns is not None:
            end_ns = (
                max(end_ns, stats.requests_end_ns) if end_ns else stats.requests_end_ns
            )
        phases.append(
            {
                "phase": str(phase),
                "phase_index": phase_index,
                "phase_name": getattr(stats, "phase_name", None),
                "total_records": stats.total_records,
                "successful_request_count": stats.success_records,
                "error_request_count": stats.error_records,
            }
        )

    return {
        "partial": True,
        "written_ns": time.time_ns(),
        "total_records": total,
        "successful_request_count": success,
        "error_request_count": errors,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "phases": phases,
    }
