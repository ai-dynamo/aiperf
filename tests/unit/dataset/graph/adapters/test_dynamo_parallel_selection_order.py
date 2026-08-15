# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both parallel dynamo entry points must select trees in ARRIVAL order.

``--num-dataset-entries`` / ``--num-conversations`` keep the FIRST N eligible
trees, so the order the roots are enumerated in decides WHICH conversations get
benchmarked. ``trace.order_trees_by_recorded_start`` states the contract: the cap
must yield a contiguous slice of the recorded TIMELINE, not of the alphabet.

``stream_dynamo_trace_segment_payloads`` -- the entry point the production store
build uses (``store_build.py`` routes ``dynamo_trace`` there) -- enumerated
``sorted(set(root_of.values()))`` instead, so a bounded load silently benchmarked
the alphabetically-first N session ids. Its sibling
``maybe_build_fused_parallel`` was corrected to arrival order and this one was
missed, which also meant the same corpus under the same flags selected a
DIFFERENT set depending only on whether the tree count cleared the parallel
threshold.

The pre-existing selection suite cannot catch this: its fixtures stamp one
identical ``event_time_unix_ms`` on every record, so arrival ties resolve to root
id and the two orderings coincide. These fixtures make arrival order the exact
REVERSE of alphabetical order so the two can never be confused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.dataset.graph.adapters.dynamo import trace_parallel
from aiperf.dataset.graph.adapters.dynamo.trace import root_of_sessions
from aiperf.dataset.graph.adapters.dynamo.trace_parallel import (
    _roots_in_arrival_order,
    _scan_grouping,
)
from aiperf.dataset.graph.adapters.shared import selection as shared_selection

from .conftest import write_jsonl

T0 = 1_700_000_000_000
BLOCK_SIZE = 64
TREES = 12


def _record(index: int) -> dict[str, Any]:
    """Session ``s{index}``; arrival DESCENDING in index, so s11 is earliest."""
    sid = f"s{index:02d}"
    received = T0 + (TREES - index) * 10_000
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": received + 1_000,
        "agent_context": {"session_id": sid, "trajectory_id": sid},
        "request": {
            "request_id": f"r-{sid}",
            "output_tokens": 8,
            "request_received_ms": received,
            "total_time_ms": 1_000.0,
            "replay": {
                "trace_block_size": BLOCK_SIZE,
                "input_length": 2 * BLOCK_SIZE,
                "input_sequence_hashes": [1_000 + index, 2_000 + index],
            },
        },
    }


@pytest.fixture
def reversed_arrival_trace(tmp_path: Path) -> Path:
    """12 single-turn root sessions whose arrival order reverses their ids."""
    return write_jsonl(tmp_path / "order.jsonl", [_record(i) for i in range(TREES)])


def test_fixture_makes_arrival_and_alphabetical_order_disagree(
    reversed_arrival_trace: Path,
) -> None:
    """Guard: if these two ever coincide, the tests below prove nothing."""
    scan = _scan_grouping(reversed_arrival_trace, threads=2, capture_peak=True)
    root_of = root_of_sessions(scan.request_end_sessions, scan.parent_link)

    arrival = _roots_in_arrival_order(scan, root_of)
    assert arrival == [f"s{i:02d}" for i in reversed(range(TREES))]
    assert arrival != sorted(set(root_of.values()))


def test_streaming_payload_path_selects_in_arrival_order(
    reversed_arrival_trace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store-build entry point must hand the cap arrival-ordered roots.

    Captured at the selection seam rather than asserted on the built output: the
    ordering decision happens before the build, and reaching the parallel build
    for real would spin a multiprocessing pool for a property that is settled by
    then.
    """
    seen: dict[str, list[str]] = {}
    real = trace_parallel._select_roots_filter_then_cap

    def _capture(scan, *, root_of, roots, **kwargs):
        seen["roots"] = list(roots)
        return real(scan, root_of=root_of, roots=roots, **kwargs)

    # Spy on the SHARED helper module as well as trace_parallel's by-value
    # import: the serial fallback re-selects inside ``from_dynamo_trace``, which
    # resolves ``log_selection_summary`` through its own function-local import.
    # Patching only the trace_parallel binding hid a double-log.
    summaries: list[object] = []
    spy = lambda stats, **kw: summaries.append(stats)  # noqa: E731
    monkeypatch.setattr(trace_parallel, "_select_roots_filter_then_cap", _capture)
    monkeypatch.setattr(trace_parallel, "log_selection_summary", spy)
    monkeypatch.setattr(shared_selection, "log_selection_summary", spy)

    list(
        trace_parallel.stream_dynamo_trace_segment_payloads(
            reversed_arrival_trace,
            content_root_seed=42,
            idle_gap_cap_seconds=None,
            content_tokenizer="builtin",
            prompt_corpus="coding",
            release_replay=False,
            max_depth=32,
            num_dataset_entries=3,
        )
    )

    assert seen["roots"] == [f"s{i:02d}" for i in reversed(range(TREES))]
    # Screening/capping a corpus silently is the other half of the defect: the
    # operator must be told how many trees the cap dropped -- ONCE. This run
    # declines to the serial fallback, which logs it itself, so the parallel
    # entry point must NOT also log. ``log_selection_summary`` documents
    # "exactly once no matter how the build parallelizes".
    assert len(summaries) == 1
