# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--num-dataset-entries`` caps the dynamo corpus by TIME, not by session id.

The cap keeps the first N eligible session-trees in the adapter's scan order,
and that order was the sorted root session id. With unordered ids -- the norm
for recorded captures -- the kept set is a lexicographic sample scattered across
the whole capture, which is the "sparse arrivals separated by large idle gaps"
shape the graph selection is documented to avoid. These pin the cap as a slice
of the recorded TIMELINE, on BOTH the serial and the parallel build paths (they
are asserted to select the same set, so they must be reordered together).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from aiperf.dataset.graph.parse_context import GraphParseContext
from aiperf.dataset.graph.parser import parse_graph

# ids deliberately sort the OPPOSITE way to time, so a root-id-ordered cap and a
# recorded-start-ordered cap select disjoint sets.
_ID_ORDER_REVERSED = [
    ("aaa", 60_000),
    ("bbb", 45_000),
    ("mmm", 30_000),
    ("yyy", 15_000),
    ("zzz", 0),
]

_BASE_MS = 1_700_000_000_000


def _write_capture(directory: Path, specs: list[tuple[str, int]]) -> Path:
    """Write a one-request-per-session dynamo capture at the given offsets.

    Block alignment: one replay hash at ``trace_block_size=16`` requires
    ``0 < input_length <= 16``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "trace.000000.jsonl.gz"
    with gzip.open(out, "wt") as handle:
        for index, (session_id, offset_ms) in enumerate(specs):
            handle.write(
                json.dumps(
                    {
                        "schema": "dynamo.request.trace.v1",
                        "event_type": "request_end",
                        "event_time_unix_ms": _BASE_MS + offset_ms,
                        "event_source": "dynamo",
                        "agent_context": {"session_id": session_id},
                        "request": {
                            "request_id": f"{session_id}-0",
                            "model": "m",
                            "input_tokens": 16,
                            "output_tokens": 8,
                            "cached_tokens": 0,
                            "replay": {
                                "trace_block_size": 16,
                                "input_length": 16,
                                "input_sequence_hashes": [9_000 + index],
                            },
                        },
                    }
                )
                + "\n"
            )
    return directory


def _selected_ids(path: Path, *, limit: int) -> list[str]:
    ctx = GraphParseContext(
        content_tokenizer="builtin",
        content_root_seed=1234,
        num_dataset_entries=limit,
    )
    parsed = parse_graph(path, format="dynamo_trace", ctx=ctx)
    return sorted(trace.id for trace in parsed.traces)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_serial_cap_keeps_the_earliest_traces(tmp_path, limit: int) -> None:
    """The serial path caps by recorded start, not by root session id."""
    capture = _write_capture(tmp_path / "cap", _ID_ORDER_REVERSED)

    selected = _selected_ids(capture, limit=limit)

    earliest = sorted(
        sid for sid, _ in sorted(_ID_ORDER_REVERSED, key=lambda s: s[1])[:limit]
    )
    assert selected == earliest, (
        f"--num-dataset-entries {limit} must keep the {limit} earliest traces; "
        f"got {selected}, expected {earliest}"
    )


@pytest.mark.parametrize("limit", [2, 3])
def test_parallel_cap_selects_the_same_set_as_serial(
    tmp_path, monkeypatch, caplog, limit: int
) -> None:
    """The parallel path must agree with the serial one, trace for trace.

    The two selections are asserted to match by construction, so reordering one
    without the other would silently split the corpus a build produces from the
    corpus its schedule plane expects. Forcing the threshold to 0 routes this
    tiny capture through the parallel grouping scan. ``limit=1`` is excluded on
    purpose: a single selected tree resolves to one worker, so the build
    correctly declines to fan out and there is no parallel build to compare --
    :func:`test_roots_in_arrival_order_*` covers the ordering directly instead.
    """
    from aiperf.common.environment import Environment

    capture = _write_capture(tmp_path / "cap", _ID_ORDER_REVERSED)
    serial = _selected_ids(capture, limit=limit)

    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    caplog.clear()
    with caplog.at_level("INFO"):
        parallel = _selected_ids(capture, limit=limit)

    # Guard against a vacuous comparison: if the build declined to fan out, this
    # test would be re-running the serial path and asserting it equals itself.
    assert "parallel build declined" not in caplog.text, (
        "the parallel path was not exercised, so this parity check proves nothing"
    )
    assert parallel == serial, (
        f"parallel selection {parallel} diverged from serial {serial}"
    )


def test_cap_is_stable_when_ids_and_time_agree(tmp_path) -> None:
    """A capture whose ids already sort by time is unaffected by the change."""
    aligned = [("s0", 0), ("s1", 10_000), ("s2", 20_000), ("s3", 30_000)]
    capture = _write_capture(tmp_path / "cap", aligned)

    assert _selected_ids(capture, limit=2) == ["s0", "s1"]


def test_equal_recorded_starts_fall_back_to_id_order(tmp_path) -> None:
    """Ties keep the previous deterministic root-id ordering."""
    tied = [("ccc", 0), ("aaa", 0), ("bbb", 0)]
    capture = _write_capture(tmp_path / "cap", tied)

    assert _selected_ids(capture, limit=2) == ["aaa", "bbb"]


# --- The parallel root ordering, unit-tested directly ------------------------


def _scan_with(starts: dict[str, int]):
    from aiperf.dataset.graph.adapters.dynamo.trace_parallel import _GroupingScan

    scan = _GroupingScan()
    scan.session_start_ms = dict(starts)
    return scan


def test_roots_in_arrival_order_sorts_by_first_request_end() -> None:
    """Roots order by arrival, not by id."""
    from aiperf.dataset.graph.adapters.dynamo.trace_parallel import (
        _roots_in_arrival_order,
    )

    scan = _scan_with({"aaa": 300, "mmm": 200, "zzz": 100})
    root_of = {"aaa": "aaa", "mmm": "mmm", "zzz": "zzz"}

    assert _roots_in_arrival_order(scan, root_of) == ["zzz", "mmm", "aaa"]


def test_roots_in_arrival_order_uses_the_tree_earliest_session() -> None:
    """A tree arrives when its EARLIEST session does, parent or child."""
    from aiperf.dataset.graph.adapters.dynamo.trace_parallel import (
        _roots_in_arrival_order,
    )

    # Root "late_parent" starts at 500, but its subagent starts at 50, so the
    # tree arrives at 50 -- ahead of the tree rooted at 100.
    scan = _scan_with({"late_parent": 500, "child": 50, "other": 100})
    root_of = {"late_parent": "late_parent", "child": "late_parent", "other": "other"}

    assert _roots_in_arrival_order(scan, root_of) == ["late_parent", "other"]


def test_roots_in_arrival_order_breaks_ties_on_root_id() -> None:
    """Equal arrivals fall back to root id, so the order is deterministic."""
    from aiperf.dataset.graph.adapters.dynamo.trace_parallel import (
        _roots_in_arrival_order,
    )

    scan = _scan_with({"ccc": 10, "aaa": 10, "bbb": 10})
    root_of = {"ccc": "ccc", "aaa": "aaa", "bbb": "bbb"}

    assert _roots_in_arrival_order(scan, root_of) == ["aaa", "bbb", "ccc"]


def test_roots_with_no_recorded_start_sort_last() -> None:
    """A root the scan never timestamped must not claim the earliest slot."""
    from aiperf.dataset.graph.adapters.dynamo.trace_parallel import (
        _roots_in_arrival_order,
    )

    scan = _scan_with({"timed": 900})  # "untimed" absent from the map entirely
    root_of = {"timed": "timed", "untimed": "untimed"}

    assert _roots_in_arrival_order(scan, root_of) == ["timed", "untimed"]
