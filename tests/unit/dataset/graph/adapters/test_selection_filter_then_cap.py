# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ai-dynamo/aiperf#1106 regression: schema-only filter-then-cap selection.

Both recorded-trace loaders (weka + dynamo) must honor ``--num-dataset-entries``
and ``--max-context-length`` at LOAD time: reject traces whose peak context
exceeds the ceiling, then build only the first N ELIGIBLE traces. The historic
bug built every trace and cloned to fill lanes.

Fixture shape (both adapters): 20 traces, 8 of which exceed the chosen
``max_context_length``. The 8 over-limit traces sort FIRST in each adapter's
deterministic scan order (weka: dir file name; dynamo: root session id), so a
``num_dataset_entries=10`` load that (correctly) filters-then-caps builds exactly
10 eligible traces after scanning all 8 rejects -- while the BUGGY cap-then-filter
would take the first 10 (8 rejects + 2 eligible) and build only 2.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.adapters.shared.selection import SelectionStats
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.models import LlmNode

_MAX_CTX = 1000
_NUM_ENTRIES = 10
_OVER_LIMIT_COUNT = 8
_TOTAL = 20
# Peak of an over-limit trace (2000 + 10) and an eligible one (100 + 10).
_OVER_INPUT = 2000
_UNDER_INPUT = 100
_OUTPUT = 10


@pytest.fixture(autouse=True)
def _fresh_synth_cache():
    """Isolate the process-level synthesizer cache from other tests."""
    CorpusContentSynthesizer.reset_worker_cache()
    yield
    CorpusContentSynthesizer.reset_worker_cache()


@pytest.fixture(autouse=True)
def _force_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep both adapters' builds serial + in-process for deterministic speed."""
    monkeypatch.setattr(Environment.DATASET, "WEKA_GRAPH_PARALLEL_THRESHOLD", 1000)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 1000)


def _is_over_limit(index: int) -> bool:
    """The first ``_OVER_LIMIT_COUNT`` traces (in scan order) are over-limit."""
    return index < _OVER_LIMIT_COUNT


# --- weka -----------------------------------------------------------------


def _weka_trace_dict(index: int) -> dict:
    """One valid weka trace whose sole top-level request is over/under the cap."""
    input_length = _OVER_INPUT if _is_over_limit(index) else _UNDER_INPUT
    return {
        "id": f"trace-{index:02d}",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "m",
                "in": input_length,
                "out": _OUTPUT,
                "hash_ids": [1, 2],
            }
        ],
    }


def _write_weka_dir(root: Path) -> Path:
    """20 weka trace files named ``t00..t19`` (over-limit ones sort first)."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(_TOTAL):
        (root / f"t{index:02d}.json").write_bytes(orjson.dumps(_weka_trace_dict(index)))
    return root


def test_weka_filter_then_cap_builds_exactly_n_eligible(tmp_path: Path) -> None:
    root = _write_weka_dir(tmp_path / "weka")
    stats_out: list[SelectionStats] = []

    parsed = from_weka_trace(
        root,
        num_dataset_entries=_NUM_ENTRIES,
        max_context_length=_MAX_CTX,
        selection_out=stats_out,
    )

    # Exactly N eligible built -- NOT N minus the rejects that fell in the prefix.
    assert len(parsed.traces) == _NUM_ENTRIES
    assert len(stats_out) == 1
    stats = stats_out[0]
    assert stats.loaded == _NUM_ENTRIES
    assert stats.eligible == _NUM_ENTRIES
    assert stats.rejected_by_maxctx == _OVER_LIMIT_COUNT
    assert stats.largest_observed == _OVER_INPUT + _OUTPUT
    # Every built trace is an eligible (under-limit) one.
    built_ids = {t.id for t in parsed.traces}
    assert all(int(tid.split("-")[1]) >= _OVER_LIMIT_COUNT for tid in built_ids)


def test_weka_both_none_builds_everything(tmp_path: Path) -> None:
    root = _write_weka_dir(tmp_path / "weka")
    stats_out: list[SelectionStats] = []

    parsed = from_weka_trace(root, selection_out=stats_out)

    assert len(parsed.traces) == _TOTAL
    assert stats_out == []  # no selection performed when both knobs are unset


# --- dynamo ---------------------------------------------------------------


def _dynamo_record(session_id: str, input_tokens: int) -> dict:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": 1000,
        "event_source": "dynamo",
        "agent_context": {"session_id": session_id, "trajectory_id": session_id},
        "request": {
            "request_id": f"r-{session_id}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": _OUTPUT,
            "ttft_ms": 10.0,
        },
    }


def _write_dynamo_trace(path: Path) -> Path:
    """20 independent single-turn root sessions (= 20 trees), rejects sort first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for index in range(_TOTAL):
            input_tokens = _OVER_INPUT if _is_over_limit(index) else _UNDER_INPUT
            f.write(orjson.dumps(_dynamo_record(f"s{index:02d}", input_tokens)))
            f.write(b"\n")
    return path


_BLOCK_SIZE = 16


def _aligned_hashes(input_length: int, *, base: int) -> list[int]:
    """Block-aligned hash list for ``input_length`` (satisfies _assert_block_aligned).

    Dynamo records full-block hashes plus one partial tail, so a consistent
    record has ``ceil(input_length / block_size)`` hashes; distinct ``base`` per
    session keeps content-addressed segments from aliasing across trees.
    """
    n = max(1, -(-input_length // _BLOCK_SIZE))
    return [base + i for i in range(n)]


def _dynamo_replay_record(
    session_id: str, *, input_length: int, input_tokens: int, base: int
) -> dict:
    """A request_end carrying a real ``replay`` block (input_length + hashes).

    ``input_length`` (the field the peak helpers must read) is set INDEPENDENTLY
    of ``input_tokens`` so a peak computed off the wrong field would diverge.
    """
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": 1000,
        "event_source": "dynamo",
        "agent_context": {"session_id": session_id, "trajectory_id": session_id},
        "request": {
            "request_id": f"r-{session_id}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": _OUTPUT,
            "ttft_ms": 10.0,
            "replay": {
                "trace_block_size": _BLOCK_SIZE,
                "input_length": input_length,
                "input_sequence_hashes": _aligned_hashes(input_length, base=base),
            },
        },
    }


def _write_dynamo_replay_trace(path: Path) -> Path:
    """20 replay-bearing trees; peak is decided by ``replay.input_length``.

    ``input_tokens`` is INVERTED relative to ``input_length``: the 8 over-limit
    trees (by input_length) carry a small input_tokens, and the 12 eligible ones
    carry a large input_tokens. So a peak computed off ``input_tokens`` (the
    wrong field) would REJECT the eligibles and KEEP the over-limits -- a total
    inversion the fused-vs-serial parity assertion would catch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for index in range(_TOTAL):
            if _is_over_limit(index):
                input_length, input_tokens = _OVER_INPUT, _UNDER_INPUT
            else:
                input_length, input_tokens = _UNDER_INPUT, _OVER_INPUT
            record = _dynamo_replay_record(
                f"s{index:02d}",
                input_length=input_length,
                input_tokens=input_tokens,
                base=index * 1000 + 1,
            )
            f.write(orjson.dumps(record))
            f.write(b"\n")
    return path


def _session_ids(parsed) -> set[str]:
    """Every dynamo session across ALL selected trees.

    Dynamo now emits one ``GraphRecord`` per session-tree under
    ``parsed.graphs`` (multi-graph), so gather node sessions across every
    per-tree graph (falling back to the single ``parsed.graph`` for a
    single-graph parse); ``parsed.graph`` alone is only the FIRST tree.
    """
    graphs = list(parsed.graphs.values()) or [parsed.graph]
    return {
        node.metadata["dynamo"]["session_id"]
        for graph in graphs
        for node in graph.nodes.values()
        if isinstance(node, LlmNode)
    }


def test_dynamo_filter_then_cap_builds_exactly_n_eligible(tmp_path: Path) -> None:
    path = _write_dynamo_trace(tmp_path / "trace.jsonl")
    stats_out: list[SelectionStats] = []

    parsed = from_dynamo_trace(
        path,
        num_dataset_entries=_NUM_ENTRIES,
        max_context_length=_MAX_CTX,
        selection_out=stats_out,
    )

    # One node per selected single-turn tree; exactly N eligible built.
    assert len(_session_ids(parsed)) == _NUM_ENTRIES
    assert len(stats_out) == 1
    stats = stats_out[0]
    assert stats.loaded == _NUM_ENTRIES
    assert stats.eligible == _NUM_ENTRIES
    assert stats.rejected_by_maxctx == _OVER_LIMIT_COUNT
    assert stats.largest_observed == _OVER_INPUT + _OUTPUT
    # The kept trees are the first N eligible (s08..s17), never any over-limit one.
    expected = {
        f"s{i:02d}" for i in range(_OVER_LIMIT_COUNT, _OVER_LIMIT_COUNT + _NUM_ENTRIES)
    }
    assert _session_ids(parsed) == expected


def test_dynamo_both_none_builds_everything(tmp_path: Path) -> None:
    path = _write_dynamo_trace(tmp_path / "trace.jsonl")
    stats_out: list[SelectionStats] = []

    parsed = from_dynamo_trace(path, selection_out=stats_out)

    assert len(_session_ids(parsed)) == _TOTAL
    assert stats_out == []


def test_dynamo_fused_parallel_selection_matches_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection applies identically on the fused-parallel build path.

    Uses a REPLAY-bearing corpus (populated ``input_sequence_hashes`` +
    ``replay.input_length``), so the fused path's hash-free scan exercises the
    ``replay.input_length`` branch of ``_line_peak_context`` AND the prefix-bound
    hash-exclusion cut (the production path for real captures) -- not just the
    ``input_tokens`` fallback. ``input_tokens`` is INVERTED relative to
    ``input_length``, so a scan reading the wrong field would flip the selected
    set; forcing the pool path (threshold 0, 2 workers) must still select the
    SAME trees and stats as the serial ``dynamo_tree_peak_context`` path.
    """
    path = _write_dynamo_replay_trace(tmp_path / "trace.jsonl")

    serial_stats: list[SelectionStats] = []
    serial = from_dynamo_trace(
        path,
        num_dataset_entries=_NUM_ENTRIES,
        max_context_length=_MAX_CTX,
        selection_out=serial_stats,
    )

    # Selection was decided by replay.input_length (not the inverted
    # input_tokens): the kept trees are the low-input_length ones, s08..s17.
    expected = {
        f"s{i:02d}" for i in range(_OVER_LIMIT_COUNT, _OVER_LIMIT_COUNT + _NUM_ENTRIES)
    }
    assert _session_ids(serial) == expected
    assert serial_stats[0].largest_observed == _OVER_INPUT + _OUTPUT

    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", 2)
    parallel_stats: list[SelectionStats] = []
    parallel = from_dynamo_trace(
        path,
        num_dataset_entries=_NUM_ENTRIES,
        max_context_length=_MAX_CTX,
        selection_out=parallel_stats,
    )

    assert _session_ids(parallel) == _session_ids(serial)
    assert len(_session_ids(parallel)) == _NUM_ENTRIES
    assert parallel_stats[0] == serial_stats[0]
    assert parallel_stats[0].rejected_by_maxctx == _OVER_LIMIT_COUNT
