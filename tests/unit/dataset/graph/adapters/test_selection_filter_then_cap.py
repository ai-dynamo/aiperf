# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ai-dynamo/aiperf#1106 regression: the recorded-trace loader must filter by max context THEN cap to N at LOAD time."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.adapters.shared.selection import SelectionStats
from aiperf.dataset.graph.models import LlmNode, ParsedGraph

# The historic bug built every trace and cloned to fill lanes. Fixture shape: 20
# traces, 8 of which exceed max_context_length and sort FIRST in the adapter's
# deterministic scan order (dynamo: root session id). A correct
# num_dataset_entries=10 load therefore builds exactly 10 eligible traces after
# scanning past all 8 rejects, while the buggy cap-then-filter would take the
# first 10 (8 rejects + 2 eligible) and build only 2.

_MAX_CTX = 1000
_NUM_ENTRIES = 10
_OVER_LIMIT_COUNT = 8
_TOTAL = 20
# Peak of an over-limit trace (2000 + 10) and an eligible one (100 + 10).
_OVER_INPUT = 2000
_UNDER_INPUT = 100
_OUTPUT = 10
_BLOCK_SIZE = 16

# The kept trees are always the first N eligible ones, s08..s17.
_EXPECTED_KEPT = {
    f"s{i:02d}" for i in range(_OVER_LIMIT_COUNT, _OVER_LIMIT_COUNT + _NUM_ENTRIES)
}


@pytest.fixture(autouse=True)
def _fresh_synth_cache() -> Iterator[None]:
    """Isolate the process-level synthesizer cache from other tests."""
    CorpusContentSynthesizer.reset_worker_cache()
    yield
    CorpusContentSynthesizer.reset_worker_cache()


@pytest.fixture(autouse=True)
def _force_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the dynamo build serial + in-process for deterministic speed."""
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 1000)


def _is_over_limit(index: int) -> bool:
    """The first ``_OVER_LIMIT_COUNT`` traces (in scan order) are over-limit."""
    return index < _OVER_LIMIT_COUNT


# --- dynamo ---------------------------------------------------------------


def _dynamo_record(session_id: str, input_tokens: int) -> dict[str, Any]:
    """A schema-only ``request_end`` whose peak context comes from ``input_tokens``."""
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


def _aligned_hashes(input_length: int, *, base: int) -> list[int]:
    """``ceil(input_length / block_size)`` hashes from ``base``, matching dynamo's full-blocks-plus-partial-tail layout."""
    # Distinct base per session keeps content-addressed segments from aliasing
    # across trees; the count satisfies _assert_block_aligned.
    n = max(1, -(-input_length // _BLOCK_SIZE))
    return [base + i for i in range(n)]


def _dynamo_replay_record(
    session_id: str, *, input_length: int, input_tokens: int, base: int
) -> dict[str, Any]:
    """A ``request_end`` with a real ``replay`` block whose ``input_length`` is set INDEPENDENTLY of ``input_tokens``."""
    # Independence means a peak computed off the wrong field would diverge.
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
    """20 replay-bearing trees whose peak is decided by ``replay.input_length``, with ``input_tokens`` INVERTED against it."""
    # The inversion (over-limit trees carry small input_tokens, eligible ones
    # large) means a peak read off input_tokens would reject the eligibles and
    # keep the over-limits -- a total inversion the parity assertions catch.
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


def _session_ids(parsed: ParsedGraph) -> set[str]:
    """Every dynamo session across ALL selected per-tree graphs, falling back to the single-graph parse."""
    # Dynamo emits one GraphRecord per tree, so parsed.graph alone is only the FIRST tree.
    graphs = list(parsed.graphs.values()) or [parsed.graph]
    return {
        node.metadata["dynamo"]["session_id"]
        for graph in graphs
        for node in graph.nodes.values()
        if isinstance(node, LlmNode)
    }


def test_dynamo_filter_then_cap_builds_exactly_n_eligible(tmp_path: Path) -> None:
    """A capped load builds exactly N eligible trees (s08..s17) and reports the 8 max-context rejects in its stats."""
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
    assert _session_ids(parsed) == _EXPECTED_KEPT


def test_dynamo_both_none_builds_everything(tmp_path: Path) -> None:
    """With neither cap nor context ceiling, every tree is built and no selection stats are emitted."""
    path = _write_dynamo_trace(tmp_path / "trace.jsonl")
    stats_out: list[SelectionStats] = []

    parsed = from_dynamo_trace(path, selection_out=stats_out)

    assert len(_session_ids(parsed)) == _TOTAL
    assert stats_out == []


def test_dynamo_fused_parallel_selection_matches_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection applies identically on the fused-parallel path, decided by ``replay.input_length`` rather than the inverted ``input_tokens``."""
    # A replay-bearing corpus makes the fused path's hash-free scan exercise the
    # replay.input_length branch of _line_peak_context AND the prefix-bound
    # hash-exclusion cut -- the production path for real captures.
    path = _write_dynamo_replay_trace(tmp_path / "trace.jsonl")

    serial_stats: list[SelectionStats] = []
    serial = from_dynamo_trace(
        path,
        num_dataset_entries=_NUM_ENTRIES,
        max_context_length=_MAX_CTX,
        selection_out=serial_stats,
    )

    assert _session_ids(serial) == _EXPECTED_KEPT
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
