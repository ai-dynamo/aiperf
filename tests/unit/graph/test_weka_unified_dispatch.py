# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.dataset.graph.adapters.weka import trace_parallel as parallel
from aiperf.dataset.graph.adapters.weka.trace import EmptyWekaTraceError

FIX_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"

PARSE_KWARGS: dict[str, Any] = {
    "tag": "from-weka-trace",
    "idle_gap_cap_seconds": None,
    "content_root_seed": 42,
    "content_tokenizer": None,
    "prompt_corpus": None,
    "max_osl": None,
}


def _row_items(count: int) -> list[parallel._WorkItem]:
    raw = orjson.loads(FIX_MIN.read_bytes())
    items = []
    for i in range(count):
        row = dict(raw)
        row["id"] = f"trace-{i}"
        items.append(parallel._WorkItem(source=f"src#{i}", path=None, row=row))
    return items


def test_parse_items_serial_below_threshold_merges_all_traces() -> None:
    parsed = parallel.parse_items(
        _row_items(2),
        source_label="src",
        threshold=8,
        parse_kwargs=dict(PARSE_KWARGS),
    )
    assert [t.id for t in parsed.traces] == ["trace-0", "trace-1"]
    assert set(parsed.graphs) == {"trace-0", "trace-1"}


def test_parse_items_file_and_row_items_produce_identical_traces() -> None:
    raw = orjson.loads(FIX_MIN.read_bytes())
    by_row = parallel.parse_items(
        [parallel._WorkItem(source="r", path=None, row=dict(raw))],
        source_label="r",
        threshold=8,
        parse_kwargs=dict(PARSE_KWARGS),
    )
    by_file = parallel.parse_items(
        [parallel._WorkItem(source=str(FIX_MIN), path=str(FIX_MIN), row=None)],
        source_label=str(FIX_MIN),
        threshold=8,
        parse_kwargs=dict(PARSE_KWARGS),
    )
    assert [t.id for t in by_row.traces] == [t.id for t in by_file.traces]
    assert by_row.graph.nodes.keys() == by_file.graph.nodes.keys()


def test_parse_items_above_threshold_routes_through_streaming_pool(
    monkeypatch,
) -> None:
    seen_workers: list[int] = []

    def fake_pool(worker_fn, work_items, *, workers, **_kwargs):  # noqa: ANN001
        seen_workers.append(workers)
        for task in work_items:
            yield worker_fn(task)

    monkeypatch.setattr(parallel, "_run_pool_streaming", fake_pool)
    items = _row_items(3)
    parsed = parallel.parse_items(
        items,
        source_label="src",
        item_count=len(items),
        threshold=2,
        workers=8,
        parse_kwargs=dict(PARSE_KWARGS),
    )
    assert len(parsed.traces) == 3
    # Known item_count caps the worker fan-out (previously only the directory
    # path applied this cap; row sources spawned the full configured count).
    assert seen_workers == [3]


def test_parse_items_zero_items_raises_empty_error() -> None:
    with pytest.raises(EmptyWekaTraceError, match="src-label"):
        parallel.parse_items(
            [],
            source_label="src-label",
            threshold=8,
            parse_kwargs=dict(PARSE_KWARGS),
        )


def test_iter_item_segment_payloads_first_yield_consumes_only_prefetch_window(
    monkeypatch,
) -> None:
    raw = orjson.loads(FIX_MIN.read_bytes())
    consumed: list[int] = []

    def items():
        for index in range(10):
            consumed.append(index)
            row = dict(raw)
            row["id"] = f"trace-{index}"
            yield parallel._WorkItem(source=f"src#{index}", path=None, row=row)

    def fake_pool(worker_fn, work_items, **_kwargs):  # noqa: ANN001
        assert consumed == [0, 1, 2]
        iterator = iter(work_items)
        yield worker_fn(next(iterator))

    monkeypatch.setattr(parallel, "_run_pool_streaming", fake_pool)
    payloads_iter = parallel.iter_item_segment_payloads(
        items(),
        source_label="src",
        threshold=2,
        workers=1,
        parse_kwargs=dict(PARSE_KWARGS),
    )
    first = next(payloads_iter)
    assert first.trace_id == "trace-0"
    assert consumed == [0, 1, 2]


def test_workers_never_reenter_public_route(monkeypatch) -> None:
    # The pool worker core must call the per-item parsers directly, not the
    # public dispatcher (which re-runs HF-id and is_dir detection per file).
    from aiperf.dataset.graph.adapters.weka import trace as weka_trace

    def boom(*_a: Any, **_k: Any):
        raise AssertionError("worker re-entered from_weka_trace")

    monkeypatch.setattr(weka_trace, "from_weka_trace", boom)
    item = parallel._WorkItem(source=str(FIX_MIN), path=str(FIX_MIN), row=None)
    blob = parallel._parse_item_to_msgpack((item, dict(PARSE_KWARGS)))
    assert isinstance(blob, bytes) and blob
