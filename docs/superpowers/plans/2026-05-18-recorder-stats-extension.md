# Recorder Stats Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the mock-server request-recorder summary with histograms and unique-value counts for `isl` and `requested_osl`, in both `<path>.summary.json` and the stdout summary.

**Architecture:** Two new pure helpers in `request_recorder.py` — `_histogram(values)` computes equal-width bins (with a `max_bin_width=100` cap and `min_bins=10` floor) and `_render_histogram(...)` formats them as indented stdout lines. `_build_summary` adds `histogram` + `unique_values` to each per-metric stats block; `_print_summary` calls the renderer after the existing percentile lines for ISL and OSL.

**Tech Stack:** Python 3.13, stdlib only (`statistics`, `math`, `collections`), `pytest`, `orjson`.

**Spec:** [`docs/superpowers/specs/2026-05-18-recorder-stats-extension-design.md`](../specs/2026-05-18-recorder-stats-extension-design.md)

---

## Task 1: Add `_histogram` helper with unit tests

**Files:**
- Create: `tests/unit/aiperf_mock_server/__init__.py`
- Create: `tests/unit/aiperf_mock_server/test_request_recorder.py`
- Modify: `tests/aiperf_mock_server/request_recorder.py` (add `import math`, two module constants, `_histogram` function)

- [ ] **Step 1: Create the empty `__init__.py` for the new test package**

```bash
mkdir -p tests/unit/aiperf_mock_server
: > tests/unit/aiperf_mock_server/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Write to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the request-recorder helpers."""

from aiperf_mock_server.request_recorder import _histogram


class TestHistogram:
    def test_empty_returns_none(self) -> None:
        assert _histogram([]) is None

    def test_single_value_returns_one_bin(self) -> None:
        hist = _histogram([42])
        assert hist == {"bin_edges": [42.0, 42.0], "counts": [1]}

    def test_all_equal_returns_one_bin(self) -> None:
        hist = _histogram([100, 100, 100])
        assert hist == {"bin_edges": [100.0, 100.0], "counts": [3]}

    def test_narrow_range_hits_min_bins_floor(self) -> None:
        # range 25..230 (width 205) -> ceil(205/100) = 3, but min_bins=10 wins
        values = list(range(25, 231, 5))  # 42 values spanning 25..230
        hist = _histogram(values)
        assert hist is not None
        assert len(hist["counts"]) == 10
        assert len(hist["bin_edges"]) == 11
        assert hist["bin_edges"][0] == 25.0
        assert hist["bin_edges"][-1] == 230.0
        assert sum(hist["counts"]) == len(values)

    def test_wide_range_hits_max_bin_width_cap(self) -> None:
        # range 207..1821 (width 1614) -> ceil(1614/100) = 17 bins
        values = list(range(207, 1822, 1))  # 1615 values
        hist = _histogram(values)
        assert hist is not None
        assert len(hist["counts"]) == 17
        assert len(hist["bin_edges"]) == 18
        assert hist["bin_edges"][0] == 207.0
        assert hist["bin_edges"][-1] == 1821.0
        assert sum(hist["counts"]) == len(values)

    def test_max_value_lands_in_last_bin(self) -> None:
        # Without the last-bin-closed rule, max would fall just past the last edge.
        values = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        hist = _histogram(values)
        assert hist is not None
        # 1000 must land in the last bin, not be lost
        assert sum(hist["counts"]) == len(values)
        assert hist["counts"][-1] >= 1

    def test_bin_widths_are_equal(self) -> None:
        hist = _histogram(list(range(0, 1001)))
        assert hist is not None
        edges = hist["bin_edges"]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        # All widths within 1e-9 of each other (allowing float drift)
        assert max(widths) - min(widths) < 1e-9
```

- [ ] **Step 3: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
```

Expected: `ImportError` or `AttributeError` for `_histogram` — the function doesn't exist yet.

- [ ] **Step 4: Implement `_histogram` and constants in `request_recorder.py`**

In `tests/aiperf_mock_server/request_recorder.py`, add `import math` to the imports block (alphabetical: between `logging` and `statistics`):

```python
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import IO, Any

import orjson
```

Add the two constants just after the imports and `logger = logging.getLogger(__name__)`:

```python
# Histogram bucketing rule: at least _HISTOGRAM_MIN_BINS bins, and bin width
# never exceeds _HISTOGRAM_MAX_BIN_WIDTH. Floor keeps narrow ranges informative;
# cap keeps wide ranges from collapsing 10 bins onto a 1500-token spread.
_HISTOGRAM_MIN_BINS = 10
_HISTOGRAM_MAX_BIN_WIDTH = 100.0
```

Add the `_histogram` function (place it just before `_quantiles`):

```python
def _histogram(values: list[int]) -> dict[str, list[float]] | None:
    """Equal-width histogram with the max_bin_width / min_bins rule.

    Returns ``None`` for an empty input, ``{"bin_edges": [v, v], "counts": [n]}``
    when all values are equal, and otherwise a dict with ``len(bin_edges) ==
    len(counts) + 1``. The last bin is closed on both ends so the observed
    maximum lands in it instead of just past the last edge.
    """
    if not values:
        return None
    lo = float(min(values))
    hi = float(max(values))
    if lo == hi:
        return {"bin_edges": [lo, hi], "counts": [len(values)]}
    span = hi - lo
    num_bins = max(
        _HISTOGRAM_MIN_BINS, math.ceil(span / _HISTOGRAM_MAX_BIN_WIDTH)
    )
    width = span / num_bins
    edges = [lo + i * width for i in range(num_bins + 1)]
    edges[-1] = hi  # pin last edge exactly to max to avoid float drift
    counts = [0] * num_bins
    for v in values:
        if v >= hi:
            idx = num_bins - 1
        else:
            idx = int((v - lo) / width)
            if idx >= num_bins:
                idx = num_bins - 1
        counts[idx] += 1
    return {"bin_edges": edges, "counts": counts}
```

- [ ] **Step 5: Run tests, confirm they pass**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/aiperf_mock_server/__init__.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py \
        tests/aiperf_mock_server/request_recorder.py
git commit -s -m "feat(mock-server): add _histogram helper for recorder summary"
```

---

## Task 2: Wire `histogram` + `unique_values` into `_build_summary`

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (`_build_summary`)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (add test class for `_build_summary`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
from collections import Counter, defaultdict

from aiperf_mock_server.request_recorder import _build_summary


class TestBuildSummary:
    def test_isl_block_has_histogram_and_unique_values(self) -> None:
        isls: dict = defaultdict(list, {"/v1/chat/completions": [10, 20, 30, 10]})
        osls: dict = defaultdict(list, {"/v1/chat/completions": [100, 200, 100]})
        summary = _build_summary(
            total=4,
            isls=isls,
            osls=osls,
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        isl_stats = summary["per_endpoint"]["/v1/chat/completions"]["isl"]
        assert isl_stats["unique_values"] == 3
        assert isinstance(isl_stats["histogram"], dict)
        assert sum(isl_stats["histogram"]["counts"]) == 4

    def test_requested_osl_unique_count(self) -> None:
        osls: dict = defaultdict(list, {"/v1/chat/completions": [16, 32, 16, 64]})
        summary = _build_summary(
            total=4,
            isls=defaultdict(list, {"/v1/chat/completions": [1, 2, 3, 4]}),
            osls=osls,
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        osl_stats = summary["per_endpoint"]["/v1/chat/completions"]["requested_osl"]
        assert osl_stats["unique_values"] == 3
        assert isinstance(osl_stats["histogram"], dict)

    def test_empty_osl_block_has_null_histogram(self) -> None:
        # Mimics /v1/embeddings — ISL is recorded, requested_osl never is.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/embeddings": [50, 60]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        osl_stats = summary["per_endpoint"]["/v1/embeddings"]["requested_osl"]
        assert osl_stats["histogram"] is None
        assert osl_stats["unique_values"] == 0
        # ISL block should still get a histogram
        isl_stats = summary["per_endpoint"]["/v1/embeddings"]["isl"]
        assert isinstance(isl_stats["histogram"], dict)
        assert isl_stats["unique_values"] == 2

    def test_min_tokens_block_unchanged(self) -> None:
        # min_tokens deliberately does NOT get the new fields.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/chat/completions": [10, 20]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list, {"/v1/chat/completions": [4, 8]}),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        mn = summary["per_endpoint"]["/v1/chat/completions"]["min_tokens"]
        assert "histogram" not in mn
        assert "unique_values" not in mn
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestBuildSummary -v
```

Expected: all 4 fail — `KeyError: 'unique_values'` or `KeyError: 'histogram'`.

- [ ] **Step 3: Extend `_build_summary` in `request_recorder.py`**

Replace the existing `_build_summary` function body with this version (the per-endpoint dict gets two new fields, applied only to `isl` and `requested_osl`):

```python
def _build_summary(
    total: int,
    isls: dict[str, list[int]],
    osls: dict[str, list[int]],
    min_tokens: dict[str, list[int]],
    streamed: dict[str, int],
    ignore_eos: dict[str, int],
    reasoning_efforts: dict[str, Counter[str]],
) -> dict[str, Any]:
    per_endpoint: dict[str, Any] = {}
    for ep in sorted(isls.keys()):
        isl_vals = isls[ep]
        osl_vals = osls.get(ep, [])
        isl_block = _quantiles(isl_vals) or {}
        isl_block["unique_values"] = len(set(isl_vals))
        isl_block["histogram"] = _histogram(isl_vals)
        osl_block = _quantiles(osl_vals) or {}
        osl_block["unique_values"] = len(set(osl_vals))
        osl_block["histogram"] = _histogram(osl_vals)
        per_endpoint[ep] = {
            "count": len(isl_vals),
            "streamed_count": streamed.get(ep, 0),
            "ignore_eos_count": ignore_eos.get(ep, 0),
            "reasoning_effort_counts": dict(reasoning_efforts.get(ep, Counter()))
            or None,
            "isl": isl_block or None,
            "requested_osl": osl_block if osl_vals or osl_block.get("histogram") else None,
            "min_tokens": _quantiles(min_tokens.get(ep, [])),
        }
    return {"total_requests": total, "per_endpoint": per_endpoint}
```

Note: the `requested_osl` block carries `unique_values: 0` and `histogram: None` when no OSL was ever set (embeddings). To preserve the existing JSON contract — where `requested_osl` is a flat `null` in that case — keep it as `None` when there are no values *and* no histogram. The `osl_block if osl_vals or ...` guard returns `None` in the empty case, matching the existing test fixtures.

Wait — the existing integration test asserts `emb_stats["requested_osl"] is None` directly. To keep that contract, when there are zero OSL values, emit the whole block as `None` (don't expose `unique_values: 0` at the top level). The unit test `test_empty_osl_block_has_null_histogram` expects the block to *exist* with `histogram: None` and `unique_values: 0`, which contradicts the existing integration assertion. Reconcile by updating the unit test instead — drop `test_empty_osl_block_has_null_histogram` and replace it with:

```python
    def test_empty_osl_block_is_none(self) -> None:
        # Mimics /v1/embeddings — requested_osl block stays `None` when no values.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/embeddings": [50, 60]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        assert summary["per_endpoint"]["/v1/embeddings"]["requested_osl"] is None
        # ISL block should still get a histogram
        isl_stats = summary["per_endpoint"]["/v1/embeddings"]["isl"]
        assert isinstance(isl_stats["histogram"], dict)
        assert isl_stats["unique_values"] == 2
```

And simplify the build-summary body — since `requested_osl` is either fully populated or `None`, use:

```python
def _build_summary(
    total: int,
    isls: dict[str, list[int]],
    osls: dict[str, list[int]],
    min_tokens: dict[str, list[int]],
    streamed: dict[str, int],
    ignore_eos: dict[str, int],
    reasoning_efforts: dict[str, Counter[str]],
) -> dict[str, Any]:
    per_endpoint: dict[str, Any] = {}
    for ep in sorted(isls.keys()):
        isl_vals = isls[ep]
        osl_vals = osls.get(ep, [])
        per_endpoint[ep] = {
            "count": len(isl_vals),
            "streamed_count": streamed.get(ep, 0),
            "ignore_eos_count": ignore_eos.get(ep, 0),
            "reasoning_effort_counts": dict(reasoning_efforts.get(ep, Counter()))
            or None,
            "isl": _stat_block(isl_vals),
            "requested_osl": _stat_block(osl_vals),
            "min_tokens": _quantiles(min_tokens.get(ep, [])),
        }
    return {"total_requests": total, "per_endpoint": per_endpoint}


def _stat_block(values: list[int]) -> dict[str, Any] | None:
    """Build the percentiles + histogram + unique_values block, or None when empty."""
    if not values:
        return None
    block = _quantiles(values)
    assert block is not None  # `_quantiles` only returns None for empty input
    block["unique_values"] = len(set(values))
    block["histogram"] = _histogram(values)
    return block
```

- [ ] **Step 4: Run unit tests, confirm they pass**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
```

Expected: 11 passed (7 from Task 1 + 4 from Task 2).

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): wire histogram + unique_values into summary"
```

---

## Task 3: Add `_render_histogram` helper

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (new `_render_histogram` function)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (new `TestRenderHistogram` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
from aiperf_mock_server.request_recorder import _render_histogram


class TestRenderHistogram:
    def test_header_line(self) -> None:
        hist = {"bin_edges": [0.0, 5.0, 10.0], "counts": [1, 3]}
        lines = _render_histogram("ISL", hist, count=4, unique=4)
        assert lines[0] == "    ISL histogram (2 bins, n=4, 4 unique)"

    def test_row_count_matches_bins(self) -> None:
        hist = {"bin_edges": [0.0, 5.0, 10.0, 15.0], "counts": [1, 2, 1]}
        lines = _render_histogram("ISL", hist, count=4, unique=4)
        assert len(lines) == 1 + 3  # header + 3 bin rows

    def test_bars_scaled_to_tallest_bin(self) -> None:
        hist = {"bin_edges": [0.0, 1.0, 2.0], "counts": [10, 5]}
        lines = _render_histogram("ISL", hist, count=15, unique=2)
        # First bin (max) should be fully filled — 20 block chars.
        assert lines[1].count("█") == 20
        # Second bin: 5/10 = 50% -> 10 filled, 10 unfilled.
        assert lines[2].count("█") == 10
        assert lines[2].count("░") == 10

    def test_empty_counts_returns_only_header(self) -> None:
        hist = {"bin_edges": [0.0, 0.0], "counts": []}
        lines = _render_histogram("ISL", hist, count=0, unique=0)
        assert lines == ["    ISL histogram (0 bins, n=0, 0 unique)"]

    def test_single_bin_renders(self) -> None:
        hist = {"bin_edges": [42.0, 42.0], "counts": [3]}
        lines = _render_histogram("ISL", hist, count=3, unique=1)
        assert len(lines) == 2
        # label_width=2 (from "42"), count_width=3 (floor), bar fully filled.
        assert lines[1] == "      42- 42    3 " + "█" * 20
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestRenderHistogram -v
```

Expected: `ImportError` for `_render_histogram`.

- [ ] **Step 3: Implement `_render_histogram` in `request_recorder.py`**

Add this function just after `_histogram`:

```python
def _render_histogram(
    metric: str,
    hist: dict[str, list[float]],
    count: int,
    unique: int,
) -> list[str]:
    """Render a histogram as 4-/6-space-indented stdout lines (header + bin rows).

    Bars are 20 chars wide, scaled so the tallest bin is full width. Bin range
    labels and the count column align within the histogram.
    """
    edges = hist["bin_edges"]
    counts = hist["counts"]
    num_bins = len(counts)
    header = f"    {metric} histogram ({num_bins} bins, n={count}, {unique} unique)"
    if not counts:
        return [header]
    max_count = max(counts) or 1
    bar_width = 20
    label_width = max(len(str(round(e))) for e in edges)
    count_width = max(3, len(str(max_count)))
    lines = [header]
    for i, c in enumerate(counts):
        filled = round(bar_width * c / max_count)
        bar = "█" * filled + "░" * (bar_width - filled)
        lo = round(edges[i])
        hi = round(edges[i + 1])
        lines.append(
            f"      {lo:>{label_width}d}- {hi:>{label_width}d}"
            f"  {c:>{count_width}d} {bar}"
        )
    return lines
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestRenderHistogram -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): add _render_histogram helper"
```

---

## Task 4: Wire `_render_histogram` into `_print_summary`

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (`_print_summary`)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (new `TestPrintSummary` class using `capsys`)

- [ ] **Step 1: Write the failing test using `capsys`**

Append to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
from aiperf_mock_server.request_recorder import _print_summary


class TestPrintSummary:
    def test_isl_histogram_block_printed(self, capsys) -> None:
        summary = {
            "total_requests": 4,
            "per_endpoint": {
                "/v1/chat/completions": {
                    "count": 4,
                    "streamed_count": 0,
                    "ignore_eos_count": 0,
                    "reasoning_effort_counts": None,
                    "isl": {
                        "min": 10.0, "max": 40.0, "mean": 25.0, "stdev": 12.91,
                        "p50": 25.0, "p90": 38.0, "p95": 39.0, "p99": 39.8,
                        "unique_values": 4,
                        "histogram": {
                            "bin_edges": [10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0,
                                          31.0, 34.0, 37.0, 40.0],
                            "counts": [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                        },
                    },
                    "requested_osl": None,
                    "min_tokens": None,
                },
            },
        }
        _print_summary(summary)
        out = capsys.readouterr().out
        assert "ISL histogram (10 bins, n=4, 4 unique)" in out

    def test_osl_histogram_skipped_when_null(self, capsys) -> None:
        summary = {
            "total_requests": 2,
            "per_endpoint": {
                "/v1/embeddings": {
                    "count": 2,
                    "streamed_count": 0,
                    "ignore_eos_count": 0,
                    "reasoning_effort_counts": None,
                    "isl": {
                        "min": 5.0, "max": 6.0, "mean": 5.5, "stdev": 0.5,
                        "p50": 5.5, "p90": 6.0, "p95": 6.0, "p99": 6.0,
                        "unique_values": 2,
                        "histogram": {"bin_edges": [5.0, 5.5, 6.0], "counts": [1, 1]},
                    },
                    "requested_osl": None,
                    "min_tokens": None,
                },
            },
        }
        _print_summary(summary)
        out = capsys.readouterr().out
        assert "ISL histogram" in out
        assert "OSL histogram" not in out
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestPrintSummary -v
```

Expected: 2 fail — histogram lines are not yet emitted.

- [ ] **Step 3: Extend `_print_summary` in `request_recorder.py`**

Replace the current `_print_summary` body with this version. The change: after the existing per-metric percentile lines for ISL and OSL, accumulate histogram lines and emit them after both stats lines, before `min_tokens` / `ignore_eos` / `reasoning_effort`:

```python
def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nRequest distribution ({summary['total_requests']} requests)")
    print("─" * 46)
    for ep, stats in summary["per_endpoint"].items():
        print(f"  {ep}  n={stats['count']}")
        for label, s in (("ISL", stats["isl"]), ("OSL", stats["requested_osl"])):
            if s is None:
                print(f"    {label}    n/a")
            else:
                print(
                    f"    {label}    mean {s['mean']:7.1f}"
                    f"   p50 {s['p50']:5.0f}   p99 {s['p99']:5.0f}"
                )
        for label, s in (("ISL", stats["isl"]), ("OSL", stats["requested_osl"])):
            if s is None or s.get("histogram") is None:
                continue
            for line in _render_histogram(
                label, s["histogram"], stats["count"], s["unique_values"]
            ):
                print(line)
        mn = stats["min_tokens"]
        if mn is not None:
            print(
                f"    min_tokens  mean {mn['mean']:7.1f}   p50 {mn['p50']:5.0f}"
            )
        if stats["ignore_eos_count"]:
            print(f"    ignore_eos=true: {stats['ignore_eos_count']}")
        if stats["reasoning_effort_counts"]:
            print(f"    reasoning_effort: {stats['reasoning_effort_counts']}")
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
uv run pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
```

Expected: 18 passed (7 + 4 + 5 + 2).

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): render histograms in stdout summary"
```

---

## Task 5: Extend the integration test

**Files:**
- Modify: `tests/integration/test_mock_server_record_requests.py`

- [ ] **Step 1: Read the existing test to find the assertion block**

```bash
grep -n "chat_stats\|emb_stats\|requested_osl" \
  tests/integration/test_mock_server_record_requests.py
```

Note the line range of the existing assertion block on `chat_stats` and `emb_stats`.

- [ ] **Step 2: Add new assertions**

In `tests/integration/test_mock_server_record_requests.py`, after the existing `chat_stats` assertions and the existing `emb_stats` assertions in `test_records_per_request_isl_and_requested_osl`, append:

```python
        # Histogram + unique_values on the chat ISL block.
        assert chat_stats["isl"]["histogram"] is not None
        chat_isl_hist = chat_stats["isl"]["histogram"]
        assert len(chat_isl_hist["bin_edges"]) == len(chat_isl_hist["counts"]) + 1
        assert sum(chat_isl_hist["counts"]) == chat_stats["count"]
        assert chat_stats["isl"]["unique_values"] >= 1

        # Resolved requested_osl spans six distinct values across the chat fixture:
        # max_tokens in {16, 32, 64, 128, 256} on five requests plus
        # max_completion_tokens=192 on the sixth.
        assert chat_stats["requested_osl"]["unique_values"] == 6
        chat_osl_hist = chat_stats["requested_osl"]["histogram"]
        assert chat_osl_hist is not None
        assert sum(chat_osl_hist["counts"]) == chat_stats["count"]

        # Embeddings: requested_osl stays None (no histogram on it), ISL still has one.
        assert emb_stats["requested_osl"] is None
        assert emb_stats["isl"]["histogram"] is not None
        assert emb_stats["isl"]["unique_values"] >= 1
```

- [ ] **Step 3: Run the integration test**

```bash
uv run pytest tests/integration/test_mock_server_record_requests.py -v -m integration
```

Expected: all 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_mock_server_record_requests.py
git commit -s -m "test(mock-server): cover histogram + unique_values in recorder summary"
```

---

## Task 6: Update the mock-server README

**Files:**
- Modify: `tests/aiperf_mock_server/README.md`

- [ ] **Step 1: Locate the "Output format" and "Summary" sections**

```bash
grep -n "Output format\|Summary\|requested_osl" tests/aiperf_mock_server/README.md
```

Note the line numbers of the JSON example block (`"isl": {...}`) and the stdout sample block under "Summary".

- [ ] **Step 2: Refresh the JSON example**

Replace the existing per-metric JSON example in the README with one that includes the new fields:

```json
"isl": {
  "min": 207.0, "max": 1821.0, "mean": 1010.48, "stdev": 480.80,
  "p50": 997.5, "p90": 1684.8, "p95": 1745.55, "p99": 1819.02,
  "unique_values": 19,
  "histogram": {
    "bin_edges": [207.0, 301.94, 396.88, 491.82, 586.76, 681.71, 776.65,
                  871.59, 966.53, 1061.47, 1156.41, 1251.35, 1346.29,
                  1441.24, 1536.18, 1631.12, 1726.06, 1821.0],
    "counts":    [7, 6, 5, 4, 6, 8, 7, 10, 11, 9, 8, 6, 5, 4, 3, 1, 0]
  }
}
```

- [ ] **Step 3: Refresh the stdout sample**

Replace the stdout sample in the "Summary" section with one that includes the histogram block:

```
Request distribution (100 requests)
──────────────────────────────────────────────
  /v1/completions  n=100
    ISL    mean  1010.5   p50   998   p99  1819
    OSL    mean   127.5   p50   129   p99   229
    ISL histogram (17 bins, n=100, 19 unique)
       207-  302   7 ████░░░░░░░░░░░░░░░░
       302-  397   6 ███░░░░░░░░░░░░░░░░░
       ... (17 rows total)
      1726- 1821   1 ░░░░░░░░░░░░░░░░░░░░
    OSL histogram (10 bins, n=100, 11 unique)
        25-   46   3 ██░░░░░░░░░░░░░░░░░░
        ... (10 rows total)
       210-  230   6 ████░░░░░░░░░░░░░░░░
```

- [ ] **Step 4: Add a one-liner to the "Properties" or "Fields" description**

In the table or field list that describes the JSON record fields, add an entry for `unique_values` and `histogram` immediately after the `requested_osl` field:

```markdown
| `unique_values` | Count of distinct values observed for this metric (`isl`, `requested_osl`). |
| `histogram` | Equal-width histogram with parallel `bin_edges` (len N+1) and `counts` (len N). `null` when no values were observed (e.g. `requested_osl` on embeddings). Bucketing: `num_bins = max(10, ceil((max-min)/100))`. |
```

If the existing README uses a prose paragraph instead of a table, fold the description into the surrounding text in matching style.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/README.md
git commit -s -m "docs(mock-server): document recorder histogram + unique_values"
```

---

## Task 7: Pre-merge verification

- [ ] **Step 1: Run lint + format**

```bash
ruff format tests/aiperf_mock_server/request_recorder.py \
            tests/aiperf_mock_server/README.md \
            tests/unit/aiperf_mock_server/test_request_recorder.py \
            tests/integration/test_mock_server_record_requests.py
ruff check --fix tests/aiperf_mock_server/request_recorder.py \
                 tests/unit/aiperf_mock_server/test_request_recorder.py \
                 tests/integration/test_mock_server_record_requests.py
```

Expected: `All checks passed!` and `N files reformatted` / `N files unchanged`.

- [ ] **Step 2: Run unit tests**

```bash
uv run pytest tests/unit/aiperf_mock_server -v
```

Expected: 18 passed.

- [ ] **Step 3: Run the integration tests**

```bash
uv run pytest tests/integration/test_mock_server_record_requests.py -v -m integration
```

Expected: 3 passed.

- [ ] **Step 4: Run pre-commit on changed files**

```bash
pre-commit run --files \
  tests/aiperf_mock_server/request_recorder.py \
  tests/aiperf_mock_server/README.md \
  tests/unit/aiperf_mock_server/__init__.py \
  tests/unit/aiperf_mock_server/test_request_recorder.py \
  tests/integration/test_mock_server_record_requests.py
```

Expected: all hooks pass (or auto-fixes applied, which need a follow-up commit).

- [ ] **Step 5: If pre-commit modified files, amend the last commit**

```bash
git status -s
# if there are modifications:
git add -u
git commit -s --amend --no-edit
```

- [ ] **Step 6: Verify the branch state**

```bash
git log --oneline origin/main..HEAD
```

Expected: 6 commits on top of `0165f65d`, in order:
1. `feat(mock-server): add _histogram helper for recorder summary`
2. `feat(mock-server): wire histogram + unique_values into summary`
3. `feat(mock-server): add _render_histogram helper`
4. `feat(mock-server): render histograms in stdout summary`
5. `test(mock-server): cover histogram + unique_values in recorder summary`
6. `docs(mock-server): document recorder histogram + unique_values`

(Plus the spec doc commit `docs(spec): design for recorder histogram + unique-count stats` which already exists.)
