# Recorder Vocabulary-Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-endpoint `vocab_distribution` block to the mock-server request recorder that surfaces token-id coverage, top-N concentration, Shannon entropy, and an 80-bucket sparkline across the full vocab — in both `<path>.summary.json` and stdout.

**Architecture:** `RequestRecorder` gains one `Counter[int]` per endpoint that accumulates token IDs as `tokenizer.encode(text)` runs (currently the IDs are discarded after `len()`). Three pure helpers (`_compute_shape_80`, `_vocab_distribution`, `_render_vocab_lines`) compute the stats and stdout block at shutdown. `_build_summary` and `_print_summary` get the new block; the stdout pass also gains blank lines between every per-endpoint block.

**Tech Stack:** Python 3.13, stdlib only (`collections.Counter`, `math`), `pytest`, `orjson`.

**Spec:** [`docs/superpowers/specs/2026-05-18-recorder-vocab-distribution-design.md`](../specs/2026-05-18-recorder-vocab-distribution-design.md)

---

## Task 1: Capture token IDs in `RequestRecorder`

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (`RequestRecorder.__init__`, `open`, `record`)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (new `TestRecorderTokenIdTracking` class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
from collections import Counter

from aiperf_mock_server.request_recorder import RequestRecorder


class _FakeTokenizer:
    """Minimal stub for unit tests that drive `RequestRecorder.record()`."""

    def __init__(self, vocab_size: int, encodings: dict[str, list[int]]) -> None:
        self._vocab_size = vocab_size
        self._encodings = encodings

    def __len__(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return list(self._encodings.get(text, []))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def _make_recorder(tmp_path, tokenizer: _FakeTokenizer) -> RequestRecorder:
    path = tmp_path / "rec.jsonl"
    r = RequestRecorder(
        path=str(path),
        tokenizer_name="fake",
        tokenizer_revision="main",
        trust_remote_code=False,
    )
    # Bypass open() so we don't need to wire `Tokenizer.from_pretrained`.
    r._tokenizer = tokenizer
    r._vocab_size = len(tokenizer)
    r._vocab_size_source = "tokenizer"
    r._file = open(path, "wb")
    return r


class TestRecorderTokenIdTracking:
    def test_record_updates_vocab_counter(self, tmp_path) -> None:
        tok = _FakeTokenizer(vocab_size=100, encodings={"hello": [1, 2, 1, 3]})
        r = _make_recorder(tmp_path, tok)
        r.record(
            ts=0.0,
            endpoint="/v1/chat/completions",
            request_id="x",
            model="m",
            text="hello",
            stream=False,
            osl_fingerprint={},
        )
        assert r._vocab_counts["/v1/chat/completions"] == Counter({1: 2, 2: 1, 3: 1})
        r._file.close()

    def test_record_accumulates_across_calls(self, tmp_path) -> None:
        tok = _FakeTokenizer(
            vocab_size=100, encodings={"a": [1, 1], "b": [2, 3]}
        )
        r = _make_recorder(tmp_path, tok)
        for text in ("a", "b", "a"):
            r.record(
                ts=0.0,
                endpoint="/v1/chat/completions",
                request_id="x",
                model="m",
                text=text,
                stream=False,
                osl_fingerprint={},
            )
        assert r._vocab_counts["/v1/chat/completions"] == Counter({1: 4, 2: 1, 3: 1})
        r._file.close()

    def test_record_segregates_counts_by_endpoint(self, tmp_path) -> None:
        tok = _FakeTokenizer(vocab_size=100, encodings={"x": [5, 6]})
        r = _make_recorder(tmp_path, tok)
        r.record(0.0, "/v1/chat/completions", "x", "m", "x", False, {})
        r.record(0.0, "/v1/embeddings", "x", "m", "x", False, {})
        assert r._vocab_counts["/v1/chat/completions"] == Counter({5: 1, 6: 1})
        assert r._vocab_counts["/v1/embeddings"] == Counter({5: 1, 6: 1})
        assert list(r._vocab_counts.keys()) == ["/v1/chat/completions", "/v1/embeddings"]
        r._file.close()

    def test_open_sets_vocab_size_from_tokenizer(self, tmp_path) -> None:
        # Same path as production: from_pretrained → len(tokenizer).
        # Verifies the `open()` integration captures vocab_size + source.
        r = RequestRecorder(
            path=str(tmp_path / "rec.jsonl"),
            tokenizer_name="builtin",
            tokenizer_revision="main",
            trust_remote_code=False,
        )
        r.open()
        try:
            assert isinstance(r._vocab_size, int)
            assert r._vocab_size > 0
            assert r._vocab_size_source == "tokenizer"
        finally:
            r.close()
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
cd /Users/fdinatale/Code/aiperf/.worktrees/mock-server-request-recorder
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestRecorderTokenIdTracking -v
```

Expected: `AttributeError: 'RequestRecorder' object has no attribute '_vocab_counts'` (or similar).

- [ ] **Step 3: Wire token-id tracking into `RequestRecorder`**

In `tests/aiperf_mock_server/request_recorder.py`:

(a) Inside `RequestRecorder.__init__`, add the per-endpoint Counter dict alongside the existing per-endpoint state. Find the existing `self._isls: dict[str, list[int]] = defaultdict(list)` line and add immediately below it:

```python
        self._vocab_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self._vocab_size: int | None = None
        self._vocab_size_source: str = "tokenizer"
```

(b) Inside `RequestRecorder.open()`, after the `self._tokenizer = ...` assignment and before `self._file = open(...)`, capture the vocab size:

```python
        try:
            self._vocab_size = len(self._tokenizer)
            self._vocab_size_source = "tokenizer"
        except (TypeError, AttributeError):
            # Tokenizer doesn't expose __len__; we'll derive from observed ids
            # at summary time.
            self._vocab_size = None
            self._vocab_size_source = "observed"
```

(c) Inside `RequestRecorder.record()`, change the existing `isl = len(self._tokenizer.encode(text))` (or equivalent) line to keep the ids:

```python
        try:
            ids = self._tokenizer.encode(text)
        except Exception:
            logger.exception(
                "recorder: tokenization failed for %s %s", endpoint, request_id
            )
            return
        isl = len(ids)
        self._vocab_counts[endpoint].update(ids)
```

Note: read the current `record()` body before editing — the existing tokenize call is wrapped in a try/except. Preserve that structure; only the variable name and the `update(...)` line are new. If the existing code stores `isl` from a single `encode(...).__len__` call, replace it with the two-line `ids = encode(...); isl = len(ids)` pattern shown above.

- [ ] **Step 4: Run tests, confirm they pass**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestRecorderTokenIdTracking -v
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
uv run --no-sync pytest tests/integration/test_mock_server_record_requests.py -m integration -v
```

Expected: 4 new tests pass; all prior unit + integration tests still pass.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): capture per-endpoint token-id Counter in recorder"
```

---

## Task 2: `_compute_shape_80` helper

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (new helper)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (new `TestComputeShape80` class)

- [ ] **Step 1: Write the failing tests**

Append:

```python
from aiperf_mock_server.request_recorder import _compute_shape_80


class TestComputeShape80:
    def test_length_is_always_80(self) -> None:
        assert len(_compute_shape_80(Counter({0: 1, 99999: 1}), 100000)) == 80

    def test_sum_of_buckets_equals_total_observations(self) -> None:
        counts = Counter({0: 10, 1: 20, 50: 30, 99: 40, 5000: 50})
        shape = _compute_shape_80(counts, 10000)
        assert sum(shape) == 10 + 20 + 30 + 40 + 50

    def test_id_at_bucket_boundary_lands_in_lower_bucket(self) -> None:
        # vocab_size=80, bucket width = 1, so id 5 must land in bucket 5.
        shape = _compute_shape_80(Counter({5: 7}), 80)
        assert shape[5] == 7
        assert sum(shape) == 7

    def test_max_id_lands_in_last_bucket(self) -> None:
        # Highest id is vocab_size-1; spec says last bucket is closed on both
        # ends so vocab_size-1 ends up in bucket 79, not lost.
        shape = _compute_shape_80(Counter({999: 3}), 1000)
        assert shape[-1] == 3
        assert sum(shape) == 3

    def test_empty_counter_returns_all_zero_buckets(self) -> None:
        shape = _compute_shape_80(Counter(), 1000)
        assert shape == [0] * 80

    def test_buckets_partition_vocab_evenly(self) -> None:
        # vocab_size=800, bucket width = 10. Place one observation in each
        # bucket's lower bound to verify equal-width partitioning.
        counts = Counter({i * 10: 1 for i in range(80)})
        shape = _compute_shape_80(counts, 800)
        assert shape == [1] * 80

    def test_ids_above_vocab_size_are_dropped(self) -> None:
        # Defensive: if the tokenizer ever returns an id ≥ vocab_size we drop
        # it rather than silently miscount the last bucket.
        shape = _compute_shape_80(Counter({100: 5, 99: 3}), 100)
        assert shape[-1] == 3  # id=99 (last valid)
        assert sum(shape) == 3  # id=100 dropped
```

- [ ] **Step 2: Run, confirm fails**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestComputeShape80 -v
```

Expected: `ImportError` for `_compute_shape_80`.

- [ ] **Step 3: Implement `_compute_shape_80`**

In `tests/aiperf_mock_server/request_recorder.py`, place the new helper just before `_quantiles` (alongside `_histogram`):

```python
def _compute_shape_80(counts: Counter[int], vocab_size: int) -> list[int]:
    """Sum counts into 80 equal-width buckets over [0, vocab_size).

    Each bucket spans `vocab_size / 80` token ids. The last bucket is closed
    on its upper end so `vocab_size - 1` lands in bucket 79 (instead of just
    past it). Ids ≥ `vocab_size` are dropped — defensive only; should not
    occur with a well-behaved tokenizer.
    """
    shape = [0] * 80
    if vocab_size <= 0:
        return shape
    width = vocab_size / 80
    for token_id, count in counts.items():
        if token_id < 0 or token_id >= vocab_size:
            continue
        idx = int(token_id / width)
        if idx >= 80:
            idx = 79  # float-drift guard, mirrors `_histogram`
        shape[idx] += count
    return shape
```

- [ ] **Step 4: Run, confirm passes**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestComputeShape80 -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): add _compute_shape_80 helper for vocab sparkline"
```

---

## Task 3: `_vocab_distribution` helper

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (new helper + import)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (new `TestVocabDistribution` class)

- [ ] **Step 1: Write the failing tests**

Append:

```python
from aiperf_mock_server.request_recorder import _vocab_distribution


def _id_to_text(i: int) -> str:
    return f"<tok-{i}>"


class TestVocabDistribution:
    def test_returns_none_for_empty_counter(self) -> None:
        assert _vocab_distribution(Counter(), 100, "tokenizer", _id_to_text) is None

    def test_unique_ids_and_coverage_pct(self) -> None:
        vd = _vocab_distribution(
            Counter({1: 5, 2: 5, 3: 5}), 1000, "tokenizer", _id_to_text
        )
        assert vd is not None
        assert vd["vocab_size"] == 1000
        assert vd["vocab_size_source"] == "tokenizer"
        assert vd["unique_ids"] == 3
        assert vd["coverage_pct"] == 0.3
        assert vd["total_tokens"] == 15

    def test_top_tokens_length_caps_at_10(self) -> None:
        counts = Counter({i: 100 - i for i in range(20)})
        vd = _vocab_distribution(counts, 100, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["top_tokens"]) == 10
        # Sorted descending by count
        assert vd["top_tokens"][0]["count"] >= vd["top_tokens"][-1]["count"]
        assert vd["top_tokens"][0] == {"id": 0, "text": "<tok-0>", "count": 100}

    def test_top_tokens_length_matches_unique_when_below_ten(self) -> None:
        vd = _vocab_distribution(Counter({1: 3, 2: 2}), 100, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["top_tokens"]) == 2

    def test_top_tokens_falls_back_to_id_marker_when_decode_raises(self) -> None:
        def raising_decode(i: int) -> str:
            if i == 7:
                raise RuntimeError("boom")
            return f"<tok-{i}>"

        vd = _vocab_distribution(
            Counter({7: 100, 8: 50}), 100, "tokenizer", raising_decode
        )
        assert vd is not None
        # id 7 was the most frequent, so it appears first in top_tokens.
        assert vd["top_tokens"][0] == {"id": 7, "text": "<id=7>", "count": 100}
        assert vd["top_tokens"][1] == {"id": 8, "text": "<tok-8>", "count": 50}

    def test_top_10_concentration_pct(self) -> None:
        # Top 10 of these 11 ids account for 1000 of 1010 total = 99.0099%
        counts = Counter({i: 100 for i in range(10)})
        counts[99] = 10
        vd = _vocab_distribution(counts, 100, "tokenizer", _id_to_text)
        assert vd is not None
        assert abs(vd["top_10_concentration_pct"] - 99.0099) < 0.01

    def test_entropy_zero_for_single_token(self) -> None:
        vd = _vocab_distribution(Counter({42: 100}), 1000, "tokenizer", _id_to_text)
        assert vd is not None
        assert vd["entropy_bits"] == 0.0
        assert vd["max_entropy_bits"] == pytest.approx(math.log2(1000))

    def test_entropy_at_max_for_uniform_sampling(self) -> None:
        # Perfectly uniform sampling over the full vocab → entropy_bits == log2(V).
        counts = Counter({i: 5 for i in range(64)})
        vd = _vocab_distribution(counts, 64, "tokenizer", _id_to_text)
        assert vd is not None
        assert vd["entropy_bits"] == pytest.approx(math.log2(64))
        assert vd["max_entropy_bits"] == pytest.approx(math.log2(64))

    def test_shape_80_length(self) -> None:
        counts = Counter({i: 1 for i in range(80)})
        vd = _vocab_distribution(counts, 80, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["shape_80"]) == 80
        assert sum(vd["shape_80"]) == 80

    def test_frequencies_full_table_with_string_keys(self) -> None:
        counts = Counter({1: 5, 2: 3, 99: 1})
        vd = _vocab_distribution(counts, 100, "tokenizer", _id_to_text)
        assert vd is not None
        # JSON dict keys must be strings.
        assert vd["frequencies"] == {"1": 5, "2": 3, "99": 1}

    def test_vocab_size_source_observed_path_uses_max_id_plus_one(self) -> None:
        # When source == "observed" we don't trust the passed vocab_size if
        # max(observed) >= it. The helper should report the source verbatim
        # and use vocab_size as given for coverage math. Observed-fallback
        # responsibility lives in the caller (open()/record() machinery), so
        # this test just asserts the field is passed through.
        vd = _vocab_distribution(
            Counter({1: 1, 5: 1}), 10, "observed", _id_to_text
        )
        assert vd is not None
        assert vd["vocab_size_source"] == "observed"
        assert vd["vocab_size"] == 10
```

Add the imports at the top of the test file (alongside the existing imports):

```python
import math

import pytest
```

(If `math` and `pytest` are already imported, skip the addition.)

- [ ] **Step 2: Run, confirm fails**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestVocabDistribution -v
```

Expected: `ImportError` for `_vocab_distribution`.

- [ ] **Step 3: Implement `_vocab_distribution`**

Add to the imports section of `tests/aiperf_mock_server/request_recorder.py`:

```python
from collections.abc import Callable
```

Place the helper just after `_compute_shape_80`:

```python
def _vocab_distribution(
    counts: Counter[int],
    vocab_size: int,
    source: str,
    decode_fn: Callable[[int], str],
) -> dict[str, Any] | None:
    """Build the vocab_distribution JSON block, or None if there are no observations.

    `decode_fn` maps a token id to its text representation. If `decode_fn`
    raises for a given id, that entry in `top_tokens` falls back to
    ``"<id=N>"``.
    """
    total = sum(counts.values())
    if total == 0:
        return None

    sorted_items = counts.most_common(10)
    top_tokens: list[dict[str, Any]] = []
    for token_id, count in sorted_items:
        try:
            text = decode_fn(token_id)
        except Exception:
            text = f"<id={token_id}>"
        top_tokens.append({"id": int(token_id), "text": text, "count": int(count)})

    top_10_count = sum(count for _, count in sorted_items)
    top_10_concentration_pct = round(top_10_count / total * 100, 4)

    # Shannon entropy in bits.
    entropy_bits = 0.0
    for count in counts.values():
        p = count / total
        entropy_bits -= p * math.log2(p)
    max_entropy_bits = math.log2(vocab_size) if vocab_size > 1 else 0.0

    return {
        "vocab_size": int(vocab_size),
        "vocab_size_source": source,
        "unique_ids": len(counts),
        "coverage_pct": round(len(counts) / vocab_size * 100, 4) if vocab_size else 0.0,
        "total_tokens": int(total),
        "top_10_concentration_pct": top_10_concentration_pct,
        "entropy_bits": round(entropy_bits, 4),
        "max_entropy_bits": round(max_entropy_bits, 4),
        "top_tokens": top_tokens,
        "shape_80": _compute_shape_80(counts, vocab_size),
        "frequencies": {str(tid): int(c) for tid, c in counts.items()},
    }
```

- [ ] **Step 4: Run, confirm passes**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestVocabDistribution -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): add _vocab_distribution helper for recorder summary"
```

---

## Task 4: `_render_vocab_lines` helper (stdout block)

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (new helper)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (new `TestRenderVocabShape` class)

- [ ] **Step 1: Write the failing tests**

Append:

```python
from aiperf_mock_server.request_recorder import _render_vocab_lines


class TestRenderVocabShape:
    def _make_vd(
        self,
        unique_ids: int = 5,
        vocab_size: int = 1000,
        coverage_pct: float = 0.5,
        top_10_concentration_pct: float = 50.0,
        entropy_bits: float = 4.0,
        max_entropy_bits: float = 10.0,
        top_tokens: list | None = None,
        shape_80: list | None = None,
    ) -> dict:
        if top_tokens is None:
            top_tokens = [
                {"id": i, "text": f"<t{i}>", "count": 100 - i * 10}
                for i in range(5)
            ]
        if shape_80 is None:
            shape_80 = [10, 5, 2] + [0] * 77
        return {
            "vocab_size": vocab_size,
            "vocab_size_source": "tokenizer",
            "unique_ids": unique_ids,
            "coverage_pct": coverage_pct,
            "total_tokens": sum(shape_80),
            "top_10_concentration_pct": top_10_concentration_pct,
            "entropy_bits": entropy_bits,
            "max_entropy_bits": max_entropy_bits,
            "top_tokens": top_tokens,
            "shape_80": shape_80,
            "frequencies": {},
        }

    def test_headline_line_format(self) -> None:
        lines = _render_vocab_lines(self._make_vd(
            unique_ids=5234,
            vocab_size=151936,
            coverage_pct=3.4438,
            top_10_concentration_pct=47.2,
            entropy_bits=8.23,
            max_entropy_bits=17.21,
        ))
        assert lines[0] == (
            "    Vocab  used 5234/151936 (3.4%)  top-10 cover 47%"
            "  entropy 8.2/17.2 bits"
        )

    def test_top_line_format(self) -> None:
        vd = self._make_vd(top_tokens=[
            {"id": 1, "text": " the", "count": 3201},
            {"id": 2, "text": " a", "count": 2890},
        ])
        lines = _render_vocab_lines(vd)
        assert lines[1] == '      top: " the" 3201, " a" 2890'

    def test_top_line_caps_at_5(self) -> None:
        vd = self._make_vd(top_tokens=[
            {"id": i, "text": f"<t{i}>", "count": 100 - i} for i in range(10)
        ])
        lines = _render_vocab_lines(vd)
        # Only first 5 entries appear in the stdout line.
        assert lines[1].count(",") == 4

    def test_top_line_falls_back_to_unquoted_id_marker(self) -> None:
        vd = self._make_vd(top_tokens=[
            {"id": 7, "text": "<id=7>", "count": 100},
            {"id": 8, "text": " ok", "count": 50},
        ])
        lines = _render_vocab_lines(vd)
        assert "<id=7> 100" in lines[1]
        assert '" ok" 50' in lines[1]

    def test_blank_line_before_shape(self) -> None:
        lines = _render_vocab_lines(self._make_vd())
        # lines[0] = Vocab headline, lines[1] = top, lines[2] = blank
        assert lines[2] == ""

    def test_shape_header_line(self) -> None:
        lines = _render_vocab_lines(self._make_vd(vocab_size=151936))
        assert lines[3] == "    vocab shape  (80 buckets over id 0..151935, log-y)"

    def test_sparkline_is_80_chars(self) -> None:
        lines = _render_vocab_lines(self._make_vd(
            shape_80=[10, 5, 2, 1] + [0] * 76,
        ))
        # lines[4] is the sparkline, indented 4 spaces.
        sparkline = lines[4][4:]
        assert len(sparkline) == 80

    def test_zero_bucket_renders_as_space(self) -> None:
        shape = [10] + [0] * 79
        lines = _render_vocab_lines(self._make_vd(shape_80=shape))
        sparkline = lines[4][4:]
        # First bucket is the tallest (█); the rest are zero (space).
        assert sparkline[0] == "█"
        assert sparkline[1:] == " " * 79

    def test_log_y_makes_small_bars_visible(self) -> None:
        # One huge bucket and several small ones — linear scaling would render
        # all the small bars at ▁ or below. Log-y must lift them into visible
        # block characters.
        shape = [1000] + [1] * 79
        lines = _render_vocab_lines(self._make_vd(shape_80=shape))
        sparkline = lines[4][4:]
        assert sparkline[0] == "█"
        # The small buckets must render as non-space (i.e. a visible block).
        # Log scaling guarantees log1p(1)/log1p(1000) ≈ 0.10, well above the
        # ▁ threshold of 1/8 = 0.125 only just; allow ▁ as the minimum.
        block_chars = set("▁▂▃▄▅▆▇█")
        for ch in sparkline[1:]:
            assert ch in block_chars

    def test_axis_tick_line(self) -> None:
        lines = _render_vocab_lines(self._make_vd(vocab_size=151936))
        # lines[5] is the axis tick line. The leftmost label '0' starts at
        # column 4 (after the indent); the rightmost ('152K') ends at column
        # 4 + 80 = 84.
        ticks = lines[5]
        assert ticks.startswith("    0")
        assert ticks.rstrip().endswith("152K")
        # Includes the three middle ticks at 25%/50%/75% positions.
        assert "38K" in ticks
        assert "76K" in ticks
        assert "114K" in ticks
```

- [ ] **Step 2: Run, confirm fails**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestRenderVocabShape -v
```

Expected: `ImportError` for `_render_vocab_lines`.

- [ ] **Step 3: Implement `_render_vocab_lines`**

Add to `tests/aiperf_mock_server/request_recorder.py`, placed just after `_render_histogram`:

```python
_BLOCK_CHARS = "▁▂▃▄▅▆▇█"


def _format_top_tokens_line(top_tokens: list[dict[str, Any]]) -> str:
    """Format the `top:` line of the vocab stdout block (first 5 entries)."""
    pieces: list[str] = []
    for entry in top_tokens[:5]:
        text = entry["text"]
        count = entry["count"]
        if isinstance(text, str) and text.startswith("<id=") and text.endswith(">"):
            pieces.append(f"{text} {count}")
        else:
            pieces.append(f'"{text}" {count}')
    return "      top: " + ", ".join(pieces)


def _format_tick(value: int) -> str:
    """Right-side axis tick formatting: '0' / '38K' / '152K' (no decimals)."""
    if value < 1000:
        return str(value)
    return f"{value // 1000}K"


def _render_vocab_lines(vd: dict[str, Any]) -> list[str]:
    """Return the 6-line stdout block (headline, top, blank, shape header,
    sparkline, axis-ticks) for one endpoint's vocab_distribution.

    Layout (4-space indent on top-level rows, 6-space indent on `top:`):
        ``    Vocab  used N/V (P%)  top-10 cover X%  entropy E/M bits``
        ``      top: "tok1" c1, "tok2" c2, ...``
        ``    ``
        ``    vocab shape  (80 buckets over id 0..V-1, log-y)``
        ``    [80-char sparkline]``
        ``    0 ... K_q1 ... K_q2 ... K_q3 ... K_max``
    """
    headline = (
        f"    Vocab  used {vd['unique_ids']}/{vd['vocab_size']}"
        f" ({vd['coverage_pct']:.1f}%)"
        f"  top-10 cover {vd['top_10_concentration_pct']:.0f}%"
        f"  entropy {vd['entropy_bits']:.1f}/{vd['max_entropy_bits']:.1f} bits"
    )
    top_line = _format_top_tokens_line(vd["top_tokens"])
    shape_header = (
        f"    vocab shape  (80 buckets over id 0..{vd['vocab_size'] - 1}, log-y)"
    )

    shape = vd["shape_80"]
    max_count = max(shape) if shape else 0
    if max_count <= 0:
        sparkline = " " * 80
    else:
        log_max = math.log1p(max_count)
        sparkline_chars: list[str] = []
        for count in shape:
            if count <= 0:
                sparkline_chars.append(" ")
                continue
            # Log-y: map [1, max_count] → [1, 8] block-char index.
            ratio = math.log1p(count) / log_max
            idx = min(7, max(0, int(ratio * 8) - (1 if ratio == 1.0 else 0)))
            # Clamp so any non-zero count produces at least ▁.
            idx = max(0, idx)
            sparkline_chars.append(_BLOCK_CHARS[idx])
        sparkline = "".join(sparkline_chars)

    vocab_size = vd["vocab_size"]
    tick_positions = (0, vocab_size // 4, vocab_size // 2, (3 * vocab_size) // 4, vocab_size)
    tick_labels = [_format_tick(p) for p in tick_positions]
    # Each tick sits at the column index where its bucket starts.
    columns = (0, 20, 40, 60, 79)  # 80-char sparkline: 0%, 25%, 50%, 75%, 100%.
    tick_line = list(" " * 80)
    for col, label in zip(columns, tick_labels, strict=True):
        # Place label so it doesn't run past the sparkline width.
        start = min(col, 80 - len(label))
        for i, ch in enumerate(label):
            tick_line[start + i] = ch

    return [
        headline,
        top_line,
        "",
        shape_header,
        "    " + sparkline,
        "    " + "".join(tick_line).rstrip(),
    ]
```

- [ ] **Step 4: Run, confirm passes**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestRenderVocabShape -v
```

Expected: 10 passed.

If any test fails because of off-by-one rounding on the log-y mapping, adjust the `idx` formula so `count == max_count` produces `█` (index 7) and `count == 1` produces at least `▁` (index 0).

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): add _render_vocab_lines for stdout vocab block"
```

---

## Task 5: Wire vocab block into `_build_summary`

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (`_build_summary`)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (extend `TestBuildSummary` if present; otherwise add a `TestBuildSummaryVocab` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
from aiperf_mock_server.request_recorder import _build_summary


class TestBuildSummaryVocab:
    def test_endpoint_block_contains_vocab_distribution(self) -> None:
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/chat/completions": [10, 20]}),
            osls=defaultdict(list, {"/v1/chat/completions": [5, 5]}),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
            vocab_counts={"/v1/chat/completions": Counter({1: 3, 2: 2})},
            vocab_size=100,
            vocab_size_source="tokenizer",
            decode_fn=_id_to_text,
        )
        ep = summary["per_endpoint"]["/v1/chat/completions"]
        assert ep["vocab_distribution"] is not None
        assert ep["vocab_distribution"]["unique_ids"] == 2
        assert ep["vocab_distribution"]["total_tokens"] == 5

    def test_vocab_distribution_is_none_for_endpoint_with_no_observations(self) -> None:
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/embeddings": [10, 20]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
            vocab_counts={"/v1/embeddings": Counter()},
            vocab_size=100,
            vocab_size_source="tokenizer",
            decode_fn=_id_to_text,
        )
        ep = summary["per_endpoint"]["/v1/embeddings"]
        assert ep["vocab_distribution"] is None
```

- [ ] **Step 2: Run, confirm fails**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestBuildSummaryVocab -v
```

Expected: `TypeError: _build_summary() got an unexpected keyword argument 'vocab_counts'` (or similar).

- [ ] **Step 3: Modify `_build_summary` signature and body**

In `tests/aiperf_mock_server/request_recorder.py`, find the existing `_build_summary` function. Add four new keyword arguments (`vocab_counts`, `vocab_size`, `vocab_size_source`, `decode_fn`) with sensible defaults so the existing call sites that don't pass them continue to work; then add the `vocab_distribution` field to each per-endpoint dict.

```python
def _build_summary(
    total: int,
    isls: dict[str, list[int]],
    osls: dict[str, list[int]],
    min_tokens: dict[str, list[int]],
    streamed: dict[str, int],
    ignore_eos: dict[str, int],
    reasoning_efforts: dict[str, Counter[str]],
    vocab_counts: dict[str, Counter[int]] | None = None,
    vocab_size: int | None = None,
    vocab_size_source: str = "tokenizer",
    decode_fn: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    per_endpoint: dict[str, Any] = {}
    vocab_counts = vocab_counts or {}
    for ep in sorted(isls.keys()):
        isl_vals = isls[ep]
        osl_vals = osls.get(ep, [])
        ep_vocab_counter = vocab_counts.get(ep, Counter())
        if vocab_size is not None and decode_fn is not None:
            vd = _vocab_distribution(
                ep_vocab_counter,
                _resolve_vocab_size(vocab_size, vocab_size_source, ep_vocab_counter),
                vocab_size_source,
                decode_fn,
            )
        else:
            vd = None
        per_endpoint[ep] = {
            "count": len(isl_vals),
            "streamed_count": streamed.get(ep, 0),
            "ignore_eos_count": ignore_eos.get(ep, 0),
            "reasoning_effort_counts": dict(reasoning_efforts.get(ep, Counter()))
            or None,
            "isl": _stat_block(isl_vals),
            "requested_osl": _stat_block(osl_vals),
            "min_tokens": _quantiles(min_tokens.get(ep, [])),
            "vocab_distribution": vd,
        }
    return {"total_requests": total, "per_endpoint": per_endpoint}


def _resolve_vocab_size(
    declared: int | None, source: str, counts: Counter[int]
) -> int:
    """Return vocab size for the per-endpoint distribution.

    For the `"tokenizer"` source we trust the declared value. For the
    `"observed"` source we use `max_observed_id + 1` (or the declared value,
    whichever is greater) so coverage_pct stays sane when the tokenizer
    doesn't expose len().
    """
    if not counts:
        return declared or 0
    observed_max = max(counts.keys())
    if source == "observed":
        return max(declared or 0, observed_max + 1)
    return declared or (observed_max + 1)
```

- [ ] **Step 4: Run, confirm passes**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestBuildSummaryVocab -v
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
```

Expected: 2 new tests pass; all prior tests still pass (existing `TestBuildSummary` cases that don't pass the new kwargs get `vocab_distribution: None` per the default-arg path).

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): thread vocab_distribution through _build_summary"
```

---

## Task 6: Wire `_render_vocab_lines` + new spacing into `_print_summary`

**Files:**
- Modify: `tests/aiperf_mock_server/request_recorder.py` (`_print_summary`, `RequestRecorder.close`)
- Modify: `tests/unit/aiperf_mock_server/test_request_recorder.py` (extend `TestPrintSummary`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/aiperf_mock_server/test_request_recorder.py`:

```python
class TestPrintSummaryVocab:
    def _vd(self, shape: list[int] | None = None) -> dict:
        if shape is None:
            shape = [10] + [0] * 79
        return {
            "vocab_size": 1000,
            "vocab_size_source": "tokenizer",
            "unique_ids": 5,
            "coverage_pct": 0.5,
            "total_tokens": sum(shape),
            "top_10_concentration_pct": 99.0,
            "entropy_bits": 1.2,
            "max_entropy_bits": 9.97,
            "top_tokens": [
                {"id": 1, "text": " the", "count": 6},
                {"id": 2, "text": " a", "count": 2},
            ],
            "shape_80": shape,
            "frequencies": {},
        }

    def _summary(self, vd: dict | None) -> dict:
        return {
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
                            "bin_edges": [10.0, 25.0, 40.0],
                            "counts": [2, 2],
                        },
                    },
                    "requested_osl": None,
                    "min_tokens": None,
                    "vocab_distribution": vd,
                },
            },
        }

    def test_vocab_block_prints_after_histograms(self, capsys) -> None:
        _print_summary(self._summary(self._vd()))
        out = capsys.readouterr().out
        idx_isl_hist = out.index("ISL histogram")
        idx_vocab_headline = out.index("Vocab  used")
        idx_shape = out.index("vocab shape")
        assert idx_isl_hist < idx_vocab_headline < idx_shape

    def test_no_vocab_lines_when_distribution_is_none(self, capsys) -> None:
        _print_summary(self._summary(None))
        out = capsys.readouterr().out
        assert "Vocab  used" not in out
        assert "vocab shape" not in out

    def test_blank_lines_between_blocks(self, capsys) -> None:
        _print_summary(self._summary(self._vd()))
        out = capsys.readouterr().out
        lines = out.splitlines()
        # Find the index of the endpoint header.
        ep_idx = next(i for i, ln in enumerate(lines) if "/v1/chat/completions" in ln)
        # Expected ordering after header:
        #   ISL stats, OSL stats(skipped→n/a), [blank], ISL histogram (header+rows),
        #   [blank], Vocab headline, top:, [blank], vocab shape header, sparkline,
        #   axis ticks.
        # Confirm blank-line separators by looking for "" entries at the
        # expected gap positions.
        post = lines[ep_idx + 1 :]
        assert "" in post  # at least one blank-line separator present
```

- [ ] **Step 2: Run, confirm fails**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py::TestPrintSummaryVocab -v
```

Expected: ordering assertions fail because the vocab block isn't printed.

- [ ] **Step 3: Modify `_print_summary`**

Find the existing `_print_summary` function. Insert blank-line separators between blocks and print the vocab block via `_render_vocab_lines` between the histogram pass and the misc-fields pass:

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
            hist = s["histogram"]
            n = sum(hist["counts"])
            print("")  # blank line before each histogram block
            for line in _render_histogram(label, hist, n, s["unique_values"]):
                print(line)
        vd = stats.get("vocab_distribution")
        if vd is not None:
            print("")  # blank line before vocab block
            for line in _render_vocab_lines(vd):
                print(line)
        mn = stats["min_tokens"]
        if mn is not None:
            print("")  # blank line before misc lines
            print(
                f"    min_tokens  mean {mn['mean']:7.1f}   p50 {mn['p50']:5.0f}"
            )
        if stats["ignore_eos_count"]:
            print(f"    ignore_eos=true: {stats['ignore_eos_count']}")
        if stats["reasoning_effort_counts"]:
            print(f"    reasoning_effort: {stats['reasoning_effort_counts']}")
```

Then replace the existing `RequestRecorder.close()` method body with the version below. The change is purely additive — a `decode_fn` lambda built from `self._tokenizer` and four new kwargs passed to `_build_summary`; the JSON-write and `_print_summary` calls are unchanged.

```python
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        decode_fn: Callable[[int], str] | None
        if self._tokenizer is not None:
            decode_fn = lambda token_id: self._tokenizer.decode([token_id])  # noqa: E731
        else:
            decode_fn = None
        summary = _build_summary(
            total=self._total,
            isls=self._isls,
            osls=self._osls,
            min_tokens=self._min_tokens,
            streamed=self._streamed,
            ignore_eos=self._ignore_eos,
            reasoning_efforts=self._reasoning_efforts,
            vocab_counts=self._vocab_counts,
            vocab_size=self._vocab_size,
            vocab_size_source=self._vocab_size_source,
            decode_fn=decode_fn,
        )
        Path(self.path + ".summary.json").write_bytes(
            orjson.dumps(summary, option=orjson.OPT_INDENT_2)
        )
        _print_summary(summary)
```

- [ ] **Step 4: Run, confirm passes**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server/test_request_recorder.py -v
uv run --no-sync pytest tests/integration/test_mock_server_record_requests.py -m integration -v
```

Expected: all unit tests pass (including any prior `TestPrintSummary` cases that were checking the existing block ordering — those should still pass since the vocab block is additive). Integration tests pass too.

If a prior `TestPrintSummary` test breaks because of the new blank lines, update its assertion to reflect the new layout — *don't* remove the blank lines.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/request_recorder.py \
        tests/unit/aiperf_mock_server/test_request_recorder.py
git commit -s -m "feat(mock-server): render vocab distribution + add stdout spacing"
```

---

## Task 7: Extend the integration test

**Files:**
- Modify: `tests/integration/test_mock_server_record_requests.py`

- [ ] **Step 1: Add the new assertions**

Find the existing chat-block assertions inside `test_records_per_request_isl_and_requested_osl` (the ones that check `chat_stats["isl"]["histogram"]`, etc.). After those assertions, append:

```python
        # Vocab distribution block for chat: present, with valid shape and ids.
        chat_vd = chat_stats["vocab_distribution"]
        assert chat_vd is not None
        assert chat_vd["vocab_size"] > 0
        assert chat_vd["unique_ids"] >= 1
        assert chat_vd["total_tokens"] >= chat_vd["unique_ids"]
        assert 0.0 <= chat_vd["coverage_pct"] <= 100.0
        assert len(chat_vd["shape_80"]) == 80
        assert sum(chat_vd["shape_80"]) == chat_vd["total_tokens"]
        assert 1 <= len(chat_vd["top_tokens"]) <= 10
        for entry in chat_vd["top_tokens"]:
            assert isinstance(entry["id"], int)
            assert isinstance(entry["text"], str)
            assert entry["count"] >= 1
        assert 0.0 <= chat_vd["entropy_bits"] <= chat_vd["max_entropy_bits"] + 1e-6
        assert chat_vd["vocab_size_source"] in {"tokenizer", "observed"}
        # Embeddings endpoint exists in the fixture; its vocab block should
        # also exist (ISL is recorded) — sanity-check that this isn't broken.
        emb_vd = emb_stats["vocab_distribution"]
        assert emb_vd is None or emb_vd["unique_ids"] >= 1
```

- [ ] **Step 2: Run**

```bash
cd /Users/fdinatale/Code/aiperf/.worktrees/mock-server-request-recorder
uv run --no-sync pytest tests/integration/test_mock_server_record_requests.py -m integration -v
```

Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_mock_server_record_requests.py
git commit -s -m "test(mock-server): cover vocab_distribution in recorder integration test"
```

---

## Task 8: Refresh README

**Files:**
- Modify: `tests/aiperf_mock_server/README.md`

- [ ] **Step 1: Locate the Output format JSON example and the Summary stdout sample**

```bash
cd /Users/fdinatale/Code/aiperf/.worktrees/mock-server-request-recorder
grep -n "Output format\|## Summary\|requested_osl" tests/aiperf_mock_server/README.md
```

Note line ranges for the per-metric JSON example and the stdout sample.

- [ ] **Step 2: Extend the JSON example**

In the README, find the block that shows a sample `"isl": {...}` per-metric stats dict. Immediately after that block (still inside the example showing what's in `<path>.summary.json`), add a new example showing the new `vocab_distribution` field at the per-endpoint level:

```json
"vocab_distribution": {
  "vocab_size": 151936,
  "vocab_size_source": "tokenizer",
  "unique_ids": 5234,
  "coverage_pct": 3.44,
  "total_tokens": 102000,
  "top_10_concentration_pct": 47.21,
  "entropy_bits": 8.23,
  "max_entropy_bits": 17.21,
  "top_tokens": [
    {"id": 264, "text": " the", "count": 3201},
    {"id": 318, "text": " a",   "count": 2890}
  ],
  "shape_80": [3201, 412, 311, 0, 47, "..."],
  "frequencies": {"264": 3201, "318": 2890, "...": 0}
}
```

Add a short prose paragraph above or below:

> The optional `vocab_distribution` block (per endpoint) characterises sampling across the tokenizer's vocabulary: coverage of distinct ids, top-N concentration, Shannon entropy with the uniform-sampling ceiling for comparison, an 80-bucket sparkline across the full id space, and the full `token_id → count` frequency table for offline analysis. The block is `null` when no requests reached the endpoint.

- [ ] **Step 3: Refresh the stdout sample**

Find the `## Summary` section's stdout sample block. Replace its contents with the new layout (note the blank lines between blocks):

```
Request distribution (100 requests)
──────────────────────────────────────────────
  /v1/chat/completions  n=100
    ISL    mean  1010.5   p50   998   p99  1819
    OSL    mean   127.5   p50   129   p99   229

    ISL histogram (17 bins, n=100, 19 unique)
       207-  302   7 ████░░░░░░░░░░░░░░░░
       ... (17 rows total)
      1726- 1821   1 ░░░░░░░░░░░░░░░░░░░░

    OSL histogram (10 bins, n=100, 11 unique)
        25-   46   3 ██░░░░░░░░░░░░░░░░░░
       ... (10 rows total)
       210-  230   6 ████░░░░░░░░░░░░░░░░

    Vocab  used 5234/151936 (3.4%)  top-10 cover 47%  entropy 8.2/17.2 bits
      top: " the" 3201, " a" 2890, " of" 2455, " to" 2103, " and" 1987

    vocab shape  (80 buckets over id 0..151935, log-y)
    ▇▇▇▅▅▄▄▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▁ ▁ ▁  ▁  ▁  ▁         ▁       ▁    ▁       ▁
    0                            38K                          76K                       114K                152K

    min_tokens  mean    32.0   p50    32
```

- [ ] **Step 4: Run pre-commit on the README**

```bash
pre-commit run --files tests/aiperf_mock_server/README.md
```

Expected: all hooks pass (trailing-whitespace, codespell, end-of-file-fixer, etc.). Re-add and amend if any auto-fixes are applied.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/README.md
git commit -s -m "docs(mock-server): document recorder vocab_distribution block"
```

---

## Task 9: Pre-merge verification

- [ ] **Step 1: Lint + format**

```bash
cd /Users/fdinatale/Code/aiperf/.worktrees/mock-server-request-recorder
ruff format tests/aiperf_mock_server/request_recorder.py \
            tests/aiperf_mock_server/README.md \
            tests/unit/aiperf_mock_server/test_request_recorder.py \
            tests/integration/test_mock_server_record_requests.py
ruff check --fix tests/aiperf_mock_server/request_recorder.py \
                 tests/unit/aiperf_mock_server/test_request_recorder.py \
                 tests/integration/test_mock_server_record_requests.py
```

Expected: `All checks passed!` and N files unchanged / reformatted.

- [ ] **Step 2: Full unit test sweep**

```bash
uv run --no-sync pytest tests/unit/aiperf_mock_server -v
```

Expected: all tests pass (existing + new). Verify the new totals: 4 (Task 1) + 7 (Task 2) + 11 (Task 3) + 10 (Task 4) + 2 (Task 5) + 3 (Task 6) = 37 new tests added; prior baseline was 18; total ~55.

- [ ] **Step 3: Integration sweep**

```bash
uv run --no-sync pytest tests/integration/test_mock_server_record_requests.py -m integration -v
```

Expected: 3 tests pass.

- [ ] **Step 4: Pre-commit across all touched files**

```bash
pre-commit run --files \
  tests/aiperf_mock_server/request_recorder.py \
  tests/aiperf_mock_server/README.md \
  tests/unit/aiperf_mock_server/test_request_recorder.py \
  tests/integration/test_mock_server_record_requests.py
```

Expected: all hooks pass. If pre-commit modifies any feature file, re-add and amend the last commit.

- [ ] **Step 5: Verify branch state**

```bash
git log --oneline origin/main..HEAD
```

Expected: at least 8 new commits on top of the prior tip (`b5392674` after rebase):
1. `docs(spec): design for recorder vocabulary-distribution stats`
2. `docs(plan): implementation plan for recorder vocabulary-distribution stats` *(see note below)*
3. `feat(mock-server): capture per-endpoint token-id Counter in recorder`
4. `feat(mock-server): add _compute_shape_80 helper for vocab sparkline`
5. `feat(mock-server): add _vocab_distribution helper for recorder summary`
6. `feat(mock-server): add _render_vocab_lines for stdout vocab block`
7. `feat(mock-server): thread vocab_distribution through _build_summary`
8. `feat(mock-server): render vocab distribution + add stdout spacing`
9. `test(mock-server): cover vocab_distribution in recorder integration test`
10. `docs(mock-server): document recorder vocab_distribution block`

The plan-document commit (#2 above) lands as part of this plan's controller workflow; the spec commit (#1) already exists on the branch.

- [ ] **Step 6: Manual end-to-end smoke test (optional but recommended before pushing)**

```bash
# Terminal 1
aiperf-mock-server --record-requests /tmp/vocab.jsonl --fast --tokenizer Qwen/Qwen3-0.6B

# Terminal 2
aiperf profile \
    --endpoint-type chat \
    --url http://localhost:8000 \
    --model Qwen/Qwen3-0.6B \
    --random-range-ratio 0.2 --isl-mean 1024 --osl-mean 256 \
    --request-count 200

# Ctrl-C the mock server.
# Inspect:
cat /tmp/vocab.jsonl.summary.json | python -m json.tool | head -120
# Confirm the vocab_distribution block exists per endpoint with non-trivial shape_80.
# Confirm the stdout summary printed in terminal 1 shows the new layout with the vocab block.
```
