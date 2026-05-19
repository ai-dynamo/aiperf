# Recorder Vocabulary-Distribution Stats

**Date:** 2026-05-18
**Branch:** `fdinatale/mock-server-request-recorder`
**Status:** Design

## Motivation

The current request recorder reports per-endpoint statistics on prompt length (`isl`), requested output length (`requested_osl`), and `min_tokens`. None of those signals reveal *which* tokens the dataset is using or how concentrated the sampling is across the tokenizer's vocabulary.

For datasets meant to exercise broad vocabulary coverage — e.g. randomly sampled token IDs, randomly-permuted prompt fragments, or natural text with diverse subject matter — you want to be able to look at a recording and tell whether the sampler is in fact spreading across the vocabulary or collapsing onto a small high-frequency head. Two recordings can have identical ISL/OSL distributions and *very* different vocabulary footprints.

This spec adds a `vocab_distribution` block per endpoint, in both `<path>.summary.json` and the stdout summary, that surfaces:
- Coverage (how many distinct token IDs appeared out of the full vocab).
- Concentration (how much of the total token volume is carried by the top few IDs).
- Shape (an 80-bucket histogram across the full vocabulary, log-y scaled).
- Top-N (the most frequent tokens with their decoded text).
- Entropy (Shannon entropy of the observed distribution, with the uniform-sampling ceiling for comparison).

## Scope

In scope:
- Track per-endpoint `Counter[int]` of token-id → frequency across all requests.
- Compute headline stats and write them to the existing per-endpoint summary block.
- Render a stacked block in the stdout summary (headline + top-N, blank line, then the 80-bucket shape with axis ticks).
- Render extra blank lines between *every* per-endpoint block (ISL stats, ISL histogram, OSL histogram, vocab block) — a small spacing refresh the user asked for alongside this feature.
- Tests at unit and integration levels.

Out of scope (YAGNI):
- Per-request token-ID arrays in the JSONL records (one 1K-token prompt × 100k requests ≈ 800 MB of integers).
- Top-level / cross-endpoint aggregate vocab block (per-endpoint matches the existing summary structure; we can add a global block later if a use case appears).
- Tokenizer-aware filtering (special tokens, byte-fallback, added tokens) — every observed ID counts equally for now.
- A CLI flag to toggle vocab stats — always on when `--record-requests` is set. Cost is minimal because tokenization already runs for ISL.

## Bucketing rule (sparkline shape)

`shape_80[i]` is the sum of all token-id counts whose IDs fall in the half-open range `[i * vocab_size / 80, (i + 1) * vocab_size / 80)`, except the last bucket which is closed on both ends so the highest valid ID lands in it. With Qwen3-0.6B (`vocab_size = 151936`) this is ≈ 1899 IDs per bucket.

Rendering, per bucket:
- Convert each count to a height using `log1p(count) / log1p(max_count)` and quantize to one of `▁▂▃▄▅▆▇█`.
- If a bucket count is **exactly zero**, render a literal space (`" "`) instead of `▁` so unused vocab regions are visually obvious.
- The full sparkline is 80 characters wide.
- Below the sparkline, an axis-tick line shows the IDs at `0`, `vocab_size/4`, `vocab_size/2`, `3·vocab_size/4`, `vocab_size`, right-aligned so each tick sits under its column. Numbers are rendered with a `K` suffix for ≥ 1000 (e.g. `38K`, `152K`).

### Edge cases

| Input | Behavior |
|---|---|
| No requests recorded for an endpoint | `vocab_distribution` is `null` (matches the existing `requested_osl: None` convention). |
| All observations are the same token id | `entropy_bits = 0.0`, `top_tokens` has 1 entry, `shape_80` has one non-zero bucket, sparkline shows a single tall bar. |
| `len(self._tokenizer)` not available | Fall back to `max(observed_id) + 1`; the JSON block notes `"vocab_size_source": "observed"` (vs `"tokenizer"` for the normal path). |
| `tokenizer.decode([id])` raises for some id | Fall back to `"<id=N>"` for that entry in `top_tokens`. Don't fail the entire block. |
| Token id ≥ `vocab_size` (shouldn't happen but defensive) | Drop it (and log once at WARNING). |

## JSON shape

Per endpoint, alongside the existing `isl` / `requested_osl` / `min_tokens` blocks:

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
    {"id": 318, "text": " a",   "count": 2890},
    {"id": 290, "text": " of",  "count": 2455},
    {"id": 311, "text": " to",  "count": 2103},
    {"id": 323, "text": " and", "count": 1987},
    {"id": 304, "text": " in",  "count": 1654},
    {"id": 374, "text": " is",  "count": 1532},
    {"id": 437, "text": " for", "count": 1421},
    {"id": 393, "text": " on",  "count": 1208},
    {"id": 449, "text": " with","count": 1109}
  ],
  "shape_80": [3201, 412, 311, 0, 47, ..., 0],
  "frequencies": {"264": 3201, "318": 2890, ...}
}
```

Field notes:
- `vocab_size_source` is `"tokenizer"` (preferred) or `"observed"` (fallback).
- `top_tokens` length is exactly 10 (matches the headline "top-10 cover N%" stat). If fewer than 10 unique IDs were observed, length equals `unique_ids`.
- `top_tokens[*].text` is the raw `tokenizer.decode([id])` output. Leading spaces are preserved (BPE convention). If decode raises or returns non-printable text, falls back to `"<id=N>"` and the original code-point fidelity is preserved.
- `shape_80` is always exactly 80 ints. Empty buckets are `0`.
- `frequencies` is the full `str(token_id) → count` table. JSON dict keys must be strings, so we stringify the IDs. Worst case size for a fully-saturated 152K vocab is ≈ 3 MB; typical runs are well under that.
- `entropy_bits` uses `math.log2(p) * p` summed over observed IDs; `max_entropy_bits = math.log2(vocab_size)`.

## Stdout layout

The full per-endpoint block, after this change. New blank lines highlighted; the vocab block sits after the OSL histogram and before the `min_tokens` / `ignore_eos` / `reasoning_effort` lines:

```
  /v1/chat/completions  n=100
    ISL    mean  1010.5   p50   998   p99  1819
    OSL    mean   127.5   p50   129   p99   229

    ISL histogram (17 bins, n=100, 19 unique)
       207-  302   7 ████░░░░░░░░░░░░░░░░
       ... (rows omitted)
      1726- 1821   1 ░░░░░░░░░░░░░░░░░░░░

    OSL histogram (10 bins, n=100, 11 unique)
        25-   46   3 ██░░░░░░░░░░░░░░░░░░
       ... (rows omitted)
       210-  230   6 ████░░░░░░░░░░░░░░░░

    Vocab  used 5234/151936 (3.4%)  top-10 cover 47%  entropy 8.2/17.2 bits
      top: " the" 3201, " a" 2890, " of" 2455, " to" 2103, " and" 1987

    vocab shape  (80 buckets over id 0..151935, log-y)
    ▇▇▇▅▅▄▄▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▁ ▁ ▁  ▁  ▁  ▁         ▁       ▁    ▁       ▁
    0                            38K                          76K                       114K                152K

    min_tokens  mean    32.0   p50    32
    ignore_eos=true: 41
```

Rules:
- The blank lines between every block are added unconditionally as part of this change.
- The vocab block is rendered only when `vocab_distribution` is non-null. When it's null (no requests on the endpoint), neither the `Vocab` line nor the `vocab shape` block prints.
- `top:` line shows the first 5 entries of `top_tokens` formatted as `"<text>" <count>` joined with `, `. The text is wrapped in literal double quotes with any leading/trailing whitespace preserved (e.g. `" the" 3201`). If `text` contains a non-printable character or `tokenizer.decode([id])` raised, render that entry as `<id=N> <count>` instead (no quotes).
- `top-10 cover N%` percentage is `sum(top_10_counts) / total_tokens * 100`, rounded to integer.
- Axis tick labels: integers with `K` suffix at ≥1000 (no `M` suffix needed — typical vocabs are well under a million). Each tick is right-aligned to its column index in the 80-char sparkline.

## Implementation site

All in `tests/aiperf_mock_server/request_recorder.py`:

- `RequestRecorder.open()` adds:
  ```python
  try:
      self._vocab_size = len(self._tokenizer)
      self._vocab_size_source = "tokenizer"
  except (TypeError, AttributeError):
      self._vocab_size = None  # set lazily from max observed id
      self._vocab_size_source = "observed"
  ```
- `RequestRecorder.__init__` adds `self._vocab_counts: dict[str, Counter[int]] = defaultdict(Counter)`.
- `RequestRecorder.record()` keeps the token-id list:
  ```python
  ids = self._tokenizer.encode(text)
  isl = len(ids)
  self._vocab_counts[endpoint].update(ids)
  ```
- New helpers (placed near the existing `_histogram` / `_render_histogram`):
  - `_vocab_distribution(counts: Counter[int], vocab_size: int, source: str, tokenizer) -> dict | None` — builds the JSON block.
  - `_compute_shape_80(counts: Counter[int], vocab_size: int) -> list[int]` — bucket counts.
  - `_render_vocab_lines(vd: dict) -> list[str]` — returns the four-line stdout block (headline, top, blank, shape, ticks).
- `_build_summary` wires `vocab_distribution = _vocab_distribution(...)` into each per-endpoint dict.
- `_print_summary` inserts the new blank lines between blocks and calls `_render_vocab_lines` when `vd is not None`.

## Memory and runtime

- One `Counter[int]` per endpoint, bounded by vocab size. For Qwen3-0.6B that's 152K × ~50 bytes/entry ≈ 8 MB worst case (rarely reached). For typical runs (a few thousand observed IDs), well under 1 MB.
- `Counter.update(ids)` is O(len(ids)). Tokenization is the dominant cost and already runs.
- `_compute_shape_80` is O(unique_ids) at shutdown — single pass.
- Entropy: O(unique_ids) sum at shutdown.

## Tests

In `tests/unit/aiperf_mock_server/test_request_recorder.py`:

- `TestVocabDistribution`:
  - `test_returns_none_for_empty_counter`
  - `test_coverage_pct_and_unique_ids` — feed a known Counter, assert exact stats.
  - `test_top_tokens_length_caps_at_10`
  - `test_top_tokens_falls_back_to_id_marker_when_decode_raises` (mock a tokenizer whose `decode([id])` raises for a specific id).
  - `test_entropy_zero_for_single_token` (`H = 0`, ratio = 0).
  - `test_entropy_at_max_for_uniform_sampling` (synthesize a perfectly-uniform Counter, assert `entropy_bits == log2(vocab_size)` within float tolerance).
  - `test_shape_80_length_is_always_80`
  - `test_shape_80_buckets_partition_full_vocab` (assert sum of bucket counts equals total observations; assert IDs at bucket boundaries land in the correct bucket).
  - `test_vocab_size_source_observed_when_len_unavailable` (mock a tokenizer where `len(...)` raises).

- `TestRenderVocabShape`:
  - `test_zero_bucket_renders_as_space`
  - `test_log_y_makes_smaller_bars_visible` (give one giant bucket and several small ones; assert the small buckets render as non-`▁` block chars, not invisible).
  - `test_axis_tick_alignment` — assert tick labels start at the column positions corresponding to 0%, 25%, 50%, 75%, 100% of the 80-char sparkline.

- `TestPrintSummary`:
  - `test_vocab_block_prints_between_osl_hist_and_min_tokens` (capsys, verify exact slot in the per-endpoint block).
  - `test_blank_lines_between_blocks` — verify the new spacing rules (one blank line between ISL stats / ISL hist / OSL hist / vocab block).
  - `test_no_vocab_block_when_distribution_is_null`

Integration test (`tests/integration/test_mock_server_record_requests.py`):
- Extend `test_records_per_request_isl_and_requested_osl` to assert the chat endpoint summary contains a `vocab_distribution` block with the expected keys, `unique_ids >= 1`, `len(shape_80) == 80`, `top_tokens` non-empty, and `entropy_bits >= 0`.

## Pre-merge

- `ruff format . && ruff check --fix .` on touched files.
- `uv run pytest tests/unit/aiperf_mock_server -v`
- `uv run pytest tests/integration/test_mock_server_record_requests.py -m integration -v`
- `pre-commit run --files <changed files>`
- README refresh in `tests/aiperf_mock_server/README.md`: Output format JSON example gains a `vocab_distribution` snippet; Summary stdout sample shows the new layout with vocab block and blank lines.

## Open design notes (informational, no action required)

- **JSON size**: `frequencies` could push the summary JSON into the multi-MB range for runs that touch a large fraction of the vocab. If a user objects, we can add `--record-requests-skip-vocab-table` later; for now the table is always included because skipping it is YAGNI.
- **Entropy framing**: `entropy_bits` is info-theory jargon. The JSON exposes both `entropy_bits` and `max_entropy_bits` so a downstream tool can compute any ratio. Stdout shows `entropy A/B bits` as a compromise: bits for fluency, the ratio readable by inspection.
- **Top-N count**: stdout shows the first 5 tokens to keep the line readable; JSON keeps 10 to give downstream analyses a slightly richer head. If 5 vs 10 ends up feeling arbitrary, both can be made constants and revisited.
