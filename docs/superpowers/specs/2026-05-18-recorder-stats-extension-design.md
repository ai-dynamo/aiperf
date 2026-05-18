# Recorder Stats Extension: Histograms + Unique Counts

**Date:** 2026-05-18
**Branch:** `fdinatale/mock-server-request-recorder`
**Status:** Design

## Motivation

The mock-server request recorder emits per-endpoint distribution stats for `isl`, `requested_osl`, and `min_tokens` today — min, max, mean, stdev, p50/p90/p95/p99. Those numbers tell you the spread but not the shape: a uniform distribution and a bimodal distribution can produce nearly identical percentiles. When validating that `--random-range-ratio` / `--strict-isl` / `--isl-mean` actually produce the intended distribution on the wire, shape matters more than tail percentiles.

Add a histogram and a unique-value count to the `isl` and `requested_osl` stats blocks, in both `<path>.summary.json` and the stdout summary, so the actual distribution shape is visible at a glance.

## Scope

In scope:
- Extend the per-endpoint stats blocks for `isl` and `requested_osl` with `histogram` + `unique_values` fields.
- Render the histogram in both `<path>.summary.json` and stdout.
- Test the new fields in the existing recorder integration test.

Out of scope (YAGNI):
- Histogramming `min_tokens` (rarely set; existing percentiles are sufficient).
- Per-request JSONL changes — the histogram is computed at summary time from in-memory lists.
- CLI flags for bin parameters — fixed defaults for now; can add if it becomes painful.
- Histograms across categorical breakdowns (per-model, per-stream). Defer until the basic shape is in place.

## Bucketing

```
num_bins = max(10, ceil((max - min) / 100))
bin_width = (max - min) / num_bins   # equal-width
```

Each bin `[edges[i], edges[i+1])` is right-open, except the last bin which is closed on both ends so the observed maximum lands in the last bin (not just past it). With observed ISL range 207–1821, this yields 17 bins of width ~95 tokens. With observed OSL range 25–230, this yields 10 bins (floor wins). With a tight `max_tokens ∈ {16,32,64,128,256}` distribution (range 240), it also yields 10 bins.

The `max_bin_width=100` cap forces meaningful resolution on wide ranges; the `min_bins=10` floor keeps narrow ranges from collapsing to a few uninformative bins. Both are hardcoded module-level constants in `request_recorder.py` (`_HISTOGRAM_MIN_BINS = 10`, `_HISTOGRAM_MAX_BIN_WIDTH = 100`).

### Edge cases

| Input | Behavior |
|---|---|
| No observations (e.g. OSL on `/v1/embeddings`) | `histogram: null`, `unique_values: 0`. Mirrors the existing `requested_osl: null` convention. |
| All values equal (`min == max`) | One bin `[min, min]` with the full count. `bin_edges` has length 2. `num_bins = 1` (overrides the floor). |
| Single observation | Same as `min == max`. |

## Output shape

### JSON

The histogram and unique count nest inside the existing per-metric stats block:

```json
"isl": {
  "min": 207.0, "max": 1821.0, "mean": 1010.48, "stdev": 480.80,
  "p50": 997.5, "p90": 1684.8, "p95": 1745.55, "p99": 1819.02,
  "unique_values": 19,
  "histogram": {
    "bin_edges": [207.0, 301.94, 396.88, ..., 1821.0],
    "counts":    [13, 8, 7, 9, 19, 19, 10, 6, 4, 5, ...]
  }
}
```

`bin_edges` length == `counts` length + 1. Parallel-array shape matches `numpy.histogram` / `pandas.cut`, so analysis code can drop it straight into a DataFrame.

`requested_osl` block gets the same two new fields. `min_tokens` block does not — left alone for now.

### Stdout

The horizontal-bar layout, rendered after the existing percentile lines for ISL and OSL:

```
  /v1/completions  n=100
    ISL    mean  1010.5   p50   998   p99  1819
    OSL    mean   127.5   p50   129   p99   229
    ISL histogram (17 bins, n=100, 19 unique)
       207- 302   7 ████░░░░░░░░░░░░░░░░
       302- 397   6 ███░░░░░░░░░░░░░░░░░
       ... (17 rows total)
      1726-1821   3 █░░░░░░░░░░░░░░░░░░░
    OSL histogram (10 bins, n=100, 11 unique)
       25-  46   3 ██░░░░░░░░░░░░░░░░░░
       ... (10 rows total)
```

Layout rules:
- Bars: 20 chars wide, filled with `█`, padded with `░`, scaled so the tallest bin in *that* histogram is full-width.
- Bin range labels: low-high formatted as integers (rounded from float edges), left-padded so the numbers right-align and columns line up within a metric.
- Count column: right-aligned, width auto-sized to fit the largest bin count (minimum width 3 chars; grows as needed for runs >999/bin).
- When `requested_osl` is `null` (embeddings), skip the OSL histogram block entirely — same convention as the existing "OSL n/a" line.

## Implementation

All new code in `tests/aiperf_mock_server/request_recorder.py`:

```python
_HISTOGRAM_MIN_BINS = 10
_HISTOGRAM_MAX_BIN_WIDTH = 100.0

def _histogram(values: list[int]) -> dict[str, list[float]] | None:
    """Equal-width histogram with max_bin_width/min_bins rule. None if empty."""

def _render_histogram(metric: str, hist: dict, unique: int) -> list[str]:
    """Render the bar-chart block as a list of stdout lines (indented 4 spaces)."""
```

Wiring:
- `_build_summary` adds `unique_values = len(set(values))` and `histogram = _histogram(values)` to each `isl` / `requested_osl` block.
- `_print_summary` calls `_render_histogram` after the existing percentile line for ISL and OSL only. Skip when `histogram is None`.

`_quantiles` is unchanged.

## Testing

Extend `tests/integration/test_mock_server_record_requests.py::TestRecordRequests::test_records_per_request_isl_and_requested_osl`:

- Assert `chat_stats["isl"]["unique_values"] >= 1` and `chat_stats["isl"]["histogram"]` is a dict.
- Assert `len(chat_stats["isl"]["histogram"]["bin_edges"]) == len(chat_stats["isl"]["histogram"]["counts"]) + 1`.
- Assert `sum(chat_stats["isl"]["histogram"]["counts"]) == chat_stats["count"]`.
- Assert `chat_stats["requested_osl"]["unique_values"] == 6` (the test sends max_tokens ∈ {16, 32, 64, 128, 256} on five requests plus max_completion_tokens=192 on one — six distinct resolved OSL values).
- Assert `emb_stats["requested_osl"]["histogram"] is None` and `emb_stats["requested_osl"]["unique_values"] == 0`.
- Embedded ISL histogram for embeddings should still be present (`emb_stats["isl"]["histogram"]` is a dict, since ISL is always recorded).

No new tests needed for the rendering helper specifically — the integration test exercises the full path including stdout (although it doesn't currently capture stdout). If we want a dedicated unit test for `_render_histogram`, add one that feeds a known dict and asserts the lines come out with the expected widths; defer unless the rendering proves fragile.

## Pre-merge

- `ruff format . && ruff check --fix .`
- `uv run pytest tests/integration/test_mock_server_record_requests.py -m integration`
- `pre-commit run --all-files`
- Update the README in `tests/aiperf_mock_server/` — the existing "Output format" / "Summary" sections need the new fields and a refreshed sample showing the histogram.
