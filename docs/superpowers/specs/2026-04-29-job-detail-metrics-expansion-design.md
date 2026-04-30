# Job Detail Page — Metrics Table Expansion

**Status:** Approved
**Date:** 2026-04-29
**Scope:** `src/aiperf/operator/ui-v1/pages/job-detail.js` only.

## Problem

The Full Metrics Breakdown table on the job-detail page surfaces ~12 metric tags. The metrics registry produces 60+. Many useful, commonly-emitted metrics are buried in the raw `profile_export_aiperf.json` and not visible from the UI: per-user throughputs, goodput, request counts, the HTTP-trace family, usage-token counters, reasoning-model efficiency, vision metrics, OSL-mismatch quality, and streaming setup latencies. Custom or future-added metrics are also invisible.

A second issue: the existing HTTP rows use the keys `request_duration`, `connection_overhead`, `dns_lookup`. The actual registry tags are `http_req_duration`, `http_req_connection_overhead`, `http_req_dns_lookup`. The current rows are silently filtered out by `results[row.key] != null`.

## Approach

Curated groups (expanded with the missing metrics) plus an auto-discovery "Other Metrics" tail group. Curated rows keep their pretty labels and column whitelists; the tail catches everything else so plugin/extension metrics and future registry additions show up automatically without a UI change.

## Column set

Header order (10 numeric columns):

```
avg, std, p1, p10, p50, p90, p95, p99, min, max
```

Per-row `cols` array still gates which cells render data vs `---`. Tooltips on each new header:

- `std` — Standard deviation across observations
- `p1` — 1st percentile (best case)
- `p10` — 10th percentile

The existing horizontal-scroll wrapper (`overflow-x: auto`) handles the wider table on narrow viewports.

## Curated groups

Group order: Throughput → Latency → Tokens → Sequence Lengths → Counts & Totals → HTTP → Reasoning → Vision → Other Metrics.

### Throughput (palette.blue)

| key | cols |
|---|---|
| `request_throughput` | avg |
| `output_token_throughput` | avg |
| `total_token_throughput` | avg |
| `goodput` | avg |
| `output_token_throughput_per_user` | avg, p50, p99 |
| `e2e_output_token_throughput` | avg |
| `prefill_throughput_per_user` | avg, p50, p99 |

### Latency (palette.peach)

| key | cols |
|---|---|
| `request_latency` | avg, std, p1, p10, p50, p90, p95, p99, min, max |
| `time_to_first_token` | avg, std, p1, p10, p50, p90, p95, p99, min, max |
| `inter_token_latency` | avg, std, p1, p10, p50, p90, p95, p99, min, max |
| `time_to_second_token` | avg, p50, p95, p99 |
| `inter_chunk_latency` | avg, p50, p90, p99 |
| `time_to_first_output_token` | avg, p50, p90, p99 |
| `stream_setup_latency` | avg, p50, p99 |
| `stream_prefill_latency` | avg, p50, p99 |
| `image_latency` | avg, p50, p99 |

### Tokens (palette.mauve, NEW)

Usage-token counts per request (model-reported). All rows: `avg, p50, p99`.

- `usage_prompt_tokens`
- `usage_completion_tokens`
- `usage_total_tokens`
- `reasoning_token_count`
- `output_token_count`

### Sequence Lengths (palette.teal)

| key | cols |
|---|---|
| `input_sequence_length` | avg, p50, p99 |
| `output_sequence_length` | avg, p50, p99 |
| `requested_osl` | avg, p50 |
| `osl_mismatch_diff_pct` | avg |
| `error_isl` | avg |

### Counts & Totals (palette.sapphire, NEW)

Run-aggregate counters and totals. All rows: `avg` only.

- `request_count`, `good_request_count`, `error_request_count`
- `total_output_tokens`, `total_isl`, `total_osl`, `total_error_isl`
- `total_usage_prompt_tokens`, `total_usage_completion_tokens`, `total_usage_total_tokens`, `total_reasoning_tokens`
- `benchmark_duration`

### HTTP (palette.mauve)

**Fix existing keys:** `request_duration` → `http_req_duration`, `connection_overhead` → `http_req_connection_overhead`, `dns_lookup` → `http_req_dns_lookup`.

Timing rows (`avg, p50, p99`):
- `http_req_duration`, `http_req_total`, `http_req_waiting`, `http_req_connecting`, `http_req_sending`, `http_req_receiving`, `http_req_blocked`, `http_req_dns_lookup`, `http_req_connection_overhead`

Data/chunk rows (`avg, min, max`):
- `http_req_data_sent`, `http_req_data_received`, `http_req_chunks_sent`, `http_req_chunks_received`, `http_req_connection_reused`

### Reasoning (palette.lavender, NEW)

Hidden when results contain none of these tags.

| key | cols |
|---|---|
| `thinking_efficiency` | avg, p50, p99 |
| `overall_thinking_efficiency` | avg |

### Vision (palette.green, NEW)

Hidden when results contain none of these tags.

| key | cols |
|---|---|
| `num_images` | avg |
| `image_throughput` | avg |
| `video_inference_time` | avg, p50, p99 |
| `video_peak_memory` | avg, max |

### Other Metrics (palette.overlay1, NEW — auto-discovery tail)

Catches every key in `results` not claimed by a curated row. Behavior:

1. Build `curatedKeys: Set<string>` from every row's `key` across all curated groups.
2. For each `[k, v]` in `Object.entries(results)`:
   - Skip if `curatedKeys.has(k)`.
   - Skip if `v` is not an object (filters scalars like `error_rate`).
   - Skip if `v` has none of `avg, p50, sum, count` (filters non-metric structs).
3. Render rows alphabetized by key.
4. `cols` for each tail row: every column where the value is non-null in the data — i.e. show whatever the metric provided.
5. `label`: prettify the tag (replace `_` with space, title-case).
6. Group is hidden when no rows survive the filter.

## Auto-empty-hide

Each curated group already filters out rows whose key is missing from `results` and renders nothing if zero rows survive (`if (visibleRows.length === 0) return null`). Reasoning and Vision rely on this — they render nothing on text-only or non-reasoning runs.

## Non-goals

- No KPI-row changes.
- No new charts.
- No API changes (operator response unchanged).
- No changes to non-job-detail pages, despite some sharing the same `MetricsTable` patterns.
- No changes to baselines (`tools/ergonomics_baseline.json`, `tools/ruff_baseline.json`).

## Test plan

Manual / visual:

1. Load a completed job that has the full metrics complement. Confirm every curated group renders with the new rows; confirm `Other Metrics` tail renders with anything left over.
2. Load a text-only run (no vision/reasoning). Confirm Vision and Reasoning groups are hidden.
3. Load a streaming run. Confirm streaming-specific latency rows populate.
4. Confirm HTTP group now shows data (previously broken because of key mismatch).
5. Confirm column order matches spec; std/p1/p10 columns render values where present, `---` otherwise.
6. Confirm horizontal scroll engages on narrower viewports.
7. Confirm collapse/expand still works per group.

Automated:

- The page has no current unit tests for this component. No new tests added (out of scope; would require fixturing the full `results` payload).

## Files changed

- `src/aiperf/operator/ui-v1/pages/job-detail.js` — `METRIC_GROUPS` constant and `MetricsTable` component only.
