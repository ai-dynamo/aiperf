# Job-Detail Server-Metrics Section — Design

**Date:** 2026-04-29
**Scope:** `src/aiperf/operator/ui-v1/pages/job-detail.js` server-metrics rendering only.
**Status:** Approved 2026-04-29 (verbal in brainstorming; user said "do it full plan and implementation. no more questions").

## Problem

The current `ServerMetricsSection` (`pages/job-detail.js:1148-1285`) is a 140-line hardcoded vLLM-only renderer:

- Five KPI cards: `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions`, prefix-cache hit-rate.
- Three histogram charts: `vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`.
- Reads only `series[0].stats.avg`.

This silently drops:
- **Backend coverage** — Dynamo Frontend (`dynamo_frontend_*`), SGLang (`sglang:*`), TensorRT-LLM (`trtllm:*`), Dynamo Component KV stats (`dynamo_component_kvstats_*`).
- **Stat field richness** — every counter `rate`, every gauge `max`, every histogram `p50_estimate` / `p99_estimate` / `sum_rate` is unread.
- **Multi-endpoint data** — replicas (esp. disaggregated prefill/decode roles) past `series[0]` are invisible. Per-endpoint role detail is the highest-value gap because prefill and decode behave differently and aggregating them is misleading.
- **Time evolution** — every series carries `timeslices[]` arrays the UI never plots.

## Goals

1. Auto-detect which backend(s) emitted metrics and render only the available capabilities.
2. Show per-endpoint detail by default (not aggregate-only) so disaggregated deployments are legible.
3. Surface throughput and tail-latency KPIs (not just averages).
4. Render `timeslices[]` as line charts when present.
5. Show static deployment context (`*_info` metrics) as a single combined card.
6. Keep the section silently hidden when no data exists, and explicit-empty-state when the export file existed but carried no observations (current behavior preserved).

## Non-Goals

- No backend metric collection changes (purely a UI refactor).
- No new JSON export schema fields.
- No multi-job comparison (lives elsewhere).
- No JS unit-test scaffolding (ui-v1 has no JS test framework today; e2e tests target the legacy `operator/ui/`).

## File Layout

```
src/aiperf/operator/ui-v1/components/server-metrics/
├── index.js               # ServerMetricsSection — orchestrator, the only export consumed by job-detail.js
├── helpers.js             # pure data helpers (no preact, no DOM)
├── deployment-card.js     # static *_info / *_model_* metrics
├── capacity-section.js    # KV cache, queue depth, preemptions, requests running/waiting
├── throughput-section.js  # req/s, prompt-tok/s, gen-tok/s, output-tok/s
├── latency-section.js     # p50/p99 KPIs + histogram small-multiples
└── timeline-section.js    # line charts from `timeslices[]`
```

`pages/job-detail.js` loses ~200 lines (`getSeriesValue`, `buildHistogramChartData`, `ServerMetricsSection`) and gains a single import.

## Visual Layout (top-down inside a "Server Metrics" wrapper)

```
┌─ Server Metrics (header) ─────────────────────────────────┐
│                                                           │
│ ▸ Deployment            (single combined card)          │
│ ▸ Capacity              (KV%, queue, preempts, R/W)      │
│ ▸ Throughput            (req/s, prompt+gen+output tok/s) │
│ ▸ Latency               (p50/p99 KPIs + histograms)      │
│ ▸ Timeline              (line charts from timeslices)    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

Each sub-section is a card. Sub-sections render `null` when no relevant metrics are present. Wrapper renders an explicit empty-state if all sub-sections returned null but the JSON existed.

## Data Model

`server_metrics_export.json` shape (per `docs/server-metrics/server-metrics-json-schema.md`):

```js
{
  summary: { ... },
  metrics: {
    "vllm:kv_cache_usage_perc": {
      type: "gauge" | "counter" | "histogram" | "info",
      series: [
        {
          endpoint_url: "http://...",
          labels: { engine: "0", model_name: "...", ... } | null,
          stats: { avg, max, min, p50, p99, ... } | { total, rate, rate_max, ... } | { count, sum, avg, sum_rate, p50_estimate, p99_estimate, ... },
          buckets: { "0.005": N, ..., "+Inf": N },     // histograms only
          timeslices: [ { start_ns, end_ns, avg|rate|count, ... }, ... ]  // optional
        },
        { /* per-endpoint replica */ },
        ...
      ]
    },
    ...
  }
}
```

Critical contract: **all series of one metric share bucket boundaries and label families**, but each is its own endpoint+label combination.

## Helpers (`helpers.js`)

Pure functions. No preact. Importable in isolation.

```js
// Backend detection
detectBackends(metrics) ->
  { dynamoFrontend, dynamoComponent, vllm, sglang, trtllm, kvbm }   // booleans

// Stat extraction (returns null on missing)
extractStat(metric, field)                  // metric.series[*].stats[field], single-series fallback to series[0]
extractStatPerSeries(metric, field)         // -> [{ series, value }] for all series
seriesLabel(series)                         // short display label: prefer dynamo_component, then tp_rank/pp_rank, then short(endpoint_url)
seriesColor(series, idx, palette)           // stable color assignment per-series within a render

// Time evolution
extractTimeslices(metric, field)            // -> [{ series, points: [{ t_sec_from_start, value }] }]
                                            //    field may be 'avg' (gauge), 'rate' (counter), 'avg' (hist sum/count)

// Histogram helpers (move from job-detail.js)
buildHistogramChartData(metric, color, options?)   // identical contract to current
sumBucketsAcrossSeries(metric)                     // for a fallback "all-endpoints" histogram (not used by default)

// Capability lookup
backendMetric(backendsPresent, capability)         // capability ∈ { 'kvCachePct', 'queueDepth', 'requestsRunning', 'preemptions',
                                                   //                'reqRate', 'promptTokRate', 'genTokRate', 'outputTokRate',
                                                   //                'ttftP99', 'e2eP99', 'itlP99', 'queueTimeP99', ... }
                                                   // returns { name, type, statField } picking the best metric for a capability
                                                   // given which backend(s) emitted; prefers Dynamo Frontend (user-facing)
                                                   // when both Dynamo and a backend are present.

// Static info
extractInfoLabels(metric)   // for type="info", returns merged labels across series with conflict markers
```

## Sub-Section Contracts

All sub-sections receive `{ metrics, backendsPresent }` and return either an HTML card or `null`.

### Deployment Card (`deployment-card.js`)

Single combined card (no per-endpoint split). Two-column key/value grid. Sources:
- Dynamo Frontend gauges: `model_context_length`, `model_kv_cache_block_size`, `model_max_num_batched_tokens`, `model_max_num_seqs`, `model_total_kv_blocks` — read `series[0].stats.avg` (constant gauges).
- vLLM: labels of `vllm:cache_config_info` (`block_size`, `cache_dtype`, `enable_prefix_caching`, `gpu_memory_utilization`, `num_gpu_blocks`).
- SGLang gauges: `engine_startup_time`, `engine_load_weights_time`, `is_cuda_graph`.

If a value differs across series for the same key, append a small "differs across N endpoints" note inline rather than picking one. Card hidden if no info-class fields are present.

### Capacity Section (`capacity-section.js`)

KPIs + per-endpoint table. Capabilities:

| KPI | Dynamo Frontend | vLLM | SGLang | Stat field |
|---|---|---|---|---|
| KV cache usage % | `dynamo_component_kvstats_gpu_cache_usage_percent` | `vllm:kv_cache_usage_perc` | `sglang:token_usage` | gauge `max` (peak), label shows avg in tooltip |
| Requests running | `dynamo_frontend_inflight_requests` | `vllm:num_requests_running` | `sglang:num_running_reqs` | gauge `avg` |
| Requests waiting | `dynamo_frontend_queued_requests` | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | gauge `avg`, label shows max in tooltip |
| Preemptions | — | `vllm:num_preemptions` | `sglang:num_retracted_reqs` | counter `total` |
| Prefix cache hit rate | `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (ratio) | `sglang:cache_hit_rate` | gauge `avg` or computed ratio |

Aggregate KPI tile: max-of-maxes (KV%, queue), sum (preemptions), avg (running, waiting, hit-rate). Below it: a per-endpoint table — one row per series — with columns `Endpoint | Role | KV% (avg/max) | Running | Waiting | Preempts | Hit-rate`. Role column derived from `seriesLabel()`.

### Throughput Section (`throughput-section.js`)

KPI tiles read counter `rate`:

| KPI | Dynamo Frontend | vLLM | SGLang | Stat field |
|---|---|---|---|---|
| Requests/sec | `dynamo_frontend_requests` (status=success) | `vllm:request_success` | — | counter `rate` |
| Prompt tokens/sec | — | `vllm:prompt_tokens` | — | counter `rate` |
| Generation tokens/sec | `dynamo_frontend_output_tokens` | `vllm:generation_tokens` | `sglang:gen_throughput` | counter `rate` (gauge `avg` for SGLang) |
| Output tokens (total) | `dynamo_frontend_output_tokens` | `vllm:generation_tokens` | — | counter `total` |

Per-endpoint rows with per-series rates show imbalanced workers immediately.

### Latency Section (`latency-section.js`)

Per capability, render: KPI tile (p50 + p99) + histogram chart (small-multiples by endpoint, max 6 endpoints — overflow shows "+N more" link).

| Capability | Dynamo Frontend | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|
| End-to-end latency | `dynamo_frontend_request_duration_seconds` | `vllm:e2e_request_latency_seconds` | — | `trtllm:e2e_request_latency_seconds` |
| TTFT | `dynamo_frontend_time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` | — | `trtllm:time_to_first_token_seconds` |
| ITL | `dynamo_frontend_inter_token_latency_seconds` | `vllm:inter_token_latency_seconds` | — | `trtllm:time_per_output_token_seconds` |
| Queue time | — | `vllm:request_queue_time_seconds` | `sglang:queue_time_seconds` | `trtllm:request_queue_time_seconds` |

KPI tiles use `stats.p50_estimate` and `stats.p99_estimate` — formatted via `fmtLatencyStr` (auto ms→s).

Histograms: keep current `buildHistogramChartData` algorithm (cumulative→delta, trim zero-tails, +Inf normalization). For multi-endpoint, render side-by-side small-multiples in a CSS grid; cap at 6 charts (chosen by sorted `endpoint_url`). If N > 6 the wrapper card adds a dim sub-line "showing 6 of N endpoints" so the truncation is observed.

When both Dynamo Frontend and a backend (vLLM/SGLang/TRT-LLM) emit the same capability, prefer Dynamo Frontend (user-facing measurement). Show backend version under a sub-header "Backend-side ..." below.

### Timeline Section (`timeline-section.js`)

Line charts of evolution across the run. Driver: pick metrics whose `timeslices` carry meaningful change. X-axis is seconds-from-run-start (`(end_ns - first_start_ns) / 1e9`). Y-axis depends on metric.

Charts (each conditional on data):

1. **KV cache usage over time** — gauge `avg` per timeslice; one colored line per series.
2. **Queue depth over time** — `requests_waiting` gauge `avg` per timeslice.
3. **Throughput over time** — generation-tokens counter `rate` per timeslice; one line per series.
4. **TTFT p50 over time** — histogram `avg` per timeslice (true percentiles need bucket recomputation; `avg` is a reasonable proxy and keeps the math simple — the design accepts this trade-off for the timeline view).

Each chart is a `<ChartWrapper type="line">` with multi-dataset `data.datasets`. Labels (X) shared, values (Y) per-series. Use `seriesColor()` for stable per-endpoint coloring.

## Empty / Loading States

Preserved from current code:

- `serverMetrics === null` AND export file present in listing → "Loading server metrics…" placeholder (job-detail.js owns this).
- `serverMetrics` parsed but every sub-section returns `null` → explicit "No server metrics collected for this run. The endpoint did not expose vLLM-compatible metrics, or the scrape interval did not capture any points." card with `data-testid="job-detail-server-metrics-empty"`.
- Export file absent in listing → no UI at all (today's behavior).

## Backend-Overlap Rules

When both Dynamo Frontend (`dynamo_frontend_*`) and a backend engine (vLLM/SGLang/TRT-LLM) report the same capability:

- KPIs: Dynamo Frontend wins (user-facing). Backend value shown in a dim "Backend-side: X" sub-line.
- Histograms in Latency: render Dynamo Frontend version primary; backend version below under a "Backend-side" sub-header.
- Capacity: KV cache is mutually exclusive in practice (Dynamo's source IS one of the backends' values), so just take whichever is present, preferring `dynamo_component_kvstats_gpu_cache_usage_percent` when both.

## Per-Endpoint Identification (`seriesLabel`)

Decision order for a short display label per series:
1. `labels.dynamo_component` if present (e.g. `prefill`, `decode`).
2. If `labels.tp_rank` and/or `labels.pp_rank` present, label as `tp{N}` or `tp{N}/pp{M}`.
3. If `labels.engine` present (vLLM), label as `engine-{N}`.
4. Fallback: derived from `endpoint_url` — strip protocol, take hostname's first labeled segment (e.g. `prefill-worker-0` from `http://prefill-worker-0.svc:9090/metrics`).

This naturally surfaces the prefill/decode role for disaggregated deployments without any user config.

## Color Assignment

`seriesColor(series, idx, palette)`: stable, deterministic per-render. Cycles through `[blue, accent, peach, mauve, sapphire, yellow, red, pink]` keyed by sorted `endpoint_url`.

## CSS

Reuse existing `.card`, `.card-title`, `.kpi-row`, `.metric-card`, `.text-dim`, `.chart-container`. New utility classes added to `style.css` only if a sub-section needs them (e.g. small-multiples grid `.sm-grid` and per-endpoint table `.per-endpoint-table`). Keep additions minimal.

## Implementation Plan

Sequential (one file depends on the previous one's exports):

1. **`helpers.js`** — pure functions; the surface every other file imports. Includes `detectBackends`, `extractStat`, `extractStatPerSeries`, `seriesLabel`, `seriesColor`, `extractTimeslices`, `buildHistogramChartData` (relocated from job-detail.js), `backendMetric`, `extractInfoLabels`, `formatTimesliceX`.
2. **`deployment-card.js`** — uses helpers only.
3. **`capacity-section.js`** — uses helpers + `KpiCard`.
4. **`throughput-section.js`** — uses helpers + `KpiCard`.
5. **`latency-section.js`** — uses helpers + `KpiCard` + `ChartWrapper`.
6. **`timeline-section.js`** — uses helpers + `ChartWrapper`.
7. **`index.js`** — wires the above; runs `detectBackends` once; passes derived data down; owns the empty-state card.
8. **`pages/job-detail.js`** — delete `getSeriesValue`, `buildHistogramChartData`, `ServerMetricsSection` (lines 1083-1285); replace inline use at line 2223 with `<ServerMetricsSection serverMetrics=${serverMetrics} />` imported from `../components/server-metrics/index.js`.
9. **`style.css`** — add `.sm-grid`, `.per-endpoint-table`, `.server-metrics-section` minor styling if needed; reuse otherwise.

## Verification

- `node --check` each new `.js` file (parse-only smoke test).
- Manual browser load: navigate to a completed job's detail page, confirm the section renders with realistic data. Required test cases:
  - vLLM-only single endpoint (today's baseline) → Capacity/Throughput/Latency populated, Timeline shows lines, Deployment shows cache-config labels.
  - Dynamo Frontend + Dynamo Component + vLLM (full Dynamo deployment) → Dynamo Frontend wins KPIs, backend shown as "Backend-side"; Deployment card shows model context length etc.
  - Disaggregated prefill+decode (multiple Dynamo Component series with different `dynamo_component` labels) → per-endpoint rows clearly labeled `prefill` and `decode`.
  - Empty `metrics: {}` (export existed but no data) → existing empty-state card renders.
  - File absent → no Server Metrics section at all.
- Confirm no regression of existing test-id `job-detail-server-metrics-empty`.

## Out-of-Scope (deferred)

- Per-endpoint cross-filtering (clicking a role to filter the rest of the page).
- Bucket re-computation for true aggregate p99 from multi-series histograms.
- Compare-mode integration (`pages/compare.js` does not currently handle server metrics).
- Surfacing KVBM block-transfer counters (only relevant when KVBM is enabled; skip until needed).
