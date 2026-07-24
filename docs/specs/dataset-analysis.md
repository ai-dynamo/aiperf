<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dry-run dataset analysis

## Purpose

Turn a `dry_run` into the product's analytical preview surface. A dry run
executes the full authored schedule in virtual time against the analytic latency
leaf (no server, sub-second), so it already produces *real* per-request execution
records: when each request and turn actually fired, how deep concurrency ran,
what each request's fabricated TTFT/ITL/latency was. This record separates the
requirement to distil those records — together with the compiled dataset's shape
and its prefix/KV-cache-reuse structure — into one human- and machine-readable
analysis report emitted at the end of every dry run.

The analysis answers, before a single real GPU-second is spent: *what does this
workload actually look like, and how will it stress a server?* — ISL/OSL
distributions, turn-by-turn history growth, prefix-cache reuse (ideal and under
finite KV capacity with eviction), concurrency over the run, achieved throughput,
and scheduling backlog.

## Built

- The `dry_run` transport executes the authored workload on a virtual `SimClock`
  and fabricates each request's timing from the analytic latency model
  (`ttft`/`itl` with ISL/OSL scaling, concurrency-contention terms, and seeded
  jitter). See [execution-model.md](execution-model.md) and
  `rust/runtime/src/engine/dry_run.rs`.
- Every finished request is captured as a `CapturedRecord` wrapping a
  `RecordIngest`, which carries `start_ns`, `end_ns`, `first_token_ns` (TTFT
  anchor), `token_arrival_ns`, input/output token counts, `turn_index`,
  `conversation_id`, `session_num`, and phase. On the graph/dry-run path the full
  set is retained in `GraphPhaseRunOutput.captured` in retain mode.
- The compiled dataset is in scope at the executor as the graph input bundle
  (`GraphInputBundle`: per-trace plans, the content-addressed segment store, and
  metadata), carrying per-turn ISL/OSL and per-turn cache-block identities
  (`hash_ids` + `block_size`) plus shared system/user-context segment handles.
  See [dataset.md](dataset.md).
- The runtime owns a Unicode-aware console table/percentile/histogram renderer
  (`render_table`) and the typed `NativeReport`; the `genai_perf` exporter
  already emits the ISL/OSL percentile stat set (`avg,min,max,sum,p1..p99,std`)
  post-run.
- The CLI enables dry-run through `--dry-run` (sets `transport.type: dry_run`),
  and the per-record artifact writers (`records`/parquet/csv) are driven directly
  from the executor where both records and the dataset are in scope — the
  precedent this analysis follows.

### Placement and gating

The analysis is a **runner-driven writer**, not an `Exporter`: it runs inside the
executor (`rust/runtime/src/engine/dataset_analysis_writer.rs`) where both the
retained record set and the compiled `GraphInputBundle` are simultaneously in
scope (the report/exporter stage sees neither). It emits its artifacts alongside
the existing per-record artifact writers.

It is **on by default for a dry run** and a no-op otherwise (`--dry-run &&
!--no-dataset-analysis`, `cli/src/load.rs`): the CLI populates
`Artifacts.dataset_analysis_path` (and the cache-model knobs) whenever `--dry-run`
is set and not suppressed; the executor writes the report when that path is
present.

On the graph path (recorded `weka_trace`/`dynamo_trace` or authored `dag_jsonl`)
the per-request structure is present *statically* in the compiled
`GraphInputBundle`, so a `--dry-run` that dispatches zero records still yields a
fully populated dataset characterization from each `LlmNode`'s input token count,
generation cap, conversation id/turn index, and ordered content-addressed prompt
segment handles — those BLAKE3 prefix-chained handles stand in for exact prefix
identity (`IdentitySource::HashIds`). The scheduled path has no
`GraphInputBundle`; there turn observations derive from the retained
`CapturedRecord` set, and block identities absent from that set fall back to the
length-structure identity source.

### Analysis catalog

All distributions report the existing stat set (count, mean, std, min,
p1/p5/p10/p25/p50/p75/p90/p95/p99, max, sum) and, where useful, an ASCII
histogram. Every section degrades gracefully and labels missing inputs.

**A · Dataset shape.** Conversation / turn / request counts; turns-per-conversation
distribution + histogram; single- vs multi-turn split; models and endpoints
present; streaming on/off counts; whether authored timing data exists.

**B · Sequence lengths.** ISL, OSL, and ISL+OSL total distributions; ISL:OSL
ratio; token budgets (Σ prompt, Σ completion, grand total); ISL and OSL
histograms.

**C · Turn-by-turn.** Aggregated by turn index: per-index ISL and OSL
distributions, mean history growth (Δ-ISL from the prior turn), and the count of
conversations reaching each turn; authored inter-turn think-time (`delay_ms`)
distribution. Optional full per-conversation listing for small datasets.

**D · Prefix / KV-cache reuse (centerpiece).** Computed exactly from a tiered
block-identity source — per-turn `hash_ids` when present, else chained
block-hashes over materialized token ids, else the length-structure fallback
(intra-conversation history prefixes are exact from accumulated ISL alone;
cross-conversation reuse via identical shared system/user-context handles). The
identity source and its confidence are labeled.
- *Ideal (unbounded):* prefix-cache hit rate = cached ÷ total prompt tokens;
  prefill tokens saved; unique-prefix (root) count; unique-block count;
  **intra- vs cross-conversation reuse split** (tokens and %); **shared-prefix-rate
  distribution** (share of requests reusing ≥25/50/75/90/100% of their prompt
  prefix); per-block reuse-frequency histogram and hottest blocks;
  system-prompt reuse (distinct system prompts and their lengths).
- *Realized (finite KV + LRU, arrival order):* block-LRU replay over the actual
  dry-run arrival order across a **cache-size sweep** (fractions of the
  working-set footprint) → realized hit rate, evictions, and average reuse
  distance at each size, rendered as a hit-rate-vs-cache-size curve; an explicit
  `--kv-cache-blocks` pins one point. Reports the ideal↔realized gap (reuse lost
  to eviction). Block size defaults to 16 (`--kv-block-size`).

**E · Compute proxy.** Σ prefill tokens with vs without cache (prefill reduction
from caching), Σ decode tokens, prefill:decode ratio, KV footprint in blocks.

**F · Execution timeline (from the real dry-run schedule).**
- *Concurrency over time:* in-flight count reconstructed by a sweep-line over
  record `[start_ns, end_ns)` intervals — peak, average, percentiles, ASCII curve.
- *Throughput over time:* achieved requests/s and output-tokens/s bucketed across
  the run timeline (curve + steady-state average); shows ramp, plateau, drain.
- *Turn-by-turn timing:* actual dispatch time per turn index, authored vs realized
  inter-turn think-time, and per-session timelines.
- *Queue/backlog:* arrivals waiting when a request-rate schedule outpaces served
  capacity under the analytic latency model, plus an optional per-request
  timeline/gantt artifact.

### Architecture

- `rust/runtime/src/dataset/analysis.rs` — pure-logic, deterministic, serde
  types and the `analyze` entry point over a record iterator + dataset
  structure. Unit-testable in isolation.
- `rust/runtime/src/dataset/analysis/prefix_cache.rs` — block-identity extraction
  (tiered source), the prefix tree for ideal reuse, and the arrival-order LRU
  replay for realized reuse.
- `rust/runtime/src/export/analysis_txt.rs` — console rendering reusing the
  `render_table` primitive → `dataset_analysis.txt`.
- `rust/runtime/src/export/dataset_analysis.rs` — `dataset_analysis.json` and
  `.csv` in the `genai_perf` stat-key style.
- `rust/runtime/src/export/analysis_html.rs` — a standalone interactive
  `dataset_analysis.html` report.
- `rust/runtime/src/engine/dataset_analysis_writer.rs` — the executor adapter:
  projects captured records and the compiled `GraphInputBundle` into the
  neutral analysis inputs and drives all four sink writers.
- CLI: `Artifacts.dataset_analysis_path` populated when `--dry-run` and not
  `--no-dataset-analysis` (`cli/src/load.rs`); `--kv-block-size`,
  `--kv-cache-blocks`, `--dataset-analysis-per-conversation` (`cli/src/flags.rs`).

### Verification

Analysis-only (no dispatch), so the generated-token timing e2e requirement does
not apply. Coverage is unit tests across `dataset/analysis.rs`,
`dataset/analysis/prefix_cache.rs`, and the export/writer modules: hand-computed
fixtures for known ISL/OSL sets (exact percentiles), multi-turn + shared-system
fixtures (exact ideal hit rate, intra/cross split, realized LRU hit rate at a
known capacity), and known interval sets (exact concurrency curve).

## Source anchors

- `rust/runtime/src/engine/dry_run.rs` — the analytic dry-run leaf whose schedule
  and fabricated timing the timeline sections analyze.
- `rust/runtime/src/metrics_core/ingest.rs` — `RecordIngest` (per-record execution
  data); `rust/runtime/src/engine/records.rs` — `CapturedRecord`.
- `rust/runtime/src/engine/dataset_analysis_writer.rs` — executor wiring: maps
  captured records and the `GraphInputBundle` into analysis inputs and writes
  `dataset_analysis.{txt,json,csv,html}`.
- `rust/runtime/src/dataset/analysis.rs`, `dataset/analysis/prefix_cache.rs` —
  the pure-logic analysis.
- `rust/runtime/src/export/analysis_txt.rs`, `export/dataset_analysis.rs`,
  `export/analysis_html.rs` — the four sink writers.
- `rust/runtime/src/graph/input.rs` — `GraphInputBundle` (compiled dataset,
  segments, per-turn ISL/OSL and prompt-segment handles) available at the
  executor.
- `rust/runtime/src/dataset/` — the compose/materialize pipeline and the shared
  system/user-context handles; see [dataset.md](dataset.md).
- `rust/runtime/src/export/console_txt.rs` — `render_table`, reused for the
  console report; `rust/runtime/src/export/genai_perf.rs` — the stat-key style the
  JSON/CSV mirrors.
- `rust/cli/src/model/artifacts.rs` — the artifact toggle; `rust/cli/src/flags.rs`
  and `rust/cli/src/load.rs` — the `--dry-run` enable path and the knobs.
