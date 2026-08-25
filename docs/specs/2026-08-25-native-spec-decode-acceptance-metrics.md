<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native per-request speculative-decode acceptance metrics

## Status

Approved design for the native Rust port of origin/main commit
`d32f4bb98edbeac1374ec816aee32d7e4517c5ae`.

## Problem

The native runtime captures OpenAI prediction-token usage and SGLang
server-wide speculative-decoding gauges, but neither surface describes how one
vLLM request's speculative verification behaved. Native profiles consequently
cannot report verification steps, proposed draft volume, acceptance length,
accepted-per-verified ratio, or the distribution of accepted drafts per step.

The port must keep this request data distinct from endpoint `usage` and from
server telemetry. It must also survive the native thread-per-core, sketch, and
cellular folds without adding synchronization to the request or token hot path.

## Goals

1. Capture vLLM's per-choice `speculative_decoding_stats` for chat and
   completions, streaming and non-streaming.
2. Normalize it into one engine-neutral, validated request record.
3. Emit the exact upstream scalar metric identities and formulas.
4. Pool exact accepted-draft histogram counts across profiling requests while
   keeping warmup and concrete authored phase instances separate.
5. Emit canonical per-record, native-report, GenAI-Perf v1 JSON, and console
   representations, omitting all of them when stats are absent.
6. Prove the complete path with the deterministic Rust mock server and a real
   `aiperf profile` subprocess.

## Non-goals

- Reinterpreting `usage_accepted_prediction_tokens` or
  `usage_rejected_prediction_tokens` as speculative-decoding data.
- Changing the existing SGLang server-metrics console table.
- Adding an engine-selection flag or accepting arbitrary provider shapes.
- Exposing the pooled histogram in CSV or Parquet; the upstream aggregate is a
  JSON/console dictionary and the canonical request record remains in JSONL.
- Defining pooled histograms for rolling/timeslice windows. Like upstream, a
  time-bounded export omits this run-level dictionary.

## Canonical request contract

The transport-neutral observer owns a serializable
`ObservedSpecDecodeAcceptance` value with these public artifact fields:

- `engine: String`
- `mean_acceptance_length: f64`
- `draft_acceptance_rate: f64` as a `0..=1` fraction
- `acceptance_histogram: BTreeMap<u64, u64>`
- `num_accepted_draft_tokens: u64`
- `num_draft_tokens: u64`
- `num_spec_steps: u64`
- `num_spec_tokens: Option<u64>`
- `completion_tokens: Option<u64>`
- `per_step_accepted: Option<Vec<u64>>`
- `per_step_drafted: Option<Vec<u64>>`

Normalization accepts only finite mean/rate values and non-negative integer
counts. Histogram keys are decimal JSON object keys. Its count sum must equal
`num_spec_steps`; its key-weighted sum must equal
`num_accepted_draft_tokens`; accepted drafts cannot exceed proposed drafts.
Each optional per-step vector must have `num_spec_steps` elements and reconcile
to its aggregate, and paired elements must satisfy accepted <= drafted.
Floating relationships are not re-derived, avoiding false rejection from
provider rounding. A malformed signature-matching payload logs one structured
warning and degrades to absence; it does not fail the request or profile.

## Wire capture and reduction

The accepted wire location is exactly
`choices[0].speculative_decoding_stats`. A `choices` array of length other than
one is suppressed because request-level usage cannot be attributed to one of
multiple sequences. Across a stream, the last non-empty stats object wins.

The ordinary decoded-JSON path extracts that object before endpoint response
reduction. The streamed-chat typed fast path adds one optional
`speculative_decoding_stats` field to `ChatChoice`; it must retain a finish-only
chunk even when the chunk has neither content nor usage. This is load-bearing:
vLLM emits stats on the finish-reason chunk and the later usage-only chunk takes
the generic path. Normal streaming chunks retain the allocation behavior of the
current typed codec because only the terminal stats object builds a
`serde_json::Value`.

After every response is reduced, the dispatch path normalizes the last stats
object using the reconciled completion-token count and emits exactly one
`RequestObserver::on_spec_decode_acceptance` callback before terminal status.
`NativeMetricsObserver` stores the owned value in worker-local pending state and
moves it into an appended `RecordIngest` field. `ObserverTee` forwards the new
event in deterministic delegate order. No mutex, channel, or cross-thread
shared state is added.

## Metrics and pooled histogram

The following tags append to `MetricTag` so existing dense-column indices stay
stable:

| Tag | Kind | Formula | Console |
| --- | --- | --- | --- |
| `spec_decode_acceptance_length` | record | wire `mean_acceptance_length` | Spec Decode |
| `spec_decode_token_weighted_acceptance_length` | derived | `1 + total_accepted / total_steps` | Spec Decode |
| `spec_decode_draft_acceptance_rate` | record | `100 * wire draft_acceptance_rate` | Spec Decode |
| `spec_decode_overall_draft_acceptance_rate` | derived | `100 * total_accepted / total_draft` | Spec Decode |
| `spec_decode_accepted_per_verified` | record | `(accepted + steps) / (draft + steps)` | Spec Decode |
| `spec_decode_steps` | record | request `num_spec_steps` | Spec Decode |
| `spec_decode_accepted_draft_tokens` | record | request accepted drafts | hidden |
| `spec_decode_draft_tokens` | record | request proposed drafts | hidden |
| `total_spec_decode_steps` | derived sum | sum of request steps | hidden |
| `total_accepted_draft_tokens` | derived sum | sum of request accepted drafts | hidden |
| `total_draft_tokens` | derived sum | sum of request proposed drafts | hidden |

Zero denominators suppress only the undefined derived metric. The four visible
distribution/scalar families are larger-is-better where upstream marks them;
steps and token counts are neutral. Display orders are 5000, 5010, 5020, 5025,
5030, 5040, 5050, 5060, 5140, 5150, and 5160 respectively.

`ColumnStore` retains the optional canonical record beside each exact row and
an exact `u128` pooled counter keyed by `(Phase, Option<phase_index>)`. The
cellular MessagePack form retains `u64` counts for compatibility and refuses a
pooled value that cannot be narrowed exactly. Valid,
non-cancelled records update the counter during insertion. `ExportContext`
gains an optional concrete `phase_index`: exact row masks, bounded-memory
sketch keys, and histogram selection all apply it when present; a phase-only
context merges every same-kind instance. Time-bounded summaries return no
pooled histogram. Sketch mode harvests the scalar tags normally, clears the
row-owned canonical value, and retains the phase-instance pool. `append_store`
merges both structures, so worker and serialized cellular
`ColumnStorePartition` folds preserve counts. `AccumulatorSummary` carries the
optional sorted pool separately from scalar results. Histogram count sum must
equal the finite `total_spec_decode_steps` value for the same phase.

## Reports and artifacts

`NativeReport` carries an optional
`pooled_spec_decode_acceptance_histogram`, serialized only when non-empty. The
GenAI-Perf v1 JSON exporter copies the full map to the identically named
top-level field; JSON object keys are decimal strings. Scalar tags follow the
normal report/export projection and CSV remains scalar-only.

`profile_export.jsonl` adds an optional top-level `spec_decode_acceptance`
object copied from `RecordIngest`. The record's six record metrics remain in
its normal `metrics` object. Absent stats omit the top-level field and every
spec-decode metric.

`MetricConsoleGroup::SpecDecode` renders after Reasoning and before Default.
The title is `NVIDIA AIPerf: Spec Decode`. Immediately beneath its scalar table,
the console writes:

`Accepted drafts per step (% of steps):  0: 12%   1: 12%   2: 25%   3: 38%   4: 12%`

Buckets are shown from zero through the maximum, including empty gaps. If any
bucket is at least eight, buckets eight and above fold into `>=8`. An empty pool
prints no line. This block does not alter the separately rendered SGLang
server-metrics table.

## Deterministic integration contract

The Rust mock server adds an opt-in fixture that emits vLLM's canonical worked
example: per-step accepted drafts `[2, 3, 1, 4, 2, 0, 3, 3]`, histogram
`{0:1, 1:1, 2:2, 3:3, 4:1}`, eight steps, 18 accepted drafts, 32 proposed
drafts, mean acceptance length 3.25, draft acceptance rate 0.5625, and fixed
draft length four. Streaming chat puts this object on the finish-reason chunk,
with no content or usage, and preserves the later usage-only chunk.

`rust/e2e-tests/tests/test_spec_decode_acceptance.rs` launches that in-process
mock and a real `aiperf profile` subprocess. For `N` successful requests it
asserts totals `8N`, `18N`, and `32N`, token-weighted acceptance length 3.25,
overall draft acceptance rate 56.25, per-request accepted-per-verified 0.65,
the full pooled histogram scaled by `N`, the exact console block, and every
processed JSONL canonical record. A second run with the fixture disabled
asserts complete absence across console, `*_aiperf.json`, and JSONL.
