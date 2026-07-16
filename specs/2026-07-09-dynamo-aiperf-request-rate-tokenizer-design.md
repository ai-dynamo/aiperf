# dynamo-aiperf Increment 2: request-rate + tokenizer-exact via shared WorkloadDriver — Design

- **Date:** 2026-07-09
- **Status:** SUPERSEDED — historical precursor / lineage only. This increment was implemented (2026-07-09, commit `449b4f8`) against the **prior `dynamo-aiperf-native` tree**, not the current standalone AIPerf Rust workspace under `rust/`. It is retained as the design lineage for the request-rate/tokenizer increment that later informed `loadgen-core`, `aiperf_runtime::rng`, the dataset store, and the scheduled-workload designs. Current product truth is the native `aiperf` binary (crate `aiperf-cli`) front door plus its
`aiperf --execute` execution engine (Python Config-v2 frontend only on the `AIPERF_NATIVE=0` path); default Rust builds carry no Dynamo dependency, and optional Dynamo replay enters only through the curated `dynosim` adapter. Read this file for design intent, not for current reality.
- **Original working tree (historical):** `/home/anthony/nvidia/projects/dynamo-aiperf-native`, branch `ajc/dynamo-aiperf-skeleton`.
- **What was built (in that tree):** Tokenizer-exact prompts (`promptgen.rs`, TinyLlama fixture), concurrency + Poisson request-rate driven through the shared `WorkloadDriver` (`driven.rs`), CLI flags, `CollectorObserver` extracted. Zero `lib/mocker` changes needed (all driver/trace APIs were already `pub`). Proven via binary in both modes (50 reqs: concurrency-8 ~27k tok/s burst vs 200 qps ~7k tok/s paced; 1600 output tokens correct). Refinement discovered during grounding: `WorkloadDriver` mints its own uuids and synthesizes tokens, so the driver is used purely as a scheduler and prompts are generated per-`ReadyTurn` (no text side-table) — un-deferring Task 12 cleanly.
- **Builds on:** Increment 1 (walking skeleton), spec `2026-07-09-dynamo-aiperf-shared-core-design.md`. Un-defers that plan's Task 12.

## 1. Goal

Two capabilities, plus the integration that ties them together:
1. **Tokenizer-exact prompts** — generate prompts of an exact token length using a real HF tokenizer (removing increment 1's approx-length simplification).
2. **Request-rate load mode** — Poisson/constant arrival scheduling, in addition to closed-loop concurrency.
3. **Drive the HTTP path through `dynamo-mocker`'s shared `WorkloadDriver`** — un-deferring Task 12. This is now clean because a real tokenizer gives us actual token ids, dissolving the text-vs-token impedance that forced the deferral.

## 2. Why the impedance is gone

Increment 1 deferred Task 12 because the HTTP workload was text-only while `WorkloadDriver` is token/hash-native. With a tokenizer, each synthetic prompt is a real token sequence *and* real text: the token ids feed the driver's `Trace` (KV-hash bookkeeping), the decoded text feeds `HttpSink`. A per-request side table (`uuid -> prompt_text`) bridges the driver's token-native `ReadyTurn` back to the HTTP payload.

## 3. Enabling APIs (grounded)

- **Tokenizer:** `dynamo_llm::tokenizers` (re-export of `dynamo-tokenizers`), `Encoding` type; `tokenizer.encode(&str) -> Result<Encoding>`, `.decode(&[token])`. Fixtures for tests: `lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct/` (and siblings) contain `tokenizer.json`.
- **Synthetic trace:** `Trace::synthetic(SyntheticTraceSpec { block_size, num_sessions, turns_per_session, input_tokens: LengthSpec, output_tokens: LengthSpec, shared_prefix_ratio, num_prefix_groups, first_turn_arrivals: ArrivalSpec, inter_turn_delays: DelaySpec, seed })`. `ArrivalSpec::{Burst, ConstantQps, PoissonQps, GammaQps}` already exist.
- **Driver:** `Trace::into_trace_driver_with_block_size(bs)` and `into_concurrency_driver_with_block_size(bs, max)` produce a `WorkloadDriver`. Drive via `pop_ready(now_ms, limit) -> Vec<ReadyTurn>`, `next_ready_time_ms()`, `on_complete(uuid, now_ms)`, `release_cap_slot`, `is_drained()`. `ReadyTurn { request_uuid, session_id, turn_index, scheduled_ready_at_ms, request: DirectRequest, .. }`.
- **Facade gap:** `Trace::synthetic`, the `into_*_driver` methods, `SyntheticTraceSpec`, `ArrivalSpec`, `DelaySpec`, `LengthSpec`, and the `WorkloadDriver` driving methods must be reachable from `dynamo-aiperf`. Some are `pub` already (increment 1 exposed `WorkloadDriver`, `Trace`, `SyntheticTraceSpec`, `ReadyTurn`); the `into_*_driver` methods and `Trace::synthetic` need visibility confirmation/widening (curated facade, same pattern as increment 1).

## 4. Architecture

```mermaid
flowchart LR
  TOK[HF tokenizer\ndynamo-tokenizers] --> GEN[Prompt generator\nrandom tokens -> text]
  GEN --> TRACE[Trace::synthetic\n+ arrival spec]
  GEN --> MAP[(uuid -> prompt_text\nside table)]
  TRACE --> DRV[WorkloadDriver\nconcurrency | request-rate]
  DRV -->|pop_ready ReadyTurn| LOOP[run loop]
  MAP --> LOOP
  LOOP -->|DispatchRequest w/ prompt_text| HTTP[HttpSink]
  HTTP --> OBS[RequestObserver -> TraceCollector]
  LOOP -->|on_complete| DRV
```

- **Prompt generator** (`dynamo-aiperf`): given a tokenizer + target ISL, produce `(token_ids, text)` pairs. Approach: sample in-vocab token ids to the exact length, decode to text; the token ids and text are consistent by construction.
- **Trace synthesis**: build a `SyntheticTraceSpec` (single-turn sessions for now) with the chosen `ArrivalSpec`; `Trace::synthetic`. The driver owns concurrency/arrival timing.
- **Driver loop** (replaces increment 1's semaphore loop in `run.rs`): pump `pop_ready`, spawn `HttpSink::dispatch` per `ReadyTurn` (looking up `prompt_text` by `request_uuid`), call `on_complete` on terminal, sleep to `next_ready_time_ms()` between rounds, exit on `is_drained` + zero in-flight.
- **Measurement**: unchanged — `on_arrival` at release, `HttpSink` emits `on_admit`/`on_token`/`on_terminal`, shared `TraceCollector`.

## 5. Scope

**In:**
- HF tokenizer load (`--tokenizer <path>`), exact-ISL prompt generation.
- `SkeletonWorkload` gains a mode: `Concurrency { max_in_flight }` or `RequestRate { qps, arrival: poisson|constant }`.
- Drive both modes through `WorkloadDriver` (concurrency-mode wiring first — mechanically un-defers Task 12 — then request-rate).
- `uuid -> prompt_text` side table; `ReadyTurn -> DispatchRequest` mapping.
- CLI flags for `--tokenizer`, `--mode`, `--request-rate`, keeping positional base_url/model.

**Out (later):**
- Multi-turn / trace-file / agentic workloads (single-turn synthetic only here).
- Shared-prefix / prefix-group control on the HTTP side (the spec exists; defer tuning).
- Non-chat endpoints.
- `loadgen-core` extraction.

## 6. Key decisions (defaults chosen; flag if you disagree)

- **Prompt strategy:** sample random in-vocab token ids to exact ISL, then decode to text. Guarantees exact input token count.
- **Tokenizer for tests:** the in-repo `mock-llama-3.1-8b-instruct/tokenizer.json` fixture (no network).
- **Order:** wire `WorkloadDriver` in concurrency mode first (regression-safe, un-defers Task 12), then add request-rate arrival.
- **Driver reuse over reimplementation:** accept that the HTTP path pays for the driver's KV-hash bookkeeping it does not consume, in exchange for sharing the real arrival/admission logic (the whole point of this increment).

## 7. Verification

- Unit: prompt generator produces exact token count (encode(text).len() == target); trace synthesis yields expected turn count; request-rate arrivals honor qps within tolerance.
- E2E: run both modes against the standalone mock server; assert request/token counts and finite TTFT/throughput; for request-rate, assert measured arrival spacing ≈ 1/qps.
- Regression gate: full `dynamo-mocker` suite (497) stays green after any facade widening.
- Binary smoke: `aiperf --tokenizer <fixture> --mode request-rate --request-rate 50 <mock-url>`.

## 8. Risks

- **`ReadyTurn` field/visibility surface:** the driving methods (`pop_ready` etc.) and `Trace::synthetic`/`into_*_driver` may need curated `pub` widening; confirm before coding (grep-first, like increment 1).
- **Tokenizer construction API:** the concrete loader type for `tokenizer.json` in `dynamo-tokenizers` is not yet pinned; a short spike is the first plan task.
- **Arrival accuracy under load:** request-rate honoring depends on the driver's `next_ready_time_ms` cadence; tolerance-based assertions only.
- **Decoded-token text round-trip:** random in-vocab ids may decode to odd text; acceptable for load generation (server still processes tokens), but note it in a comment.
