# Scheduled online path: thread-per-core worker-local metric accumulation

**Date:** 2026-07-12
**Status:** Runner path built (HTTP + gRPC scheduled); library path gated on A2.
**Scope:** Kill the single-coordinator observer-replay ceiling on the online
scheduled measurement path by giving each thread-per-core worker its own
`NativeMetricsObserver` and merging once at the join — exactly the pattern the
graph transport bench already uses. The runner product path (HTTP scheduled
`workers==1` and `workers>1`, plus the gRPC twin) ships this via a
**worker-local measured seam** (the former buffered replay path is removed); the
library `ScheduledRuntime`/`phase_runtime` path is still gated on A2 (planned) and
remains single-observer for now.

> All file:line citations are against the read-only working tree at
> `/home/anthony/nvidia/projects/aiperf/ajc/rust` as of this date. "code is
> truth" — verify before implementing; some line numbers drift.

---

## 1. Motivation — the replay ceiling, with evidence

The online scheduled path already fans transport work across OS threads
(thread-per-core), but **metric accumulation is single-threaded** because every
worker's observer events are buffered and replayed onto ONE coordinator-side
observer.

### 1.1 The buffer-and-replay machinery (pre-change HEAD — since removed)

> This subsection describes the **pre-change HEAD** machinery this spec set out
> to kill. As of the current tree the buffered path is **fully removed**:
> `BufferedObserver`, `ObserverEvent`, and the buffered `execute_turn(observer)`
> replay loop no longer exist anywhere in `rust/` (see §3.1). The file:line
> citations below are historical, describing the state the measured seam replaced.

`rust/runtime/src/runner_protocol/turn_execution.rs` (pre-change):

- `BufferedObserver` (lines 164–223): a `RequestObserver` whose `on_admit` /
  `on_token` / `on_classified_token` / `on_usage` / `on_endpoint_metrics` /
  `on_terminal` callbacks each push an `ObserverEvent` into a
  `RefCell<Vec<ObserverEvent>>`. `on_arrival` is intentionally dropped
  (lines 176–184) because arrival is owned coordinator-side.
- Each worker task builds a **worker-local** `BufferedObserver`
  (`execute_worker_command`, line 655), dispatches into it, then ships
  `observer.take()` (line 691) back to the coordinator inside `WorkerReply`
  (lines 225–228).
- The coordinator's `execute_command` drains the reply and **replays every
  buffered event onto the caller's single observer** (lines 516–518):

  ```rust
  for event in reply.events {
      event.replay(observer, run_origin_ns as f64 / 1_000_000.0);
  }
  ```

So a request that streams N output tokens produces N `ObserverEvent::Token`
values, crosses an mpsc boundary as a `Vec`, and is replayed one-by-one onto a
single-threaded observer graph. Accumulation cost is **O(total tokens) on one
core**, regardless of worker count. That is a candidate for the ~100k req/s
ceiling.

#### 1.1.1 The ceiling is per-request-bound as well as per-token-bound

**Design caveat (retained for the library path and future tuning).** This spec
relocates only *token accumulation* off the coordinator. But the coordinator
remains a
single-threaded per-*request* funnel even after A1: per request it still does
`Box::pin(async)` + `turn.clone()` (`scheduled.rs:905,924,933`),
`PreparedHttpTurn::from_turn` (`execute.rs:3283`), one mpsc `WorkerCommand`
send + one oneshot `WorkerReply` recv + a `tokio::select!` loop
(`turn_execution.rs:467-521`), plus `recorder.terminal`, credit return, and
record processors — all on one `current_thread` coordinator runtime. If the
real limit is this per-request funnel rather than per-token replay, A1 delivers
**little or no gain** for the workloads where tokens-per-request is small:
1-output-token runs, non-streaming runs, and usage-only runs (where the entire
per-token replay path is a handful of events).

Therefore A1 is a **token-accumulation** relocation only. On long-output
streaming workloads it moves the dominant O(tokens) replay off the coordinator;
on short-output/non-streaming/usage-only workloads (where per-token replay is a
handful of events) the coordinator per-request funnel still dominates and the
persistent-lane restructure (A4 Finding #1A/#4b) is the remaining lever. When
profiling for a request-bound vs token-bound ceiling, run **both** a long-output
and a short-output/usage-only workload — a single ratio floor at N=4 cannot tell
them apart.

The gRPC path has an identical twin: `rust/runtime/src/runner_protocol/grpc_turn_execution.rs`
`BufferedObserver` (lines 156–204) and replay loop (lines 376–378).

### 1.2 Who the single coordinator observer actually is (runner product path)

The runner product path does **not** use the `ScheduledRuntime` observer. In
`rust/runtime/src/runner_protocol/execute.rs`, `ConfiguredDispatcher::dispatch_turn`
(lines 3268–3287) **ignores** the `ScheduledRuntime`-supplied `_observer`
(line 3271) and feeds the backend `self.capture.observer` instead. The comment
at lines 3276–3282 is explicit:

> The runner's native-v2 report is produced entirely from RunCapture's observer
> (`self.capture`); the ScheduledRuntime's runtime observer (CollectorObserver +
> NativeMetricsObserver) is computed and then discarded … Feed only the capture
> observer so the single dispatch thread does not replay every token into a
> discarded accumulator.

`RunCapture` (lines 3067–3111) owns exactly **one** `Rc<NativeMetricsObserver>`
(line 3070). `RunCapture::begin` (line 3090) calls `register_metadata` +
`on_arrival` coordinator-side per turn, with **no `request_index`** — so the
observer assigns dense arrival-order slots. Every buffered worker token is
replayed onto this one observer. **This single observer is the runner ceiling.**

### 1.3 The reference design that already scales — graph bench

`rust/runtime/src/graph/transport_bench.rs` keeps a **per-worker**
`MetricsAccumulator` and merges once at the thread join (lines 385–395):

```rust
let mut merged = MergedMetrics::default();
for worker in workers {
    ...
    merged.native.merge(&worker.native)
        .expect("workers share one metrics configuration");
}
let native_metrics = merged.native.summarize();
```

`rust/runtime/src/runner_protocol/graph_execution.rs` builds one `NativeMetricsObserver`
per graph worker and registers metadata worker-locally
(`RunnerGraphSink`, lines 787–805). **The scheduled path is the last consumer
still routing every token through a single coordinator observer.**

---

## 2. Target architecture

Move measurement to where the tokens already are.

```
                 coordinator (current_thread + LocalSet)
  ┌───────────────────────────────────────────────────────────┐
  │ Workload / ScheduledRuntime issuance                        │
  │  - SlotPool admission (STAYS coordinator-side)              │
  │  - per-turn: arrival facts + RequestMetricMetadata + phase  │
  │  - TimingRecorder (STAYS coordinator-side, O(requests))     │
  │  - credit return / record processors (O(requests))          │
  └───────────────┬───────────────────────────────────────────┘
                  │ WorkerCommand { turn, arrival, metadata, phase }
                  ▼
   worker 0            worker 1            worker N
  ┌──────────┐        ┌──────────┐        ┌──────────┐
  │Transport │        │Transport │        │Transport │
  │Sink      │        │Sink      │        │Sink      │
  │+ local   │        │+ local   │        │+ local   │   each owns its own
  │Native-   │        │Native-   │        │Native-   │   !Send Rc observer +
  │Metrics-  │        │Metrics-  │        │Metrics-  │   MetricsAccumulator;
  │Observer  │        │Observer  │        │Observer  │   tokens accumulate LOCAL
  └────┬─────┘        └────┬─────┘        └────┬─────┘
       │ WorkerReply { outcome, ttft }  (O(requests), NOT O(tokens))
       ▼                   ▼                   ▼
  ┌───────────────────────────────────────────────────────────┐
  │ at drain: each worker returns its drained MetricsAccumulator │
  │ (Send). Coordinator MERGES once, injects telemetry scalars,  │
  │ summarize() → one AccumulatorSummary.                        │
  └───────────────────────────────────────────────────────────┘
```

Key properties:

1. **Per-token work stays on the worker's core.** `on_token` accumulates into
   the worker-local observer; nothing crosses a thread per token.
2. **Cross-thread traffic drops from O(tokens) to O(requests):** on the measured
   seam only the `WorkerReply { result, ttft, live_record? }` crosses back per
   request; no per-token `ObserverEvent` is buffered or replayed. (The legacy
   buffered `execute_turn(observer)` path has been **removed entirely**; the
   measured seam is now the sole dispatch path — see §3.1.)
3. **Merge once at the join.** Each worker returns a drained, `Send`
   `MetricsAccumulator` (or `NativeMetricsFinalizer`); the coordinator merges
   with `MetricsAccumulator::merge`, injects run-level telemetry scalars, and
   calls `summarize()` exactly once.
4. **Timeslices / sweep-lines / per-model & per-worker series stay correct**
   because they are computed from the merged *store*, not merged from partial
   summaries (see §4).

Admission (`SlotPool`), issuance pacing, the `TimingRecorder`, credit return,
and record processors remain coordinator-side and unchanged. This is a
**measurement** relocation, not a scheduling one.

---

## 2.1 The runner product path is RECORDS-first re-ingest, NOT accumulator-merge

> The runner path (built) is **records-first**, which differs from the "merge
> worker accumulators / summarize once" framing in §2 (item 3), §3.5, §4.3, and
> §8. That merge-and-summarize mechanism is the correct one for the **library
> `ScheduledRuntime` path (§3.4)** — which summarizes its accumulator directly —
> but the WRONG mechanism for the **runner product path**, which is the actual
> throughput lever this spec exists to fix. This subsection is authoritative for
> the runner.

**How the runner builds its report (built; verified).** The runner does
**not** summarize the capture observer's accumulator. It:

1. drains per-request `RecordIngest`s from the single capture observer via
   `finish_with_records()` (`execute.rs:3201`), joining them to dispatch
   identities and **rewriting each record's `admit_ns`** to the coordinator
   credit-issued time (`execute.rs:3220-3224`; also the live path at
   `execute.rs:3183-3184`); then
2. re-ingests every finished record into a **fresh** `MetricsAccumulator` in
   **dispatch order** and summarizes that (`execute.rs:1557-1560`
   `for record in &captured { accumulator.process_record(&record.ingest) }`).

The report is a pure function of those re-ingested records, not of the observer's
own accumulator. Two facts make accumulator-merge the wrong seam here:

- **`admit_ns` is rewritten per record, after the worker stamps it.** The sink
  stamps `admit_ns ≈ dispatch time` worker-side (`http.rs:895`
  `obs.on_admit(uuid, admit_ms, 0)`, "admit == dispatch time"; identical gRPC
  twin `grpc.rs:313`). The coordinator then overwrites it with the credit-issued
  time in `finish` (`execute.rs:3220-3224`), from which
  `CreditToStartLatency`, `EffectiveLatency`, and `CreditDropLatency` are derived
  (`store.rs:1097-1103`, `start_ns − credit_ns` / `end_ns − credit_ns`). A worker
  accumulator that has **already** folded records into its store has these three
  metrics baked in from the worker's dispatch-time `on_admit` (queue ≈ 0), and
  the post-run rewrite is **impossible**. `MetricsAccumulator::merge` +
  `summarize()` would therefore report ~0 `CreditToStartLatency`/
  `EffectiveLatency`/`CreditDropLatency` for **every** run with a nonzero
  credit→dispatch queue (request-rate, adaptive, back-pressured concurrency) — a
  whole-metric corruption, not the sub-ULP reorder §4.5 anticipates.

- **Records carry a local `request_index` that collides across workers.** See
  §4.2 — this is a hard panic on re-ingest, not a reorder.

**Corrected runner mechanism (records-first).** Each worker owns its
`NativeMetricsObserver` and accumulates tokens locally (the O(tokens) relocation
that is the whole point). At drain, the backend hands back its workers'
**records** — a flat `Vec<(Uuid, RecordIngest)>` collected from each worker's
`finish_with_records()` (the `drain_records(end_ns)` seam), NOT their
`MetricsAccumulator`s. The coordinator:

1. concatenates per-worker record vectors,
2. **reassigns `request_index`** (§4.2) so the merged set is a dense, collision-free
   sequence — or sets it to `None` so re-ingest uses `push_record`,
3. uuid-joins to dispatch identities in dispatch order and rewrites `admit_ns`
   per record (the existing `finish` logic, §3.3, keyed by uuid per Finding 3 —
   never by `correlation_id`),
4. re-ingests the joined records into one fresh accumulator **in dispatch order**
   (the existing `execute.rs:1557-1560` loop, unchanged), and summarizes once.

Because step 4 folds in **dispatch order** — identical to HEAD's single-observer
slot order — the runner report is **byte-identical to HEAD, including the float
latency-distribution fields** (see the §4.5 correction). No IEEE-754 reorder
occurs on the runner path. `MetricsAccumulator::merge`, `drain_accumulators`, and
the whole "merge accumulators, never summaries" apparatus are **irrelevant to the
runner product path**; they remain the mechanism only for the library
`ScheduledRuntime`/`phase_runtime` summarize-directly path (§3.4), which is not the
product throughput lever.

**gRPC twin.** The gRPC path stamps and rewrites `admit_ns` identically
(`grpc.rs:313`), so its worker-local relocation must also be records-first, and
the regression matrix must add a gRPC parity row (currently absent).

---

## 3. Exact touch points (file:line)

### 3.1 Worker command / reply — the measured seam (built)

`rust/runtime/src/runner_protocol/turn_execution.rs`:

- **`BufferedObserver` / `ObserverEvent` / `execute_turn(observer)` are DELETED.**
  The buffered replay path described in §1.1 no longer exists anywhere in `rust/`;
  the measured seam below is the sole dispatch path. (The former agentic and
  external-evaluator verticals that once consumed the buffered replay have been
  removed from the runner; the remaining accuracy consumer — the static-accuracy
  pair, `online_execution.rs::OnlineStaticAccuracyAdapter` — also dispatches
  through the same measured seam via `ConfiguredDispatcher`.)
- **The measured seam is the only path.** `HttpTurnExecutionBackend` exposes
  `configure_measurement` / `execute_turn_measured`
  (+ `execute_turn_measured_streaming`) / `drain_records` — carrying a
  `MeasuredTurnContext` in and a `MeasuredTurnOutcome` out, implemented on
  `TransportSink`, `GrpcTransportSink`, `ThreadPerCoreHttpExecutionBackend`, and
  `ThreadPerCoreGrpcExecutionBackend`. The scheduled `ConfiguredDispatcher` calls
  `execute_turn_measured`. Each `WorkerCommand` carries a **non-optional**
  `MeasuredTurnContext` (`turn_execution.rs:117-119`); there is no buffered
  fallback branch.
- **Worker owns the observer.** Each worker builds one
  worker-local `Rc<NativeMetricsObserver>` next to its `TransportSink` via
  `configure_measurement`, dispatches into that observer, and finalizes the
  record worker-side; no per-token event crosses a thread. `WorkerReply` carries
  `{ result, ttft }` in the common case, and additionally a **non-consuming
  cloned** `live_record: Option<RecordIngest>` when a live sink is attached (see
  §3.3 — never the consuming `drain_terminal_record`).
- **Coordinator drain returns RECORDS, not accumulators.** Per §2.1,
  `drain_records(end_ns) -> Vec<(Uuid, RecordIngest)>` collects the backend's
  workers' `finish_with_records()` output across the thread join into one flat
  vec. The coordinator reassigns `request_index` (§4.2), uuid-joins/rewrites
  `admit_ns`, and re-ingests in dispatch order (`execute.rs:1557-1560`). No
  `drain_accumulators` / `MetricsAccumulator::merge` seam is used on the runner
  path, where the per-record `admit_ns` rewrite (§2.1) forbids pre-summarizing
  worker accumulators; accumulator-merge is the mechanism for the **library**
  summarize-directly path (§3.4) only.
- The `workers == 1` fast path runs the transport on the coordinator reactor and
  owns one measured observer directly with no channel; it also goes through the
  measured seam.

### 3.2 gRPC twin — mirrors the change (built)

`rust/runtime/src/runner_protocol/grpc_turn_execution.rs`: the `GrpcTransportSink` and
`ThreadPerCoreGrpcExecutionBackend` implement the same measured seam
(worker-local observer + `drain_records`, `grpc_turn_execution.rs:255/268`),
exactly as on the HTTP side. Like the HTTP path, the gRPC buffered
`ObserverEvent` / `BufferedObserver` / replay loop have been removed; the
measured seam is the sole path.

### 3.3 Runner capture — the real product surface

`rust/runtime/src/runner_protocol/execute.rs`:

- `RunCapture` no longer owns a single measurement `NativeMetricsObserver` for
  the scheduled path. Its responsibilities split:
  - **arrival + metadata:** `RunCapture::begin` runs on the coordinator thread in
    `dispatch_turn`'s synchronous prefix *before* backend dispatch and returns a
    `MeasuredTurnContext` that is handed to the worker; its push order is the
    global `request_index` (see below and §4.2). The worker records only
    transport facts.
  - **phase / session_num / admit_ns are patched at `finish`, NOT registered on
    the worker.** `CapturePhaseProcessor` stores `phase` / `session_num` /
    `has_credit_timestamp` / terminal per uuid; `RunCapture::finish` applies them
    to the drained record and sets `admit_ns = has_credit_timestamp.then(issued_time)`,
    keeping the credit-latency time base identical to HEAD. This resolves Risk 2
    without forwarding phase at issue.
  - **coordinator-side maps** `outputs` and `raw_exchanges` stay coordinator-side,
    keyed by uuid, fed from the returned outcome.
- `ConfiguredDispatcher::dispatch_turn` dispatches into the backend's
  worker-local observer via the measured seam; the per-request `record_response`
  is worker-side.
- **`RunCapture::finish` (3200–3234) is the hard blocker — see §5.** It currently
  asserts `collection.records.len() == identities.len()` (3205) and **zips
  records with identities positionally**, asserting
  `ingest.correlation_id == identity.uuid.to_string()` (3211–3218). Worker-local
  merge concatenates per-worker (not global dispatch order), so this positional
  invariant breaks. Replace with a **uuid-keyed join**, keyed on the record's
  **true `Uuid`**, then iterate `identities` (dispatch order) and look each up.
  This decouples report row order from accumulation order.
  - **Key on the drain-provided `Uuid`, NEVER on `ingest.correlation_id`
    (Finding 3).** `finish_with_records` has `entry.uuid` in hand at drain
    (`metrics.rs:434`) but folds it into `correlation_id` via
    `correlation_id.unwrap_or_else(|| uuid.to_string())` (`metrics.rs:473-476`).
    Today's `correlation_id == uuid.to_string()` assertion (`execute.rs:3217`)
    holds only because `RunCapture` uses `NativeMetricsObserver::new`
    (`execute.rs:3080`, `retain_record_dimensions=true`). The moment the runner
    runs **aggregate-only** (`records:false`, which A4 Finding #6 / PR4 item 3
    proposes extending to the runner), `register_metadata` sets
    `correlation_id = Some(String::new())` (`metrics.rs:261`), so **every** record's
    `correlation_id` is `""` and a `correlation_id`-keyed map collapses all
    identities onto one `""` key. Therefore the **drain seam must return each
    record paired with its `entry.uuid`** — e.g. `Vec<(Uuid, RecordIngest)>`
    (or a `{ uuid, ingest }` struct) — and the join keys on that `Uuid`. Do
    **not** re-derive the key by parsing `correlation_id`.
  - **Reassign `request_index` to the global dispatch ordinal before re-ingest
    (§4.2 HARD PANIC).** During the join, `finish` stamps each merged record's
    `request_index` = the **identity ordinal**, i.e. the **`RunCapture::begin`
    push order** (a coordinator-owned counter incremented in `dispatch_turn`'s
    synchronous prefix), which is globally-unique, dense, monotonic, and
    independent of worker count — equal to HEAD's single-observer arrival-slot
    order. The `execute.rs:1557-1560` re-ingest then lands each record at a
    unique, hole-free row in HEAD dispatch order — not a per-worker-local slot,
    and **not** `None`/push (which would re-ingest in drain order and drift float
    fields). Using `begin` push order (rather than threading `recorder.begin`)
    gives the collision-free, dense, HEAD-ordered re-ingest §4.2 requires. See
    §4.2 for the worker-internal-slot vs. record-`request_index` decoupling.

- **Pre-worker failures break the 1:1 identity↔record count guarantee — must be
  handled explicitly (see Risk 4).** Today the record for *every* dispatched
  identity is finalized coordinator-side, because on the `Err` path
  `ConfiguredDispatcher::dispatch_turn` (execute.rs:3314–3328) still calls
  `on_terminal(Failed)` + `record_response` on the single `RunCapture` observer,
  and the library path synthesizes `Canceled`/`Failed` records with
  `start=issued_ns`, `end=now` (scheduled.rs:951–999). Under A1 the worker owns
  the observer, but a command can fail **before any worker touches the request**:
  the `WorkerCommand` send can fail ("worker stopped before accepting a command",
  turn_execution.rs:476), the worker can drop the command before completion
  (turn_execution.rs:501–503), or a `PlacementCancellation` can fire
  (turn_execution.rs:465–466,681–692) — in each case no worker observer ever
  registers arrival/terminal for that uuid, so the merged store has **fewer
  records than identities** and the naive uuid-join lookup misses.
  **Design (built):** `RunCapture` keeps a coordinator-side fallback-finalization
  path for pre-worker failures. Because measurement now happens worker-side, the
  coordinator cannot re-use the worker observer; instead, for any identity with
  **no** drained worker record, the coordinator synthesizes the same
  errored/canceled record via a **coordinator-owned fallback
  `NativeMetricsObserver`** that reuses the retained `MeasuredTurnContext`, so
  `into_record` reproduces the HEAD shape (errored/canceled flags, `admit_ns`,
  `start_ns=issued_ns`, `end_ns=now`, empty token arrivals). This keeps
  `ErrorRequestCount`, cancel counts, and the errored/canceled row fields
  byte-identical to HEAD. The uuid-join tolerates an identity served either by a
  drained worker record **or** by a fallback record — it never aborts on a
  missing lookup for an identity that failed pre-worker. The
  `records.len() == identities.len()` assertion is re-expressed as "every
  identity has exactly one record from worker-drain **or** fallback," not a
  positional count.
- `CapturePhaseProcessor::process` (3244–3254): the live-sink
  `snapshot(credit)` (3250) reads a per-request record back from the coordinator
  observer via `NativeMetricsObserver::snapshot_record` (metrics.rs:307–314), a
  **non-consuming** clone. Under worker-local observers the coordinator no longer
  owns that observer, so this call has nothing to read.
  **Do NOT use `drain_terminal_record` (metrics.rs:322–328) for this.**
  `drain_terminal_record` calls `state.take_terminal(uuid)` — it **removes** the
  request from the worker observer's state, so the record would never appear in
  that worker's `finish_with_records()` and the end-of-run aggregate would
  **undercount** every live-emitted request (violating the "no reported metric
  change" invariant). The live path must stay **non-consuming**: when — and only
  when — a `live_sink` is attached, the worker returns a **cloned**
  `snapshot_record`-equivalent `RecordIngest` for that request (O(requests),
  produced by the worker's own observer after its terminal callback) alongside
  the outcome in `WorkerReply`; the coordinator's live sink emits from that clone
  while the authoritative record stays in the worker accumulator for the final
  merge. This means `WorkerReply` conditionally carries `{ result, ttft,
  live_record: Option<RecordIngest> }` when a live sink is present —
  reconciling the §3.1 "only `{ result, ttft }`" claim, which holds **only** for
  the no-live-sink case. This path is **built**: a `--live`/streaming-results run
  (a) emits one live record per request and (b) produces an end-of-run aggregate
  whose request/token counts are unchanged from a non-live run over the same
  request set.

### 3.4 Library scheduled path (non-runner consumers)

`rust/runtime/src/scheduled.rs`:

- `ScheduledRuntime` builds one `ObserverTee` of `CollectorObserver` +
  `NativeMetricsObserver` (lines 482–486) and passes `runtime.observer`
  into `dispatcher.dispatch_turn` (lines 925/933). For worker-local
  accumulation the observer must be **owned by the backend workers**, not the
  runtime; `dispatch_turn`'s `observer` parameter becomes advisory /
  coordinator-only (arrival, `on_terminal` fallbacks at 955/983).
- `finish_at` (1103–1136) already drains via `collector.take()` +
  `native_metrics.take_finalizer_at(end_ns)` and optionally rayon-joins the two
  reductions. Under worker-local it instead **merges the per-worker drained
  accumulators** then summarizes.

`rust/runtime/src/phase_runtime.rs`:

- `run_scheduled_phases_with_aggregate` (592–610) and `_deferred` (617–643)
  attach a whole-run `CollectorObserver` as an `additional_observer`. Per-phase
  observer construction is at 760–778. These aggregate collectors are the
  library compat surface — see the A2 dependency in §5.

### 3.5 Merge primitives (already built — reused as-is)

- `MetricsAccumulator::merge` — `rust/runtime/src/metrics_core/accumulator.rs:485–514`.
- `MetricsMergeError` — same file, 54–79.
- `ColumnStore::append_store` — `rust/runtime/src/metrics_core/store.rs:569–656`
  (dense precondition asserted at 575–578).
- `NativeMetricsFinalizer` (Send, plain data) — `rust/runtime/src/metrics.rs:214–223`;
  `take_finalizer_at` 356–363; `finish`/`finish_with_records` 404–442.
- Reference merge call: `transport_bench.rs:385–395`.

No new metrics-crate primitive is required for the **runner** path — and,
per §2.1, none of the *merge* primitives above (`MetricsAccumulator::merge`,
`append_store`, `take_finalizer_at`, the `transport_bench.rs:385–395` merge call)
are used by it either. The runner reuses only `finish_with_records`
(per worker) plus its existing dispatch-order re-ingest (`execute.rs:1557-1560`).
The merge primitives are the mechanism for the **library** summarize-directly
path (§3.4).

---

## 4. Merge-correctness analysis

### 4.1 What `MetricsAccumulator::merge` covers (verified)

`merge` (accumulator.rs:485–514): rejects `ConfigMismatch` if `MetricsConfig`
differs, rejects `NetworkRttMismatch`, rejects `InjectedScalarConflict`, then
folds `network_rtt`, extends `injected_scalars`, and calls
`store.append_store(&other.store)`.

`append_store` (store.rs:569–656) merges, per appended row:

- scalar row columns: `start_ns`, `end_ns`, `generation_start_ns`,
  `observed_output_sequence_length`, `session_nums`, `turn_indices`,
  `errored`, `canceled`;
- **re-interned categoricals**: `phase`, `correlation`, `dimensions`, `worker`,
  `conversation` (so **per-model / per-endpoint series AND per-worker series
  survive the merge** — dimension and worker codes are re-interned, not lost);
- numeric columns, index-aligned with absence preserved;
- **ragged token-arrival / `InterChunkLatency` series** via `append_shifted`
  (record indices shifted by `row_offset` exactly once).

Pinned by the existing test
`worker_stores_merge_with_numeric_categorical_and_ragged_alignment`
(store.rs:1437–1471), which asserts merged worker masks, turn indices, numeric
column lengths, and ICL replay values/offsets after `append_store`.

### 4.2 What needs care — the dense precondition and record ordering

`append_store` asserts every appended store is **dense** (no holes):
`other.occupied.iter().all(|occupied| *occupied)` (store.rs:576–578).

Consequence for the design: **a worker's own store slot must NOT be the global
`request_index`.** Today `ScheduledRuntime` assigns `request_index =
recorder.begin(...)` (scheduled.rs:868, 878) and the observer places records at
that absolute slot (`on_arrival`, metrics.rs:559). A global slot on a per-worker
store creates holes (a worker owns a sparse subset of global indices) →
`append_store` panics (scratch-proven, §6). So each worker's **internal store
slot** stays **dense-local** (arrival order → `push_record`).

This is decoupled from the emitted record's `request_index` **field**, which is
the **globally-unique dense dispatch ordinal** stamped at the coordinator uuid-join
(§4.2 "Required fix" / decoupling) — that global field, not the worker slot, is
what the runner re-ingest keys on. The runner's `RunCapture::finish` positional
zip must become a uuid join (§3.3) precisely so it can pair each dense-local worker
record with its identity and stamp the global ordinal.

The runner path is already well-positioned: `RunCapture::begin` (execute.rs:3090)
does **not** set `request_index`, so its observer already uses dense
arrival-order slots — moving that same dense behavior worker-local is natural.

**HARD PANIC on the runner re-ingest path — the extracted records collide, and
neither PR1 nor the worker's internal `push_record` fixes it.** The dense-store
analysis above covers only the worker's *own* store and the `append_store`/merge
path. But the runner does not merge stores — it **re-ingests extracted
`RecordIngest`s** into a fresh accumulator (`execute.rs:1557-1560`,
`accumulator.process_record`). And `into_record` **always** stamps a concrete
slot: `request_index = self.metadata.request_index.or(Some(ordinal as usize))`
(`metrics.rs:472`), where `ordinal` is the record's dense slot **within that
worker's observer**. So even though each worker uses `push_record` internally
(no coordinator global slot), the records it hands back carry
`request_index = Some(0), Some(1), …` **local to that worker**. With N workers,
`worker0.record[0]` and `worker1.record[0]` both re-ingest to row 0:
`process_record` routes `Some(row)` to
`insert_record_at_with_token_arrivals` (`accumulator.rs:452-455`), which
`assert!(!self.occupied[row], "request slot {row} was already populated")`
(`store.rs:553-556`) → **panic** on any multi-worker run with ≥2 requests split
across workers. This is a fail-closed abort the moment worker-local records exist,
independent of the uuid-join (which only fixes *report row order*, not the
re-ingest slot).

**Required fix (runner drain path) — `request_index` MUST be a globally-unique,
dense, monotonic dispatch index (DECIDED; not `None`/push).** Before the
`execute.rs:1558` re-ingest, the coordinator reassigns every merged record's
`request_index` to the record's **global dispatch ordinal** — the single
issuance counter's value for that dispatch identity (see "single global counter"
below), unique across all workers and dense `0..N-1` because every issued index
is filled by exactly one record (worker-produced OR the fallback accumulator,
§3.3/Risk 4 — the fallback is what guarantees no holes). Re-ingesting at that
index via `insert_record_at_with_token_arrivals` is then collision-free (unique),
hole-free (dense), and in **HEAD dispatch order** (monotonic) → runner float
byte-parity (§4.5) with no separate sort. The uuid-join in `RunCapture::finish`
(§3.3) is where the reassignment happens: it already resolves each record to its
dispatch identity, and the identity carries the global ordinal.

Two options are explicitly **rejected**:
- `request_index = None` → `push_record`: slots become **drain/concatenation
  order** (`[w0 block ∥ w1 block ∥ …]`), which is *not* HEAD's interleaved
  dispatch order, so the IEEE-754 fold reorders and float fields drift from HEAD.
  A global dispatch index is required precisely to restore HEAD ordering.
- Letting a **worker** place records at the global index in its **own** store:
  a worker owns a sparse subset of the global indices → holes in its local store
  → `append_store`/`insert_record_at` panic. Decouple the two indices instead
  (next paragraph).

**Decoupling (worker internal slot vs. record `request_index`).** `metadata
.request_index` is used for TWO things today and they must be split: (a) the
worker observer's internal store slot (`on_arrival`, metrics.rs:559
`get_or_insert(state.requests.len())`) must stay a **dense LOCAL** arrival-order
slot so the worker's own store has no holes; (b) the emitted record's
`request_index` (`into_record`, metrics.rs:472) must carry the **GLOBAL** dispatch
ordinal for the coordinator re-ingest. Simplest implementation: workers accumulate
dense-local (as `RunCapture` already does, execute.rs:3090 sets no
`request_index`), and the coordinator **stamps the global ordinal during the
uuid-join** (from the identity), so the worker never needs the global value
internally.

**Single global counter (issuance).** On the runner path the global dispatch
ordinal is the **`RunCapture::begin` push order** — a coordinator-owned counter
incremented in `dispatch_turn`'s synchronous prefix before backend dispatch, so
its order is independent of worker count and equal to HEAD's single-observer
arrival-slot order. (This is used in place of threading `recorder.begin`
(scheduled.rs:868, 878), which is the library path's issuance ordinal.) The
counter is a **single coordinator-owned source** assigned before the turn is
handed to a worker; it is never reset per worker. Keeping issuance/admission
coordinator-single-threaded (only *measurement* goes worker-local) is what makes
the ordinal both globally-unique/dense AND identical to HEAD's dispatch order —
the precondition for float byte-parity.

### 4.3 Why merge accumulators, not summaries

Timeslices, sweep-lines (concurrency/effective/active), duration-weighted
stats, per-model series, and derived scalars are **computed at
`export_results`** from the merged store (accumulator.rs:527–548, 1045–1104,
1106–1172), not stored incrementally. Merging raw stores then calling
`summarize()` once yields globally-correct timeslices and sweeps.
`AccumulatorSummary` has **no** merge and merging two summaries would be wrong
(percentiles and sweep curves do not combine). Therefore the cross-thread
hand-back is the raw `MetricsAccumulator` / `NativeMetricsFinalizer`, and the
single `summarize()` runs after merge.

### 4.4 Telemetry / injected scalars

GPU energy, server `/metrics`, and network-RTT are coordinator-side producers.
Inject them **once, post-merge** via `inject_scalar` / `set_network_rtt_ns`
(accumulator.rs:475–482) — never per worker (avoids `InjectedScalarConflict`).
`remove_unpartitioned_results` (accumulator.rs:659–667) already keeps injected
run-level scalars out of per-partition series.

### 4.5 Worker-order determinism — and the HEAD-vs-A1 reorder at the SAME worker count

> **SCOPE CORRECTION (Finding 6) — this entire §4.5 reorder hazard applies ONLY
> to the merge-summary mechanism (the library `ScheduledRuntime`/`phase_runtime`
> summarize-directly path, §3.4). It does NOT apply to the runner product path,
> which under the §2.1 correction is RECORDS-first: worker records are
> concatenated, uuid-joined into **dispatch order**, and re-ingested into one
> fresh accumulator in that order (`execute.rs:1557-1560`). The IEEE-754 fold on
> the runner path is therefore over the **same dispatch order HEAD folds in**, so
> the runner's float distribution fields (`avg`/`sum`/`std`/percentiles of
> `ttft`/`ttst`/`tpot`/`itl`/`e2e`/`otpu`) are BYTE-IDENTICAL to HEAD — at any
> worker count. The two-latency-profile / ULP-tolerance apparatus below is a
> guard for the merge-summary path, not an accepted drift for the runner. Treat
> any runner-path float delta vs a HEAD golden as a real regression, because a
> correct records-first re-ingest incurs none.**

The rest of this section describes the **merge-summary** path (accumulator merge
→ `summarize()` once), where the reorder is real:

Merged summation order is `worker0 ∥ worker1 ∥ … ∥ workerN`. For a fixed worker
count and the existing deterministic round-robin assignment
(`next_worker` rotation, turn_execution.rs:458–460) the order is stable
run-to-run, so A1's own aggregates are reproducible A1-to-A1. Changing the worker
count reorders float summation → sub-ULP differences in sums/percentiles.

**The load-bearing subtlety this spec previously understated:** merged
per-worker order is **not the same order as HEAD**, even at an identical worker
count. HEAD's single `RunCapture` observer places records in **global
arrival/dispatch order** (dense slots assigned coordinator-side as each turn is
issued, execute.rs:3275; metrics.rs:559). A1 places records in **per-worker
local order** and concatenates them worker-by-worker at merge
(`append_store`, store.rs:569–656). Distribution `avg`/`sum` are an
order-sensitive left-fold (`values.iter().sum()` over store rows in slot order,
accumulator.rs:917–935; `linear_distribution` sets `avg = running_sum / count`,
kernel.rs:84–89), and `std` folds squared deviations off that same
order-sensitive `avg` (kernel.rs:103–111). So on varied-latency data the
**float** distribution fields (`avg`/`sum`/`std` and the interpolated
percentiles of `ttft`/`ttst`/`tpot`/`itl`/`e2e`/`otpu`) computed by A1 will
differ from a **HEAD-captured golden** by sub-ULP-and-up rounding, at the same
worker count. This is the direct consequence of reordering an IEEE-754 fold; it
is not a bug in the merge, but it **contradicts a naive "byte-identical to HEAD"
reading of the plan's "no reported metric change" goal.**

**Consequence for the parity gate (mandatory):**

- The **integer aggregates** — `RequestCount`, `ErrorRequestCount`,
  `TotalInputSequenceLength`, `TotalOutputSequenceLength`, `GoodRequestCount`,
  request/token counts — are order-independent (scratch-verified, §6) and MUST
  stay **byte-identical** to HEAD.
- The **float distribution fields** cannot be asserted byte-identical against a
  HEAD golden. The Harness A golden for these fields must either (a) be captured
  from **A1 output** (not HEAD) with an explicit note that the sub-ULP delta from
  HEAD is an accepted, reorder-only consequence, or (b) be compared with a tight
  **ULP/relative tolerance** (e.g. within a few ULP for `avg`/`sum`, a small
  relative band for `std`/percentiles) rather than exact bytes. Do **not** flag
  the A1 float delta as a regression, and do **not** claim the float report is
  byte-identical to HEAD in production. A SimClock fixed-*equal*-latency workload
  (every request the same latency) is the exception: with identical summands the
  fold is order-invariant, so that specific golden **is** byte-exact and should be
  used to catch true logic regressions; varied-latency goldens use the tolerance
  path.

---

## 5. The `CollectorObserver` blocker and the A2 dependency

`loadgen-core::TraceCollector` (collector.rs:466–490) and its wrapper
`loadgen-core::CollectorObserver` (`loadgen-core/src/observer.rs`, struct at
line 21) produce the compat `TraceSimulationReport`. (`CollectorObserver` moved
into `loadgen-core` when the former `aiperf-core` crate was dissolved.) Two facts:

1. **No merge path exists.** `TraceCollector` is a `FxHashMap<Uuid, …>`
   (collector.rs:467) with `finish(self)` (748–836) consuming it; there is no
   `merge`. `CollectorObserver::take` (observer.rs:44) drains but does not
   combine. Its `finish` accumulation pass (`accumulate_requests`,
   928–1013) iterates `requests.values()` and is **order-independent** — so a
   merge is *trivially implementable* (extend the uuid-keyed map across workers;
   uuids are disjoint; the SLA / worker-second / gpus-per-worker config must
   match), but **it is not built today.**

2. **The runner product path does not use it.** `ConfiguredDispatcher` feeds
   only `RunCapture`'s `NativeMetricsObserver` and explicitly discards the
   `ScheduledRuntime` `CollectorObserver` (execute.rs:3271, comment 3276–3282).

**Dependency statement:**

- For the **runner product path**, the `CollectorObserver` blocker is **already
  absent** — A1 can proceed on the runner without A2. The runner ceiling is
  purely the single `RunCapture` `NativeMetricsObserver` + `BufferedObserver`
  replay, which this spec removes.
- For the **library `ScheduledRuntime` path and the `phase_runtime` aggregate
  collectors** (phase_runtime.rs:599, 628, 762), worker-local accumulation of
  the compat collector requires either:
  - **A2 removes `CollectorObserver`/`TraceSimulationReport` entirely** — then
    A1's compat blocker vanishes and only `NativeMetricsObserver` goes
    worker-local; **or**
  - if A1 must land before A2, add a `TraceCollector::merge(&mut self, other)`
    (uuid-map extend + config-equality guard) plus a parity test proving
    merged-then-`finish` equals single-collector `finish` for the same request
    set.

**Recommendation:** sequence A2 first (or concurrently), then A1 only relocates
`NativeMetricsObserver`. If A1 ships first, ship it **runner-only** (no compat
merge needed) and defer the library `ScheduledRuntime`/`phase_runtime`
relocation until A2 lands.

---

## 6. `!Send`/`Send` verification

The `loadgen-core::RequestObserver` trait has no `Send`/`Sync` supertrait
(each worker owns an `Rc`/`RefCell` observer graph — the thread-per-core
contract). `NativeMetricsObserver` is correctly `!Send`
(`Rc<dyn Clock>` + `RefCell`, metrics.rs:190–199) and **stays worker-local**.
Only the **drained** state crosses back to the coordinator join.

Verified `Send`:

- `MetricsAccumulator` — `ColumnStore` (Vecs/`FxHashMap` of plain data) +
  `MetricsConfig` + `Option<f64>` + `FxHashMap<MetricTag, MetricValue>`. No
  `Rc`/`RefCell`.
- `AccumulatorSummary` — `BTreeMap`/`Vec` of plain result data.
- `NativeMetricsFinalizer` — holds `ObserverState`
  (`Vec<Option<PendingEntry>>` + `FxHashMap<Uuid,…>`, all plain data) +
  `MetricsAccumulator`. Its doc (metrics.rs:214–219) already states it "contains
  no clock, `Rc`, observer, or runtime handle, so an offline runtime may move it
  to a worker" — the crate was designed for exactly this hand-off.

### Scratch proof — `~/tmp/a1-spec/`

A throwaway crate (path-dep on `aiperf`, whose `metrics_core` module holds the
former `aiperf-metrics` primitives; no repo mutation) proves the merge primitives
compile and behave:

- `src/main.rs`:
  - `assert_send::<MetricsAccumulator>()` and
    `assert_send::<AccumulatorSummary>()` **compile** (static `Send` proof).
  - Builds three `MetricsAccumulator`s on **three separate OS threads**
    (`thread::spawn`), moves each back across `join()`, and merges into a
    coordinator accumulator → `record_count() == 6`,
    `summarize().finite_value(RequestCount) == 6.0`.
  - Reversed merge order yields identical `RequestCount` and
    `TotalOutputSequenceLength` (order-independent aggregates).
  - Output: `A1 VERIFY OK: Send proven; cross-thread merge -> 6 requests;
    order-independent aggregates match`.
- `src/bin/sparse.rs`: a worker accumulator built with a **global sparse**
  `request_index = Some(5)` (holes at 0..4) is **rejected** by `merge` →
  `SPARSE MERGE REJECTED as expected: workers must be dense before append`.
  This pins the §4.2 dense-local-rows design constraint.

Both binaries build and run clean under
`CARGO_TARGET_DIR=~/tmp/a1-spec/target cargo run`.

In-repo corroboration: `transport_bench.rs:385–395` performs exactly this
cross-thread `MetricsAccumulator::merge` in the production graph bench, and
`store.rs:1437–1471` pins ragged+categorical+numeric merge alignment.

---

## 7. Risks

1. **(BIGGEST — resolved) Record ordering / determinism vs. `RunCapture::finish`.**
   Drained records are per-worker-concatenated, not global dispatch order, so the
   old positional `zip(identities)` + `correlation_id` assertion is replaced by a
   uuid-keyed join that reassigns `request_index` to the `RunCapture::begin` push
   order before re-ingest. This is the single load-bearing refactor; the parity
   test (§8) proves it, and getting it wrong would fail every runner scheduled run
   closed.
2. **Phase labeling timing (resolved).** The worker records only transport
   facts; `phase` / `session_num` / `has_credit_timestamp` are stored per uuid by
   `CapturePhaseProcessor` and applied to the drained record in
   `RunCapture::finish`, which also sets `admit_ns` from the credit-issued time.
   This keeps the credit-latency time base identical to HEAD without forwarding
   phase at issue; a missed application would silently mislabel warmup rows into
   profiling, so the parity gate pins phase-tagged counts.
3. **Live results sink (CONSUMING-drain trap).** `CapturePhaseProcessor`
   (execute.rs:3244–3254) reads per-request snapshots from the coordinator
   observer via the **non-consuming** `snapshot_record` (metrics.rs:307–314).
   With a `live_sink` attached the worker must return a **cloned, non-consuming**
   `RecordIngest` per request (O(requests)) in `WorkerReply`; without a live sink,
   nothing crosses per request. **Do not use `drain_terminal_record`
   (metrics.rs:322–328)** — it calls `take_terminal` and removes the request from
   the worker accumulator, so each live-emitted request would be dropped from the
   end-of-run merge and the aggregate would undercount. This is built and tested
   (§3.3): a `--live` run's end-of-run counts equal the non-live run's.
4. **Pre-worker failures / cancellations vs the identity↔record count invariant
   (resolved).** A `WorkerCommand` send failure (turn_execution.rs:476),
   a worker dropping the command (turn_execution.rs:501–503), or a
   `PlacementCancellation` (turn_execution.rs:465–466,681–692) can fail a request
   **before any worker observer registers it** — so the drained records are fewer
   than dispatched identities. The coordinator synthesizes the identical
   errored/canceled record via a **coordinator-owned fallback
   `NativeMetricsObserver`** reusing the retained `MeasuredTurnContext` (see §3.3),
   and the uuid-join accepts an identity served by a drained worker record **or**
   a fallback record — never aborting on a missing lookup. Unit coverage pins the
   pre-worker fallback so the errored/canceled fields (`errored`, `canceled`,
   `ErrorRequestCount`, admit/start/end timing) stay byte-identical to HEAD and
   the run does not abort fail-closed.
5. **`SlotPool` stays coordinator-side (non-goal to move it).** Admission /
   session / prefill slots gate issuance on the coordinator
   (scheduled.rs issuance path). Only measurement goes worker-local. The
   `on_admit` timestamp is known at issue and is forwarded; do **not** make the
   `SlotPool` per-worker — that would change admission semantics.
6. **Config equality across workers.** Every worker accumulator must be built
   from the identical `MetricsConfig` (SLOs, `slice_duration_ns`, thresholds,
   `use_server_token_count`) or `merge` returns `ConfigMismatch`
   (accumulator.rs:486–488). Enforce a single resolved config broadcast to
   workers.
7. **Compat collector (library path).** See §5 — gated on A2 or a new
   `TraceCollector::merge`.
8. **Float summation order differs from HEAD — ONLY on the merge-summary
   (library) path, NOT the runner (§4.5 scope correction / Finding 6).** For the
   **merge-summary** path, merged per-worker order (`worker0 ∥ worker1 ∥ …`) is
   not the global dispatch order HEAD's single observer folds in, so
   avg/sum/std/percentiles of the latency distributions differ from a HEAD golden
   by (at least) sub-ULP amounts even at a pinned worker count; those fields use
   the ULP/tolerance band. **For the runner product path this risk does not
   exist:** records-first re-ingest folds in dispatch order (`execute.rs:1557-1560`),
   so the runner's float fields are byte-identical to HEAD and a runner-path float
   delta is a real regression, not accepted drift. See §4.5.

---

## 8. Implementation status

Runner path — **built**:

1. Metrics-side groundwork: `MetricsConfig` is broadcast to workers via
   `configure_measurement`; `HttpTurnExecutionBackend::drain_records(end_ns) ->
   Vec<(Uuid, RecordIngest)>` (per §2.1/§3.3 — the runner drain returns a flat vec
   of uuid-paired **records**, not accumulators). No `drain_accumulators` seam is
   used on the runner path.
2. Runner relocation: worker-local `NativeMetricsObserver` in
   `turn_execution.rs` + `grpc_turn_execution.rs` behind the measured seam; `RunCapture` uuid-joins in `finish` (keyed on the drain-provided `Uuid`,
   reassigning `request_index` to the `begin` push order before re-ingest — §3.3,
   §4.2) and patches phase/session/`admit_ns` at finish. The existing
   dispatch-order re-ingest (`execute.rs:1557-1560`) is preserved. No A2
   dependency on this path.
3. `ObserverEvent` / `BufferedObserver` / `execute_turn(observer)` have been
   **removed** — the former agentic/external-evaluator verticals that consumed the
   buffered replay are gone, and the surviving static-accuracy consumer dispatches
   through the same measured seam. Every `WorkerCommand` carries a non-optional
   `MeasuredTurnContext`; there is no buffered fallback branch.

Library path — **gated on A2** (planned):

4. The library `ScheduledRuntime` / `phase_runtime` accumulation is unchanged and
   remains single-observer until A2 removes `CollectorObserver` /
   `TraceSimulationReport` (or a `TraceCollector::merge` is added); only then does
   its `NativeMetricsObserver` go worker-local via the merge-and-summarize
   mechanism (§3.4, §4.3).

Proof (built):

5. `runner/tests/worker_local_accumulation_parity.rs` drives the real
   runner subprocess at `worker_count` 1 vs 4 over a fixed mock and asserts the
   count/token report fields are **byte-identical** (rate/throughput excluded —
   the faster four-worker run is the expected win). Unit tests cover the
   global-index reassignment, worker-split byte-parity, the pre-worker fallback,
   and the non-consuming live snapshot; `grpc_v2_stdio` (`worker_count: 2`) proves
   the gRPC twin.

---

## Appendix — verification artifacts

- Scratch project: `~/tmp/a1-spec/` (`src/main.rs`, `src/bin/sparse.rs`) —
  Send static asserts, 3-thread cross-thread merge, order-independence, and
  dense-precondition rejection. Runs clean.
- Primary code citations (the `turn_execution.rs` / `grpc_turn_execution.rs`
  buffered-path ranges are pre-change HEAD — that code is now removed, replaced by
  the measured seam): `turn_execution.rs` 113–223 / 415–521 / 647–693;
  `grpc_turn_execution.rs` 105–210 / 376–378; `execute.rs` 3067–3111 /
  3200–3234 / 3244–3287; `scheduled.rs` 420–539 / 868–1017 / 1089–1136;
  `metrics.rs` 190–443 / 546–736; `accumulator.rs` 34–79 / 485–514;
  `store.rs` 569–656 / 1437–1471; `collector.rs` 466–490 / 748–836 / 928–1013;
  `loadgen-core/src/observer.rs` (`CollectorObserver`, struct at line 21);
  `transport_bench.rs` 385–395; `graph_execution.rs`
  738 / 787–805; `phase_runtime.rs` 586–643 / 760–778.
