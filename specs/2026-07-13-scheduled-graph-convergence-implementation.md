<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scheduled ↔ graph convergence — implementation

**Date:** 2026-07-13
**Status:** built (this change). Implements the one behavioral decision the production audit deferred, plus the P1 rename pass.
**Scope:** Turn three sibling design docs into code:
- `2026-07-13-scheduled-graph-production-convergence.md` (audit) — *what is shared vs. different*. Its one genuine divergence is **failure behavior**.
- `2026-07-13-greenfield-execution-vocabulary.md` (north-star) — names that divergence **`OnFailure { Continue, Abort }`** (an enum, not a trait with two impls).
- `2026-07-13-p1-generic-execution-substrate-names.md` (rename pass) — the concrete in-tree renames that make the already-shared substrate *visibly* shared.

This spec records the design of the **`OnFailure` convergence seam** (the new decision) and the plan that carries the P1 renames and the audit's incidental cleanups into the tree, staged so each stage keeps the suite green.

## 1. The decision: one configurable failure seam, defaults preserve behavior

The audit (§"Genuinely different" #1) found the sole real divergence: graph online is **fail-fast** (`AbortTraceNodeFailurePolicy` + `FailFastRunFailurePolicy` + coordinator `ensure!(failed == 0)`), scheduled online is **resilient** (transport failures recorded, run continues; only issuance/materialization errors abort). This is a **product choice, not a necessity** — so it becomes a config knob, not two hard-coded worlds.

Greenfield names it **`OnFailure { Continue, Abort }`**. We adopt that exactly:

```rust
// rust/runtime/src/failure.rs  (new, crate-root — shared, not graph-namespaced)
pub enum OnFailure { Continue, Abort }
```

- **`Continue`** — a failed `Request`/node is recorded and the run proceeds (today's *scheduled* behavior; graph maps it to `ResilientNodeFailurePolicy` + `ContinueRunFailurePolicy`, which already exist in `graph::policy` and were previously never wired by the product).
- **`Abort`** — the first non-cancellation failure fails the whole benchmark (today's *graph* behavior; graph maps it to `AbortTraceNodeFailurePolicy` + `FailFastRunFailurePolicy`; scheduled latches `RequestRateState::fail` on a `Failed` terminal and `bail!`s at teardown).

**Cancellation is never a failure** on either path (matches `FailFastRunFailurePolicy::on_trace_result`, which ignores `TraceError::Cancelled`, and the scheduled cancel arm, which is resilient regardless).

### Default preservation (no regression)

The wire field is **optional**; when absent each path applies its historical default, so an unmodified request behaves byte-identically:

| Path | `failure_policy` absent → | `Continue` | `Abort` |
|---|---|---|---|
| scheduled | `Continue` (resilient) | resilient (today) | fail-fast on `Failed` terminal (new) |
| graph | `Abort` (fail-fast) | resilient (new; uses the dormant library defaults) | fail-fast (today) |

`OnFailure` is a plain `Copy` enum threaded by value — no `Rc<dyn>` on the hot path. The graph run/node **traits** stay (they carry the async `wait_blocked` admission gate and the abort-latch semantics the executor needs); `OnFailure` merely *selects which impl* is installed at the two wiring points, replacing the hard-coded picks. This keeps the extension seam (a third failure discipline is still a new impl) while removing the "graph is always fail-fast" hard-coding.

## 2. Wiring (the two seams the audit located)

Graph fail-fast is wired at exactly two production points; both become `OnFailure`-selected:

1. **Run-level**, `rust/runtime/src/engine/graph_phase_runtime.rs:1191` — `.with_run_failure(Rc::new(FailFastRunFailurePolicy::default()))` → select `FailFastRunFailurePolicy` (Abort) vs `ContinueRunFailurePolicy` (Continue).
2. **Node-level**, `rust/runtime/src/engine/graph_execution.rs:642` — `local.with_node_failure(Rc::new(AbortTraceNodeFailurePolicy))` → select `AbortTraceNodeFailurePolicy` (Abort) vs `ResilientNodeFailurePolicy` (Continue). Threaded via a new `on_failure: OnFailure` field on `RunnerGraphBackendFactoryConfig` (`graph_execution.rs:470`), fed from `NativeRunSpec` next to `metrics` (`execute.rs:1120`).

The coordinator `ensure!(phased.workload.failed == 0, …)` at `rust/runtime/src/engine/execute.rs:1202` stays for `Abort`; under `Continue` the graph run reports failed traces without erroring, so this assertion is relaxed to *only* fire under `Abort` (under `Continue` the failed count is expected and recorded, not an error).

Scheduled resilient/fail-fast is wired in `rust/runtime/src/request_rate.rs`: `RequestRateWorkload` gains an `on_failure: OnFailure` field; the terminal completion hook (`request_rate.rs:442`) latches `state.fail(...)` when `outcome.terminal` is `Failed` (not `Cancelled`) and `on_failure == Abort`. `Continue` leaves the existing resilient path (`scheduled.rs:980-1004`) untouched.

**Scope limit (scheduled):** fail-fast is wired only into `RequestRateWorkload` (the `Concurrency`/`Poisson`/`Gamma`/`Constant` arm). The specialized `user_centric` and `fixed_schedule` scheduled workloads do not yet honor `abort`; rather than silently staying resilient, the runner emits a `tracing::warn!` when `abort` is configured alongside those phase types (`execute.rs`). Extending fail-fast to them is a follow-up; the default (`Continue`) is unaffected.

### Config surface

- Wire form on `BenchmarkConfigWireV2` (`rust/runtime/src/engine/protocol_v2.rs:178`): `#[serde(default)] pub failure_policy: Option<String>` beside the `slos`/`goodput` policy siblings. Lowered into the shared workload-config JSON in `BenchmarkRunWireV2::into_authored` (`protocol_v2.rs:275`) so **both** `ScheduledWorkloadConfigV2` and `GraphWorkloadConfigV2` (`registry.rs:838,862`) decode the same `failure_policy: Option<OnFailure>` field.
- `OnFailure` decodes from `"continue"` / `"abort"` (serde rename, lowercase), precedent = `ArrivalPattern` (`timing/intervals.rs:30`) and the optional-policy shape of `CancellationConfig` (`ancillary.rs:29`).
- Python: `src/aiperf/orchestrator/rust_wire.py::dump_benchmark_run` emits `cfg["failure_policy"]` only when set; `src/aiperf/config/config.py` gains an optional `failure_policy` field. Absent by default → historical behavior.

## 3. P1 renames (carried in this change)

Implements the **rename** half of `2026-07-13-p1-generic-execution-substrate-names.md`; flips that spec's status to **built (renames)**. Pure rename, **no behavior change**, suite stays green unmodified (692 aiperf lib + runner stdio/graph/parity). Landed after the concurrent bodyplan work committed `http.rs`, using `\b`-anchored renames so compound names (`LocalGraphTraceExecutionBackend`) stayed untouched.

- **Group A** (shared DTOs): `PreparedHttpTurn`→`PreparedTurn`, `MeasuredTurnContext`→`MeasuredContext`, `MeasuredTurnOutcome`→`MeasuredOutcome`, `HttpTurnDispatchResult`→`DispatchResult`.
- **Group B** (placement seams, kept as two traits): `HttpTurnExecutionBackend`→`RequestExecutor`, `GraphTraceExecutionBackend`→`TracePlacement`, factories + thread-per-core impls renamed in parallel.
- **Dispatch/execute methods** renamed to the level-generic names: `dispatch_measured`, `dispatch_collect[_streaming]`, `dispatch_collect_with_observer` (private), `execute_measured[_streaming]`.

**Deferred (structural, not renames):** the method-count *fold* (collapsing `dispatch_collect_with_observer` into `dispatch_collect_streaming`) and the `inference_dimensions(&TurnToSend)`→`(&PreparedTurn)` signature decoupling — both are hot-dispatch-path signature refactors, deferred to keep this change rename-only. See the P1 spec's 2026-07-14 addendum. The two placement traits stay **two** traits — merging them is the deferred v2 structural work.

## 4. Incidental cleanups (audit §"Incidental cleanups surfaced")

- **Discarded coordinator observer — done (2026-07-14).** Verified that on the runner path the coordinator's `CollectorObserver`/`NativeMetricsObserver` retention is never read — the native-v2 report is rebuilt from drained per-worker records, and the only coordinator output consumed (`phased.reports[].issued_offset_ns`) comes from the runtime's `DetailedSchedule`, gated separately by timing-record capture. The online scheduled plan now sets `with_performance_record_capture(false)` + `with_native_metric_record_dimensions(false)`, dropping the discarded per-request record accumulation. Metrics confirmed unchanged (`worker_local_accumulation_parity` + online/http stdio suites green; e2e still emits full TTFT/ITL/latency). Hot-path allocation win, no correctness change.
- **Unwired library policies kept as documented seams, not deleted.** `DurationGraphStop` (a duration `GraphStopPolicy`) and `ScheduledGraphArrival` (an authored-offset `GraphArrivalPolicy`) are unwired by the product but are legitimate second implementations of live extension seams. Per the extensibility discipline ("always design ahead… note the extension you are leaving open"), deleting them would leave each seam with only its trivial default and read as vestigial — the opposite of forward-thinking. They gain a `///` note marking them as available-but-unwired so a future reader does not mistake them for accidental dead code. (Revised from an earlier "remove them" plan.)
- **`GraphStopPolicy` kept as a seam** (trait + `UnlimitedGraphStop`), not deleted: it is a live extension point, just always the default today.
- **Streaming dispatch is intentionally *not* switched.** The audit lists scheduled's non-streaming dispatch as a cleanup; it is **rejected with a reason**: forwarding live per-token frames to the coordinator reintroduces exactly the cross-thread per-token traffic the thread-per-core design forbids (`CLAUDE.md`: "never contend a shared collector lock per token"). SSE is parsed in-worker; only the terminal outcome crosses threads. This is the "write down why they're irreducibly different" answer for streaming.

## 5. Staging (each stage: build + suite green + graham-code-review + fix confirmed)

1. **OnFailure convergence** — `failure.rs`, both wiring points, config surface (Rust + Python), cleanups. First because it barely touches `http.rs` (minimizes collision with in-flight bodyplan/segment work).
2. **P1 Group A** — shared DTO renames + `TransportSink` dispatch consolidation. Re-sync `http.rs` against the tree before editing.
3. **P1 Group B** — placement seam renames.
4. **End-to-end proof** — `aiperf` CLI against the in-repo mock server, both scheduled and graph, plus the `OnFailure` toggle; correctness shown by **jsonl timing inspection** (arrival/TTFT/terminal ordering, and that `Abort` fails the run on an injected failure while `Continue` records-and-continues).

## 6. Parity & testing

- P1 stages: existing graph byte-exact parity, scheduled dispatch, and sim/online integration tests stay green **unmodified** — that is the correctness argument (no new tests for the rename).
- OnFailure: unit coverage that (a) absent field reproduces today's per-path behavior, (b) `Abort` on scheduled latches on a `Failed` terminal but not on `Cancelled`, (c) `Continue` on graph records failed traces without a run-level error. Plus the Stage-4 e2e jsonl proof.

## 7. Related

- `2026-07-13-scheduled-graph-production-convergence.md` — the audit this implements.
- `2026-07-13-greenfield-execution-vocabulary.md` — the `OnFailure` / substrate vocabulary this adopts.
- `2026-07-13-p1-generic-execution-substrate-names.md` — the rename pass this carries (status → built).
- `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5 — the deferred structural merge of the two placement seams (still deferred).
