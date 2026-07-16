<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: request-rate multi-turn workload — the single-loop credit issuer, faithfully

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built + implementation addendum — realizes the `request-rate | chain` row of the unified-graph-runtime spec
**Grounding:** end-to-end line-by-line read of the Python credit/timing subsystem —
`src/aiperf/timing/strategies/request_rate.py`, `credit/issuer.py`, `credit/structs.py`,
`timing/phase/stop_conditions.py`, `timing/concurrency.py`, `timing/conversation_source.py`,
`timing/intervals.py`, `timing/phase/lifecycle.py`, `common/loop_scheduler.py`,
`credit/callback_handler.py`, `timing/phase/credit_counter.py`, `timing/phase/progress_tracker.py`.
Companion: the unified-graph-runtime design (umbrella), the superseded scheduling-policy
sketch (same policy, earlier vocabulary), the dataset-segment-seam design (turn
materialization). Throughput framing from the dispatcher microbench in
`~/.claude/benchmark-findings/rust-singular-dispatcher-vs-worksteal-credit-throughput.md`.

---

## 0. What this is (and the correction it encodes)

A **request-rate multi-turn** run is the canonical policy-driven workload: dispatch
turns at a target rate against an LLM server, across multi-turn conversations, on
the `{clock}` + `{transport}` seam so the identical code runs ONLINE-real,
ONLINE-mock, and (via `SimClock`) OFFLINE.

**The semantics are not what the name suggests, and this spec exists to nail them
from source, not priors.** `--request-rate` does **not** pace conversation arrivals.
It paces **turns** — one credit per rate interval — and a conversation's turns are
spread across ticks, interleaved with every other conversation's, with **continuation
turns having priority over new-session starts**. This was verified in
`request_rate.py::execute_phase`; do not re-derive it from intuition.

---

## 1. The mechanism (ground truth, cited)

A **single-threaded credit issuer** loop. One credit = "permission to send one
request (turn)" (`structs.py:26`). Per rate interval, issue **exactly one** credit,
by priority (`request_rate.py:148-211`):

```
next_target = start + intervals.next_interval()          # absolute schedule
next_new    = conv_source.next().build_first_turn()      # cached (avoids wasting a sampler draw)
loop:
    pace to next_target (if behind: next_target = now — re-anchor, NO burst)   # :151-163
    next_target += intervals.next_interval()             # draw BEFORE issue (:165-167)
    if continuation_queue non-empty:                     # P1 — has session slot already
        issue_credit(continuation_queue.pop())           # :171-176  (blocking prefill acquire)
    elif stop.can_start_new_session():                   # P2 — needs a session slot
        match try_issue_credit(next_new):                # :181-199  (non-blocking)
            Issued  => next_new = conv_source.next().build_first_turn()
            Stop    => return
            NoSlot  => yield                              # retry next tick
    elif not stop.can_send_any_turn():                   # P3 — done
        return
    else:
        yield                                            # session-limited; await continuations
```

On worker completion (`handle_credit_return`, `:213-252`): if not the final turn,
build turn *k+1* (`TurnToSend.from_previous_credit`), and honor **think-time** —
`delay_ms` defers the *queue insertion* via `LoopScheduler.schedule_later`
(`:246-250`), NOT the rate loop. DAG children (`agent_depth > 0`) are dispatched
directly, bypassing the queue (`:235-242`).

### 1.1 Two concurrency dimensions (`concurrency.py`, release sites `callback_handler.py`)

- **Session slot** — one per conversation. Acquired on **turn-0 only**
  (`issuer.py:131`); released on the **root final turn** (`is_final_turn &&
  agent_depth==0`, `callback_handler.py:436`) plus in-flight cleanup at phase end.
  Caps concurrent conversations. New-session starts gate on it (`try_issue_credit` →
  `NoSlot`).
- **Prefill slot** — one per request. Acquired on **every turn** (`issuer.py:140`);
  released on **TTFT** (`callback_handler.on_first_token:485`) or on a return that
  never got a first token (error/cancel, `:454`). Caps concurrent prefill (the
  GPU-heavy phase).
- Each is a `DynamicConcurrencyLimit` = semaphore **+ debt tracking** for graceful
  ramp-down (`set_limit` cancels debt then adds, or drains + tracks debt),
  layered global+per-phase. **This is the debt-drain `SlotPool` already ported into
  `aiperf_runtime::timing`.** DAG children inherit the parent's session slot (no acquire).

### 1.2 Stop conditions (`stop_conditions.py`) — already mirrored in Rust

A list, first-reached-wins, each self-activating via `should_use(config)`:
Lifecycle (cancel / sending-complete), RequestCount (`requests_sent < N`),
SessionCount (`sent_sessions < N` **or** `root_requests_sent < total_session_turns`),
Duration (`time_left > 0`). `can_send_any_turn` = all pass; `can_start_new_session`
= that **plus** the session quota; `can_send_dag_child_turn` excludes SessionCount.
My `aiperf_runtime::timing::StopChecker` already carries this shape (`RunState` has
`root_requests_sent` / `total_session_turns`).

### 1.3 Numbering + counters (`credit_counter.py`) — atomic by single-loop serialization

`increment_sent(turn) -> (credit_index, is_final_credit)`: `credit_index =
requests_sent` (pre-increment — the monotonic id); on turn-0 bump `sent_sessions`
and `total_session_turns += num_turns`; `is_final_credit` = request-count cap **OR**
(`sent_sessions>=N && root_requests_sent>=total_session_turns`). DAG children bump
`requests_sent` only. All counters are lock-free because there is **no `await`
between read and write** — the single-loop property again.

---

## 2. Two planes — why this is a single-loop dispatcher (and where it fans out)

The dispatcher microbench (see the benchmark-finding) settles the deployment:

- **Control plane** — rate-pace + mint credit + acquire session/prefill slot +
  stop-check + number. One thread: **6.5–20 M/s**. **Never the bottleneck** for a
  rate-bound run (targets are 10k–1M req/s, sub-max by construction).
- **Data plane** — HTTP: TLS + SSE-parse-per-token + serde. CPU-heavy (~tens of µs
  per streaming request), so one core does only ~tens-of-k real req/s. **This must
  fan across cores** — that is the parallelism, and the real throughput limiter.

Consequence: the credit issuer is a **single-loop control plane**; the transport
sink is the **fanned-out data plane**. When HTTP is fanned to worker threads, the
control→data handoff caps ~1.7 M/s (a single dispatcher thread) — still ≫ any policy
rate, so the issuer stays off the critical path. Do **not** quote the 6.5/20 M/s
control-plane figure as achievable HTTP throughput; the operative fan-out number is
the handoff.

The `{clock}` seam picks the fan-out substrate for free:
- **ONLINE** (`RealClock`) — issuer on one `LocalSet`; dispatch `spawn_local`'d as
  async tasks (many concurrent requests multiplex on the loop). High-rate runs that
  exceed one core's HTTP CPU add worker threads for the data plane (control stays
  single-loop).
- **OFFLINE** (`SimClock`) — issuer + engine on one loop under `drive_sim`; think-time
  and inter-arrival `sleep`s are virtual; the run is deterministic. Single-owner-of-
  time is *mandatory* here, which is exactly the single-loop shape.

Work-steal (the `transport_bench` model) is **not** used for request-rate — it is the
no-policy max-throughput escape hatch only.

---

## 3. Mapping onto the crates

| Concern | Primitive | Crate | Status |
|---|---|---|---|
| Inter-arrival (Poisson/Gamma/Const/Burst) + `set_rate` | `IntervalGenerator` | `aiperf_runtime::timing` | **built** |
| Session + prefill caps (debt-drain, `set_limit`) | `SlotPool` / `ConcurrencyManager` | `aiperf_runtime::timing` | **built** |
| Stop bounds (count/session/duration/lifecycle) | `StopChecker` / `RunState` | `aiperf_runtime::timing` | **built** |
| Absolute-schedule pacer + catch-up | arrival loop | `aiperf_runtime::run` (`run_paced`) | **built** (single-turn only) |
| Pacing / think-time sleeps | `Clock::sleep` | `aiperf_runtime::clock` | **built** |
| Turn prompt = prior replies spliced | `SegmentStore` + `materialize` | `aiperf_runtime::dataset` | **built** |
| Dispatch turn + record TTFT/ITL (TTFT releases prefill) | `TurnDispatcher` + scheduled lifecycle hooks | `aiperf` / `aiperf_runtime::transport_http` | **built** |
| **Continuation queue + two-source issue loop** | `RequestRateWorkload` | `aiperf` | **built** |
| **Conversation source over the segment pool** | `ConversationSource` trait | `aiperf` / `aiperf_runtime::dataset` | **built** |
| **Prefill-release-on-TTFT wiring** | first-token lifecycle hook → `SlotGuard::drop` | `aiperf` | **built** |

The implementation addendum below records the completed continuation queue,
priority issue loop, TTFT release edge, and dataset/CLI wiring.

### 3.1 The new seams (every extension point a trait)

- **`ConversationSource`** — `fn next(&mut self) -> SampledSession` (sample a template
  + mint an `x_correlation_id`) and `fn next_turn_meta(&self, corr, turn_index) ->
  TurnMeta` (think-time + turn count). Impls: dataset-backed, synthetic. Turn prompts
  are materialized from the `SegmentStore` (splice reply *k* into turn *k+1*), never
  re-serialized.
- **`CreditIssuer`** — `issue(turn)` (blocking session[first]+prefill acquire) /
  `try_issue(turn) -> Issued|Stop|NoSlot` (non-blocking), numbering via the counter,
  dispatch through the `RequestSink`. Owns the two `SlotPool`s + `StopChecker`.
- **Continuation queue** — an `Rc<RefCell<VecDeque<TurnToSend>>>` on the loop;
  `handle_return` pushes turn *k+1* (after a `Clock::sleep(delay)` for think-time);
  the issue loop pops it with priority.
- **`Workload`** — the umbrella-spec seam; `RateWorkload` is the schedule generator
  that runs the loop above. Linear multi-turn needs no graph executor — the
  continuation queue *is* the sequencer; the DAG executor (`aiperf_runtime::graph`) is only for
  FORK/SPAWN branching.

---

## 4. Online / mock / offline parity

Same `RateWorkload` + `CreditIssuer` + `SlotPool`s + `StopChecker` code on all three;
only `{Clock, RequestSink}` are injected. ONLINE-real and ONLINE-mock differ by URL.
OFFLINE swaps `RealClock`→`SimClock` and the HTTP sink for the feature-gated in-process
engine sink. **Parity is code-path + report-schema, not
byte-identical metric values** — simulated vs real timings differ by construction
(per the port-exact ledger addendum).

---

## 5. Determinism

- The credit id (`credit_index = requests_sent++`) is unique and monotonically
  *assigned* — safe as a request key. Its mapping to wall-clock order is deterministic
  under the single loop (online *and* offline), because issuance is serial.
- Arrival RNG (`IntervalGenerator`) is seeded (`aiperf_runtime::rng`, BLAKE3-derived,
  order-independent) — bit-reproducible spacing for a given seed.
- Under `SimClock` the whole run (inter-arrival + think-time + engine steps) is a DES
  on one integer-ns timeline → byte-reproducible. No wall clock outside `RealClock`.

---

## 6. Build order (increments)

1. **`ConversationSource` (synthetic)** — yields fixed K-turn sessions over the
   segment pool; `next` + `next_turn_meta`. Unblocks multi-turn without a dataset.
2. **`CreditIssuer` + continuation queue** — the two-source priority loop; wrap the
   two `SlotPool`s + `StopChecker`; extend `run_paced` from "one request per tick" to
   "one credit (turn) per tick, continuation-priority."
3. **Prefill-release-on-TTFT** — wire the observer's first-token signal to
   `prefill_slots.release()` (also the graph path's `prefill_concurrency`).
4. **Think-time** — `delay_ms` → deferred continuation enqueue via `Clock::sleep`.
5. **Dataset-backed `ConversationSource`** — over the unified segment/dataset store.
6. **DAG branching** — route FORK/SPAWN through `aiperf_runtime::graph`; children bypass the
   session slot + the continuation queue (dispatched directly), per source.

Increments 1–4 deliver linear multi-turn request-rate online + offline; 5–6 add real
datasets and agentic branching.

---

## 7. Original risks / open questions (resolved below where noted)

- **Session-slot release site — resolved.** `IssuedCredit::is_final_turn` is
  available to the scheduled terminal callback; request/duration truncation and
  workload cleanup release non-final sessions explicitly.
- **Prefill TTFT hook — resolved.** `ScheduledRuntime::issue_turn_with_hooks`
  carries an `Rc`-local first-token callback plus terminal fallback, with no
  `Arc` or lock on the hot path.
- **Numbering vs work-steal.** This path is single-loop, so numbering is a plain
  counter. It must NOT be conflated with the work-steal atomic id (a different,
  no-policy deployment).
- **Phase model.** The Python system has warmup/profiling phases with per-phase slot
  limits; this spec covers a single phase. Phase handoff (debt-drain across phases) is
  deferred to the unified-runtime phase work.

---

## 8. One-line summary

Request-rate multi-turn is a **single-loop credit issuer** that emits **one turn per
rate interval, continuation-priority**, gated by a **session `SlotPool`** (turn-0 →
final) and a **prefill `SlotPool`** (every turn → TTFT), bounded by `StopChecker`,
with turns materialized from the segment pool and think-time deferred via
`Clock::sleep` — control-plane on one loop (never the bottleneck), HTTP fanned out as
the data plane, and the whole thing deterministic under `SimClock`. Most primitives
exist in `aiperf_runtime::timing`; `aiperf_runtime::request_rate::RequestRateWorkload` composes them
through the shared scheduled runtime.

---

## 9. Implementation addendum (2026-07-11)

The linear request-rate chain is built in
`rust/runtime/src/request_rate.rs` as `RequestRateWorkload`, a normal
`scheduled::Workload`. It owns the single issuer loop, FIFO continuation queue,
cached next sampler draw, session guards, and per-turn prefill guards. The loop
draws the next interval before admission, re-anchors rather than catching up,
issues at most one turn per tick, blocks only continuation prefill acquisition,
and schedules think-time as delayed queue insertion through
`ClockTaskScheduler`.

`ScheduledRuntime::issue_turn_with_hooks` exposes the policy-neutral first-token
and terminal lifecycle edges. Request-rate drops the prefill guard on the first
meaningful token and idempotently drops it again at terminal for error,
cancellation, empty, and non-streaming no-token fallbacks. Session guards are
held from an admitted turn zero through the final return, or released during
request/duration truncation and failure cleanup. Interval generators and slot
pools are exposed to the existing ramp and adaptive actuators; cancellation and
turn-zero endpoint selection use the same scheduled ancillary pipeline as other
workloads.

The online CLI now lowers both `--input-file` datasets and synthetic `--turns`
templates into `NativeDatasetConversationSource` and runs `--request-rate`
through this workload. `--timing-json` is supported, loader-preferred sampling is
preserved, and every continuation is materialized through the unified segment
store with the real prior assistant reply.

Executable evidence:

- `rust/runtime/tests/request_rate_sim.rs` proves continuation priority, exact
  turn pacing, cached-sample retry, session and prefill limits, TTFT release,
  terminal fallback, think time, request/session/duration stops, drain behavior,
  and reply-spliced materialization under `SimClock`.
- Dataset-backed multi-turn dispatch and reply splicing is proven under
  `SimClock` in the same `request_rate_sim.rs` scenarios above; real wall-clock
  coverage of the shared scheduled runtime that this workload composes rides
  `rust/runtime/tests/scheduled_real_mock.rs` (fixed-schedule + user-centric over
  the in-repo mock), and the runner product online path is covered by
  `rust/cli/tests/online_v2_stdio.rs`.
- Existing scheduled, ancillary, adaptive, and workspace suites cover the shared
  runtime and actuator regressions.

DAG fan-out remains owned by `aiperf_runtime::graph`. The separately designed engine sink
is now built behind the `aiperf-runtime` crate's `dynosim` feature: the unchanged
continuation-priority request-rate workload runs against Dynamo on one `SimClock`
without HTTP through `aiperf_runtime::dynosim::run_request_rate_offline`, and the
feature-gated `dynosim_offline` transport is exercised end-to-end by the runner's
`rust/cli/tests/offline_stdio.rs`. Adaptive request-rate composition is now
available in that optional backend as well —
`aiperf_runtime::dynosim::run_request_rate_offline_with_adaptive_and_ancillary` threads
the same adaptive actuators and ancillary policies through the offline path used
by real HTTP.
