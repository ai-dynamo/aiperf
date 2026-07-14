# AIPerf-Rust: `SchedulingPolicy` module sketch

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design sketch
**Companion:** `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` §5 (the credit-policy trap)
**Target:** `~/nvidia/projects/aiperf/ajc/rust/rust/aiperf/src/run.rs`

---

## The problem this fixes

`run.rs` today is pure closed-loop concurrency and nothing else:

```rust
// run.rs (current)
let sem = Arc::new(Semaphore::new(concurrency.max(1)));
for req in workload.generate() {
    let permit = sem.clone().acquire_owned().await?;   // request-slot, not session-slot
    obs.on_arrival(...);
    tokio::spawn(async move { sink.dispatch(req, obs).await });  // admit == dispatch
}
```

That is fine for a synthetic throughput spike but silently drops every scheduling
*policy* the Python credit system encoded (ledger §5): continuation-turn priority,
prefill-slot-release-on-TTFT, session-vs-request accounting, request-count recycle,
absolute-schedule rate pacing, arrival patterns, and the agentic warmup→profiling
handoff. **Delete the ZMQ credit protocol; keep the policy** — as one explicit,
single-threaded module.

---

## Shape: a `Scheduler` that owns "what to dispatch next, and when"

Runs on one thread (`!Send`, `Rc`/`RefCell`), driven by the injected `Clock` so it
works in ONLINE-real (wall) and OFFLINE-mock (sim) identically.

```rust
pub struct Scheduler {
    clock:    Rc<dyn Clock>,               // wall or sim — pacing + timeouts
    arrival:  Box<dyn IntervalGen>,        // Poisson | Gamma | Constant | ConcurrencyBurst
    sessions: SessionSlots,                // Semaphore over *sessions*, acquired on first turn
    prefill:  Option<PrefillSlots>,        // Semaphore released at TTFT, not completion
    pending:  VecDeque<TurnToSend>,        // continuation turns of in-flight sessions (priority)
    source:   ConversationSource,          // sampler over media-free metadata
    stop:     StopCondition,               // request-count(recycle) | num-conversations | duration
    phase:    CreditPhase,                 // Warmup | Profiling  (record-struct dimension)
}

/// Arrival pattern seam. 0 = concurrency-burst (issue as fast as slots free).
pub trait IntervalGen { fn next_interval_ns(&mut self, rng: &mut Rng) -> i64; }
```

### The issue loop — port `request_rate.py::execute_phase` semantics

```rust
impl Scheduler {
    pub async fn run(&mut self, sink: &dyn RequestSink, obs: &dyn RequestObserver) {
        let mut target = self.clock.now_ns();          // absolute schedule anchor
        while !self.stop.reached() {
            // 1. ABSOLUTE-SCHEDULE PACING (no drift): schedule the *next* target
            //    before issuing, and re-anchor if we fell behind by > a bounded window.
            target += self.arrival.next_interval_ns(&mut self.rng);
            self.wait_until(target).await;               // clock.sleep on timerfd / sim-advance
            if self.behind(target) { target = self.clock.now_ns(); }   // re-anchor, no burst

            // 2. PRIORITY ORDER (frees session slots faster, avoids starvation)
            if let Some(turn) = self.pending.pop_front() {
                self.dispatch(turn, sink, obs).await;    // (a) continuation turn wins
            } else if let Some(permit) = self.sessions.try_acquire() {
                let turn = self.source.next_first_turn(permit);   // (b) start a new session
                self.dispatch(turn, sink, obs).await;
            } else {
                self.clock.yield_now().await;            // (c) concurrency-burst MUST yield
            }
        }
    }

    async fn dispatch(&mut self, turn: TurnToSend, sink: &dyn RequestSink, obs: &dyn RequestObserver) {
        let prefill = self.prefill.as_ref().map(|p| p.acquire());   // GPU prompt-processing pressure
        // TTFT callback releases the prefill slot the moment the first token lands,
        // NOT at completion — this is the disagg-realism policy.
        let on_ttft = { let p = prefill.clone(); move || drop(p) };
        let result = sink.dispatch_with(turn, obs, on_ttft).await;
        self.on_return(turn, result);
    }

    /// The credit-return equivalent: release/queue based on session progress.
    fn on_return(&mut self, turn: TurnToSend, r: DispatchResult) {
        drop(turn.prefill_permit);                       // safety-net release if no TTFT fired
        match self.source.next_turn_of(turn.session) {
            Some(next) => self.pending.push_back(next),  // more turns → keep session slot, queue next
            None => {
                turn.session_permit.release();           // final turn → free the session slot
                self.stop.on_session_done(&mut self.source); // request-count → recycle dataset
            }
        }
    }
}
```

### Session vs request accounting

`SessionSlots` is a `Semaphore(max_sessions)` acquired **on the first turn only** and
held until the session's final turn returns — so `--concurrency N` means N sessions
in flight, not N requests. (Agentic: promote to a `SessionTree` slot held until the
whole subagent tree drains — port `session_tree.py` semantics.)

### Stop conditions & recycle

`StopCondition` distinguishes `--request-count N` (recycle the dataset to fill idle
session slots while long traces are mid-delay) from `--num-conversations N`
(single-pass). This is the `gotcha_aiperf_request_count_recycles_dataset` semantics.

---

## Phases & agentic handoff

`PhaseRunner` builds a fresh `Scheduler` per phase; trajectory state persists across
the boundary via a shared `ConversationSource`. The warmup→profiling handoff is a
small state machine (ledger §2 / agentx): `baseline warmup → cache-pressure warmup →
drain wire but preserve paused branches → handoff → profiling`, with a
warmup-failure backstop that aborts profiling so steady-state metrics aren't biased.
Port the *state machine* exactly; use tokio `JoinSet` + structured cancellation
instead of Python's `getattr`/`execute_async`/`_flush_tasks` plumbing.

Every record the scheduler produces carries `phase: CreditPhase` (a record-struct
dimension, per the ledger §4) so metrics separate warmup from profiling for free.

---

## Integration with `run.rs`

Replace the naive for-loop with:

```rust
let mut sched = Scheduler::new(clock, arrival_from(cfg), SessionSlots::new(cfg.concurrency),
                               prefill_slots(cfg), source, stop_from(cfg), CreditPhase::Profiling);
let report = sched.run(sink.as_ref(), obs.as_ref()).await;
```

The `sink` stays the `RequestSink` trait; the only new coupling is that `dispatch_with`
takes a TTFT callback so the scheduler can release the prefill slot early. Everything
else (thread-per-core, `Rc`/`RefCell`, sim/wall clock) is unchanged.

---

## Checklist coverage (ledger §5)

| Policy | Where in this sketch |
|---|---|
| Continuation-turn-before-new-session | `run()` step 2, `pending.pop_front()` first |
| Prefill slot released on TTFT | `dispatch()` `on_ttft` closure |
| Session-slot (not request) accounting | `SessionSlots`, acquired first-turn-only |
| `--request-count` recycle | `StopCondition::on_session_done` |
| Absolute-schedule pacing + re-anchor | `run()` step 1 |
| Arrival patterns | `IntervalGen` trait (Poisson/Gamma/Constant/Burst) |
| Agentic warmup→profiling handoff | `PhaseRunner` state machine |
| Phase as record dimension | `Scheduler.phase` stamped on every record |

---

## Addendum — 2026-07-10 (superseded)

This sketch is **superseded by `2026-07-10-unified-graph-runtime-design.md`**. That
later spec — grounded in a line-by-line read of all 37 files in `src/aiperf/timing/`
— keeps every policy this sketch names but changes the *shape*: instead of a bespoke
single-threaded `Scheduler` loop, the policy is realized on the shared graph-IR
executor as trait compositions:

- the `Scheduler`'s issue-loop → a `Workload` **schedule generator** (the executor
  drives; strategies no longer own a dispatch loop);
- session/prefill slots → `SlotPool` (with the debt-drain on capacity decrease this
  sketch omitted);
- absolute-schedule rate pacing + continuation-priority → `RatePool` (priority
  `WaitQueue`);
- arrival patterns → `IntervalGenerator`; think-time / `max(now,·)` / fixed-schedule
  timestamps → the `Gate` edge-delay arithmetic;
- warmup→profiling handoff, stop-condition chain, cancel-drain teardown → the
  `StopCondition` chain + phase lifecycle described there;
- the adaptive controller (which this sketch did not cover) → a `Controller` seam.

The original text below is retained as lineage; where it conflicts with the unified
design, the unified design governs.
