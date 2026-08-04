<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global-exact dispatch for `workers>1`

## Purpose

`workers>1` scheduled execution spawns `W` self-contained sub-cell OS threads
(`rust/runtime/src/engine/sharded_scheduled.rs`). Left to a static per-thread
partition alone, concurrency and rate targets are sliced `1/W` up front and
each thread paces and admits requests against its own local share with no
runtime coordination — only approximating a single global concurrency limit
and a single global request rate in expectation, not reproducing Python's
single shared admission gate. This spec describes the `runtime.dispatch`
selector (`sharded` | `global` | `global-hop` | `global-push`) that closes that
gap: `global` (the default for `workers>1`) admits from a shared per-cell gate
so aggregate concurrency and rate are byte-exact against a single global
limiter; `global-hop` additionally reproduces exact global issuance order for
cases where shared admission alone is insufficient; `global-push` makes that
same single-issuer order far cheaper by routing credits instead of awaiting
requests.

## Built

- `DispatchMode` (`rust/runtime/src/engine/protocol.rs`) is the `Sharded` |
  `Global` | `GlobalHop` | `GlobalPush` enum, `#[serde(rename_all = "kebab-case")]`, with
  `Global` as `#[default]`. It is configurable through `runtime.dispatch` in
  Config v2 YAML, the `--dispatch` CLI flag (`rust/cli/src/flags.rs`,
  `Flags::dispatch_mode`), and the protocol-v2 wire request. An explicit
  `--dispatch` wins over an authored `runtime.dispatch`
  (`rust/cli/src/yaml.rs`), matching the `--cells`/`runtime.cells`
  precedence. `runtime.dispatch`/`--dispatch` are config-surface fields only;
  there is no separate `--workers` CLI flag (see
  `rust/e2e-tests/tests/global_dispatch_real_clock.rs`).
- `Sharded` is today's static per-thread partition: `owned_positions`,
  `two_level_partition`, `slice_phase_for_thread`
  (`rust/runtime/src/engine/sharded_scheduled.rs`) slice concurrency, rate,
  and request budget `1/W` up front per worker thread, retained as an
  explicit throughput-oriented opt-in where byte-exact parity does not
  matter. The same `two_level_partition` also slices the **dataset**: each
  thread narrows the corpus to its own authored-index residue class and
  recycles inside it, so `W` threads walk `W` interleaved short cycles rather
  than one long one.
- `Global` (default for `workers>1`) keeps each worker thread's own
  transport, capture, and measurement, but draws concurrency and
  request-rate admission from a shared `GlobalAdmission` gate
  (`rust/runtime/src/engine/execute.rs`) built once per cell, on the main
  thread, before worker threads spawn, from the cell-local (already
  `owned_positions`-sliced, not further thread-sliced) phase budgets:
  - `GlobalSlotPool` (`rust/runtime/src/timing/slots.rs`) is the
    `Send`+`Sync` cross-thread concurrency admission gate — a semaphore with
    a runtime-adjustable limit (debt-tracked decreases, immediate-capacity
    increases) that every worker thread in the cell shares as one
    `Arc<GlobalSlotPool>` per concurrency-capped phase.
  - `GlobalRateGate` (`rust/runtime/src/timing/rate_gate.rs`) is the
    `Send`+`Sync` cross-thread rate-pacing gate: a single atomic slot counter
    modeling a fixed-interval base grid (`claim_offset_ns` hands out `0`,
    `interval_ns`, `2*interval_ns`, ... gaplessly across every calling
    thread; `claim_slot` returns the same claim's dense slot **index**
    alongside that offset). Each caller still draws its own mean-zero jitter
    offset from its local `IntervalGenerator` and adds it to its claimed base
    slot. This keeps the **aggregate rate** exact but does **not** reproduce
    true Poisson/Gamma arrival-process statistics (the resulting inter-arrival
    times are grid-plus-offset, not a renewal process); exact
    arrival-*pattern* parity is `global-hop`'s job.
  - **Slot-addressed draws** tie the conversation to the rate slot on a
    rate-paced phase (`RequestRateWorkload::slot_addressed_draws`,
    `rust/runtime/src/request_rate.rs`). The workload claims one
    `ClaimedSlot` and draws that tick's new session at corpus position
    `slot.index` through `ConversationSource::next_at_position`, so the
    admission time and the drawn conversation are two pure functions of one
    lock-free `fetch_add` and their orders coincide by construction. Without
    the tie they are separately correct but unrelated sequences: `admit_ns` is
    monotone in the claimed slot while each thread walks its own private
    `next += stride` cursor, so sorting per-record output by `admit_ns` does
    not recover the corpus walk (measured 0.75% of records in sequential
    position on a 16-worker rig, median offset 53 conversations).
    - The tie is selected only when both hold: the source is
      position-addressed, and every sampleable conversation is single-turn. A
      continuation consumes a rate slot without drawing a new conversation, so
      a multi-turn corpus would leave holes in the position walk and never
      draw the conversations at the skipped positions.
    - A tick that cannot admit retains its claimed slot, so a drawn sample is
      never discarded and the claimed position sequence stays dense.
    - Exact admission order additionally requires `constant` arrival (the
      mean-zero jitter offset is nonzero for `poisson`/`gamma` and reorders
      adjacent slots) and, because `admit_ns` is each worker's own
      `Clock::now_ns()` at issuance (`rust/runtime/src/scheduled.rs`), a clock
      whose wake is exact at the claimed slot. Under a real clock per-thread
      wake-up skew leaves a residual scatter of roughly
      `skew * rate` slots — local, bounded noise rather than the corpus-scale
      decoupling above. Byte-exact per-request issuance order remains
      `global-hop`/`global-push`, which stamp every `admit_ns` on one thread.
  - Dataset sampling is **position-addressed** over the full corpus
    (`rust/runtime/src/multiturn.rs` `DrawMode::Position`,
    `rust/runtime/src/dataset/sampler.rs` `Sampler::at_position`): worker
    thread `i` of `W` draws absolute corpus positions `i, i+W, i+2W, …` — the
    positions its `two_level_partition` residue owns — instead of recycling
    inside its own residue class. Closed form: one `next`/`stride` pair per
    thread, no lock, atomic, or cross-thread state.
    - The guarantee is over the **multiset** of drawn conversations, not a
      sequence: threads draw concurrently and nothing orders their draws
      against each other. On a rate-paced phase the slot-addressed draw above
      does order them (`admit_ns` order is the corpus walk, up to jitter and
      clock wake precision); on a concurrency phase `admit_ns` is wall-clock at
      slot-free and nothing orders the draws, so per-request issuance order
      there remains `global-hop`/`global-push`.
    - The union across threads is one unpartitioned sampler's draw multiset
      **for budget-bounded phases only**. Thread `i` contributes the positions
      `≡ i mod W`, so the union is the contiguous prefix `0..T` exactly when
      each thread's draw count equals `|{k < T : k ≡ i mod W}|`. That is what
      `slice_common` (`rust/runtime/src/engine/sharded_scheduled.rs:242-252`)
      produces: it slices `requests` and `sessions` with `owned_positions`
      (`rust/runtime/src/engine/cell_launcher.rs:272-279`,
      `(total - k).div_ceil(count)`), which is that residue-class cardinality
      by construction. Both fields are `Option`. On a **duration-bounded**
      phase (`requests: None`, `sessions: None`) neither is sliced, per-thread
      draw counts are load-dependent, and the union is a ragged set with holes
      rather than a clean prefix — still full-corpus reach, but no exact
      single-issuer multiset.
    - Applies to strategies whose draw is a pure function of position
      (`sequential`, which every concrete loader returns from
      `preferred_sampling_strategy`); RNG-stateful strategies (`random`,
      `shuffle`) have no closed form, fail the constructor's `at_position`
      probe, and keep the per-shard owned-corpus walk. `Sharded` keeps the
      owned-corpus walk in every case: it is the throughput opt-in where
      byte-exact parity does not matter.
  - `GlobalAdmission` is `Some` only under `Global`; `None` under `Sharded`
    (per-thread `1/W` slicing needs no shared gate) and under
    `GlobalHop`/`GlobalPush` (their single coordinator loop enforces the full
    cap through one local `SlotPool`, so no cross-thread gate is needed — see
    `rust/runtime/src/engine/global_hop.rs`).
  - Conversation **enumeration** is unaffected by `Global` for both
    `fixed_schedule` and `user_centric`: `ConversationSource::conversations()`
    returns `owned_metadata` (`rust/runtime/src/multiturn.rs:1855-1857`), which is
    built from the partition alone and never from the position-addressing
    flag, so each thread still enumerates only its residue class.
    - `fixed_schedule` is unaffected outright — it only enumerates
      (`rust/runtime/src/fixed_schedule.rs:85`) and continues turns by
      conversation id; it never takes a sampler draw.
    - `user_centric` **draws** through `source.next(…)`
      (`rust/runtime/src/user_centric.rs:416`) from the same source, so under
      `Global` at `workers > 1` its draws are position-addressed like any
      other. It therefore shapes its plan from
      `ConversationSource::sampled_conversations()` — the DRAW corpus — rather
      than from `conversations()`: the native source returns the full corpus
      under `DrawMode::Position` and the residue class under `DrawMode::Owned`
      (`rust/runtime/src/multiturn.rs:1859-1869`), so a `Global` shard averages
      what it samples while a `Sharded` shard keeps its self-contained
      sub-corpus. Enumeration itself (`conversations()`) is untouched in both
      modes.
      - That mean is the only thing the two bases could disagree on:
        `sampled_conversations()` feeds the empty-dataset bail, the
        `average turns >= 2` admission, and the mean turn count
        (`user_centric.rs:389-403`), and the mean reaches `plan_user_centric`
        solely as `avg_session_turns`, which sets the virtual-history depth
        `session_lifetime` and the coprime `spacing_step`
        (`rust/runtime/src/timing/user_centric.rs:105-114`) — never
        `stagger_ns` or `turn_gap_ns`, which are functions of `num_users` and
        `request_rate` alone.
      - The residue class is a systematic `1/W` sample by authored index, so
        its mean is an unbiased estimator of the corpus mean and the two bases
        agree except when turn count correlates with `authored_index % W`. The
        consequence is not confined to shaping: `avg_session_turns` gates the
        `average turns >= 2` bail, so a residue class of mostly single-turn
        conversations would abort a run the whole corpus admits, with the
        outcome depending on `workers`. `user_centric_shapes_from_the_drawn_corpus_not_the_enumerated_one`
        (`rust/runtime/src/multiturn.rs`) pins both directions.
      - Per-user turn accounting is re-derived from the concrete draw
        regardless of basis: `num_turns = min(planned.max_turns,
        actual_turns).max(1)` (`rust/runtime/src/multiturn.rs:1042-1045`), and
        the pool records that actual (`user_centric.rs:425-426`). Resolution is
        total — `metadata` and `metadata_by_id` are built over the full corpus
        unconditionally (`multiturn.rs:1605-1639`) — so drawing a conversation
        this thread does not own resolves correctly.
  - Only concurrency/rate admission moves to the shared gate; the dataset
    change above is the one other behavioural difference `Global` carries.
- `GlobalHop` is a single-coordinator hop executor
  (`rust/runtime/src/engine/turn_execution.rs`,
  `rust/runtime/src/engine/global_hop.rs`): one logical dispatcher
  (`ThreadPerCoreExecutor`) owns the full, un-thread-sliced schedule on the
  coordinator thread and hops individual prepared turns to worker OS threads
  over a bounded mpsc command queue, awaiting a oneshot reply. This
  reproduces exact request-to-thread assignment order (turn `i` -> worker
  `i % W`), not just exact aggregate concurrency/rate — the gap `Global`'s
  shared-admission-only fix cannot close because its `W` independent
  scheduling loops still race. `GlobalHop` does not consume
  `GlobalAdmission`; its exactness comes from "one loop, one full-cap local
  `SlotPool`", not a cross-thread gate.
- `GlobalPush` is a credit router (`rust/runtime/src/engine/global_push.rs`)
  sharing `GlobalHop`'s single-coordinator pipeline body
  (`global_hop::run_single_coordinator`) and its `pick_worker` placement. The
  issuer routes a credit and returns to its scheduling loop; the worker owns the
  whole request and returns the credit out of band on one shared stream
  (`WorkerCreditReport { uuid, worker, kind }`,
  `CreditReportKind::{FirstToken, CreditReturn}`) that one coordinator loop
  drains (`ScheduledRuntime::run_credit_returns`). Three consequences:
  - A worker's in-flight slot is released on CREDIT RETURN rather than at reply,
    which is the one deliberate behavioural difference from the hop;
    `HopRouting::LeastLoaded` can therefore break a tie differently, while
    `RoundRobin`/`Sticky` placement is identical.
  - A routed credit carries only identity (`CreditIdentity`: conversation,
    session, turn index, turn count) and the WORKER builds the request body
    through `WorkerMaterializer`, rebuilt per worker from a `Send + Sync`
    `WorkerMaterializationRecipe` over its own prepared endpoint table. Applied
    to single-turn sessions only: a continuation's body can splice the live
    model reply, which a worker replaying `build_turn_at` cannot reproduce.
  - A credit a worker holds is enrolled in the scheduler's drain accounting via
    `LocalTaskScheduler::begin_external_task`, so `wait_idle` still bounds the
    phase; grace escalation cancels at the worker
    (`RequestExecutor::cancel_credits`) and each credit is still RETURNED.
  Measured on 144 cores at ISL 550 / OSL 1 / concurrency 512: 95.6k requests/sec
  against `GlobalHop`'s 54.4k and `Sharded`'s 276.8k. A single issuer is bound by
  one thread doing every request's issuance work; that is the mode's ceiling, not
  a defect.
- `--cells` cellular tiling composes unchanged under every dispatch mode:
  `GlobalAdmission` is built from the cell-local phase budgets (already
  narrowed from the global run by `owned_positions(global, cell_id, cells)`
  upstream in the cellular controller), never further sliced across cells.
  Each cell process gets its own independent `GlobalAdmission`; cells remain
  separate processes and never share a gate with each other.
- Verification:
  `rust/runtime/src/engine/workers_characterization.rs` is the oracle
  covering `Sharded`/`Global`/`GlobalHop`/`GlobalPush` phase-shape parity —
  including `GlobalPush` exactly-once/deterministic merge, aggregate
  concurrency-cap and rate exactness, multi-turn completion, and byte-identical
  worker rebuild of a deferred credit — a
  `Sharded`-vs-`Global` divergence regression test, and SimClock-adjacent
  (RealClock-based) byte-exact determinism tests.
  `rust/e2e-tests/tests/global_dispatch_real_clock.rs` is a real-binary end-to-end
  `RealClock` spot-check proving `Global` mode's aggregate concurrency cap
  against a live `aiperf-mock-server` process across `workers=4` OS-thread
  sub-cells, with deterministic TTFT/ITL and raw per-record assertions per
  the CLAUDE.md generated-token-timing test requirements.

### Which mode delivers exact issuance ordering

Exact `admit_ns` ordering — sorting the per-record artifact by `credit_issued_ns`
and recovering the sequential corpus walk — **requires a single-coordinator mode**.
Measured on a 144-core rig, 997 conversations / 3,323 requests / `workers = 16`,
comparing the `credit_issued_ns` sort against the corpus walk:

| mode | exact | notes |
| --- | --- | --- |
| `global-push` | **100.00%** | both `constant`-rate and `concurrency` phases |
| `global-hop` | **100.00%** | both phase shapes |
| `global` | 4.42% | median offset 7 positions; 91.5% within 16 positions (was 0.75% before slot-addressed draws) |

`global` **cannot** reach exactness, and this is structural rather than a tuning
gap. `admit_ns` is derived from `credit.issued_ns`, which is `clock.now_ns()`
sampled *by whichever of the `W` threads issues the request* — `ScheduledRuntime`
is built per phase per shard, and each shard's issuer loop reads the clock on its
own thread (`let issued_ns = self.clock.now_ns()` in
`ScheduledRuntime::issue_turn_internal`, `rust/runtime/src/scheduled.rs`; the capture
converts it at `RunCapture::snapshot_live`/`fold_streaming`,
`rust/runtime/src/engine/execute/capture.rs`). Two threads that claim adjacent
global slots stamp them in whatever order the OS wakes them, so the sort key
carries per-thread wake skew that no amount of admission coordination removes.
The shared gate makes `global` exact in *what* is admitted and *how many*; it
cannot make the timestamps totally ordered. Removing the skew would take either
virtual time (single-reactor, and `SimClock` forces `workers = 1` — see the next
section) or funnelling every stamp through one thread, which is precisely what the
single-coordinator modes already are.

`global`'s near-ordering is configuration-dependent, so treat the 4.42% as one
measurement rather than a constant: an independent run on the same rig at
`concurrency = 64` with multi-turn sessions measured 0.06% exact and 5.1% within
16 positions. What is stable is the shape — `global` clusters near the true
position without hitting it, and the single-coordinator modes hit it every time.

For consumers that need the order and not the timestamps, the per-record artifacts
now carry `metadata.global_dispatch_index`, the dense global dispatch ordinal.
Sorting by it recovers exact issuance order in **every** mode, `global` included,
because it is assigned by the issuance authority rather than read from a clock.

### Measured: `global-push` vs `global-hop`

Both deliver the same 100% ordering, so the choice between them is throughput.
Measured this session on the same rig:

| workload | `global-push` | `global-hop` | ratio |
| --- | --- | --- | --- |
| single-turn | 91,945 rps | 51,464 rps | **1.79×** |
| multi-turn | 60,448 rps | 43,508 rps | **1.39×** |
| run-to-run variance | ±3% | ±13% | — |

The mechanism is the one `global_push.rs` documents: `global-hop` holds a
coordinator-side future and a worker slot from send *through reply*, so the single
issuer sits inside every request's lifetime; `global-push` routes an identity-only
credit the worker materializes and returns out of band, keeping the issuer in
neither the request lifetime nor its body construction.

This is a measurement, not a recommendation to retire `global-hop`.

### Boundary: `SimClock` is single-worker and Graph-only

`SimClock` unconditionally forces `workers = 1`
(`execute_prepared_native_plan_uncommitted_with_runtime_factories` in
`rust/runtime/src/engine/execute.rs`): a virtual-time run can only advance the
single reactor its idle-pump drives, while thread-per-core workers each own a
private reactor the pump cannot reach. `SimClock` is selected only by
transports whose `uses_virtual_clock()` binding says so (currently `dry_run`
with `clock: sim`) and is not a configuration `PreparedLinear` scheduled
workloads (concurrency, request-rate, user-centric, fixed-schedule) select.
`runtime.dispatch`'s `Global`/`GlobalHop`/`GlobalPush` cross-thread coordination is
therefore inert for `SimClock` runs: there is exactly one worker thread and no
cross-thread admission to coordinate. This is a permanent architectural
boundary of the clock seam, not a gap in dispatch-mode coverage — "SimClock-
driven multi-worker dispatch" is not a real configuration.

## Source anchors

- `rust/runtime/src/engine/protocol.rs` — `DispatchMode`.
- `rust/runtime/src/timing/slots.rs` — `GlobalSlotPool`.
- `rust/runtime/src/timing/rate_gate.rs` — `GlobalRateGate`, `ClaimedSlot`.
- `rust/runtime/src/request_rate.rs` — slot-addressed draws
  (`slot_addressed_draws`, `can_address_draws_by_slot`,
  `slot_addressed_offset_ns`) and their acceptance test
  `global_rate_paced_admission_order_is_the_sequential_corpus_walk`.
- `rust/runtime/src/engine/execute.rs` — `GlobalAdmission`,
  `ShardedShared::dispatch_mode`/`global_admission`, the
  `virtual_clock`/`workers = 1` SimClock gate.
- `rust/runtime/src/engine/sharded_scheduled.rs` — `Sharded` static partition
  (`owned_positions`, `two_level_partition`, `slice_phase_for_thread`,
  `run_sharded_scheduled`).
- `rust/runtime/src/engine/turn_execution.rs`,
  `rust/runtime/src/engine/global_hop.rs` — `GlobalHop`'s
  `ThreadPerCoreExecutor`-shaped single-coordinator dispatcher.
- `rust/runtime/src/engine/global_push.rs`,
  `rust/runtime/src/transport/core/dispatch.rs`,
  `rust/runtime/src/scheduled.rs`, `rust/runtime/src/multiturn.rs` —
  `GlobalPush`'s credit-routing seam, credit-return loop, and worker-side
  materialization.
- `rust/runtime/src/engine/workers_characterization.rs` — parity oracle.
- `rust/e2e-tests/tests/global_dispatch_real_clock.rs` — real-binary `RealClock`
  aggregate-concurrency e2e spot-check.
- `rust/cli/src/flags.rs`, `rust/cli/src/yaml.rs` — `--dispatch` flag and
  `runtime.dispatch` YAML surface and precedence.
- `rust/runtime/src/multiturn.rs` — conversation enumeration/partitioning for
  `fixed_schedule`/`user_centric`, unaffected by dispatch mode.
