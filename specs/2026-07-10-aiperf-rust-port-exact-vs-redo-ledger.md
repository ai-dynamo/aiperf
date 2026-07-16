# AIPerf-Rust: Port-Exact vs Redo-Cleaner Ledger

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design reference / decision ledger — start-here port-exact vs redo-cleaner vs throw-away ledger. Parity means shared code path + report schema across modes, not byte-identical real-vs-sim metric values; credit *policy* survives through the unified-runtime `Workload`/`SlotPool`/`RatePool`/`Gate` seams.
**Companions:** `2026-07-10-shared-rust-architecture-northstar.md`,
`2026-07-10-steppable-clock-injected-engine-design.md`,
`2026-07-10-aiperf-transport-rust-port-design.md`,
`2026-07-09-graph-ir-rust-port-design.md`
**Memory anchors:** `aiperf-rust-why`, `aiperf-rust-design`, `aiperf-rust-modes`,
`aiperf-dynamo-ownership`

---

## 0. Framing (authoritative, per `aiperf-rust-modes`)

AIPerf-Rust is **ONE** unified front-end (workload/request generation + metrics
collection + reporting + datasets) that drives **THREE interchangeable execution
modes** over a single `{transport, clock}` seam:

1. **ONLINE-REAL** — real HTTP to a real inference server, wall clock (what Python
   AIPerf does today).
2. **ONLINE-MOCK** — load against the mocker as a live async engine / mock server,
   wall clock.
3. **OFFLINE-MOCK** — in-process virtual-clock co-simulation of the mocker, no
   network / no GPU, deterministic (what Dynamo Replay does today).

**The whole point:** because it is ONE front-end, every feature (arrival patterns,
datasets, multi-turn, tokenization, metrics, exporters) works across ALL THREE
modes **for free** — build once, zero drift. Offline-mock is one of three EQUAL
targets, not a bonus "Layer 2." (This supersedes the earlier
"offline-subsumes-Replay / two-layer" framing.)

This ledger records, for each AIPerf concept, whether the Rust front-end should
**port it near-exact**, **redo it cleaner**, or **throw it away** — and flags the
handful of things the current `ajc/rust` spike dropped that will silently change
benchmark semantics if not restored deliberately.

State of the `ajc/rust` spike as of this writing: a "walking skeleton"
(`rust/runtime/src/run.rs:4`), ~11 commits, six crates (`aiperf-clock`,
`loadgen-core`, `aiperf-transport-http`, `aiperf-core`, `aiperf-graph`, `aiperf`). It
already implements thread-per-core single-threaded tokio runtimes, `Rc`/`RefCell`
per-trace state, raw hyper + UDS transport, `timerfd`/`SimClock`, and has deleted
ZMQ / services / credits / plugins. Architecture thesis is sound; this ledger is
mostly "keep doing that, plus these gaps."

---

## 1. The crown jewel — port the *concept* exactly

This is the most important thing and it is **not** about performance. It is what
buys all three modes for free.

- **`Clock` trait (`aiperf_runtime::clock`, formerly the `aiperf-clock` crate)** — `now_ns` / `sleep` / `next_event_time` /
  `advance_to` / `is_virtual`, with `RealClock` (Linux `timerfd`, CLOCK_MONOTONIC,
  awaited via tokio `AsyncFd`) and `SimClock` (integer-ns `BinaryHeap` DES,
  `(deadline, seq_no)` tie-break). `is_virtual()` selects `drive_sim` vs
  `drive_real` over the *same* executor.
  - **Keep integer-ns keying** (dynosim uses f64 ms). This is what makes sub-ms
    firing gates byte-exact and is why `tokio::time` (1 ms timer wheel) is
    rejected for timing.
- **Transport/sink traits (`loadgen-core/src/sink.rs`)** — `Dispatchable` /
  `RequestSink` / `RequestObserver` (`on_arrival/on_admit/on_token/on_terminal`).
  Real HTTP, mock HTTP, and in-process co-sim are three `RequestSink` impls behind
  one observer. This is the second half of "three modes for free."
- **Gap:** `rust/runtime/src/graph/mod.rs:14-15` explicitly says the offline
  virtual-clock co-sim path is **not** wired yet. The seam exists; the
  OFFLINE-MOCK sink is the missing third leg. Design it in now.

---

## 2. Port near-exact (external contracts + earned-in-blood algorithms)

These look simple, are not, and were paid for in bugs. Port the **behavior**
faithfully; guard with a byte-exact fixture harness (the `graph-rs` model:
`rust/graph-ir/tests/parity.rs`, ~19-case corpus).

| Concept | Source of truth (Python) | Why not "redo cleaner" |
|---|---|---|
| OpenAI SSE parse (fast-path `data:{…}`, `[DONE]`, backward usage walk) | `inference_response_models.py:174`, `find_last_non_empty_usage:270` | Backward walk keeps prompt/reasoning/completion counts consistent; errors silently corrupt token metrics |
| aiohttp timing breakdown — DNS/TCP/TLS split, TTFB vs **TTFH** vs **TTFT** | `models/trace.rs` (already ported), `AioHttpTraceData` | TTFT = first *non-empty content delta*, not first SSE frame; "send complete" = body EOS, not `send().await` (headers) |
| Metric formulas + genai-perf/aiperf field names | `metrics/types/*`, `metric_dicts.py` | The comparability contract + parity anchor. Field-name drift breaks downstream tooling |
| goodput avg-ITL `(e2e−ttft)/(osl−1)`, percentile rank `(len-1)*p/100` | `collector.rs:451` (already ported) | Subtle rounding; must match Python exactly |
| `reconcile_output_times` (server `usage.completion_tokens` authoritative when multi-token chunks) | agentx `observer.rs:24-46` | Preserves TTFT/E2E while stretching interior times |
| Server-metrics OpenMetrics-vs-classic routing + `_created` skip | agentx `data_collector.py:337` | vLLM Rust-frontend quirk found empirically; don't rediscover in prod |
| Graph firing-gate arithmetic (`f64` µs → truncate to ns) | `executor.rs:343` | Byte-exactness depends on preserving exact rounding (`graph-rs` proved this) |
| NaN/Inf discipline (finite-or-`None`, scrub before serialize) | `common/finite.py` | A boundary contract. In Rust make it a `FiniteFloat` newtype at the type level |
| Agentic barrier/session-tree semantics (if in scope) | `replay_dependencies.py:58`, `session_tree.py`, `t*` snapshot | Pure algorithm, subtle boundary rules (ordered touches, equal-start unordered). Port semantics; redo the plumbing with `JoinSet` |

---

## 3. Redo cleaner / throw away (accidental complexity of the Python model)

The spike already deleted these; that is correct. None are *features* — all are
GIL / multiprocess workarounds (`aiperf-rust-design`: "ZMQ, mmap, multiprocess are
ACCIDENTAL BURDENS, not benchmarking features").

| Python artifact | Fate in Rust |
|---|---|
| ZMQ message bus + 12 services/processes | In-process trait calls + tokio. **Gone** |
| `@on_message` / auto-subscription / registration / heartbeat / connection-probe (slow-joiner echo) | Exists only because processes. **Delete** |
| Credit issuer/router/return as a ZMQ round-trip | Collapses to `Semaphore` + direct `await`. **But keep the policy — see §5** |
| Sticky router as a service | Connection-pool `ConnectionReuseStrategy::StickyUserSessions` keyed by `correlation_id` (`pool.rs`, done) |
| GC disable / `gc.freeze` / manual ref-clearing / forkserver | Python-specific. **Vanish** |
| `dataclass(slots=True)` + `msgspec` tagged unions | Plain Rust structs/enums, no runtime tax |
| Per-processor JSONL/CSV shard writers + concat aggregators | Only exists because N record-processor **processes** can't share a writer. Single process → lock-free per-thread accumulators merged once (`transport_bench.rs:65-75`, done). **Whole shard layer disappears** |
| `plugins.yaml` YAML registry + Pydantic reverse-lookup | Mechanism goes (see §4 caveat) |
| mmap dataset cache | Shared memory in-process; no mmap IPC needed |

---

## 4. Design as first-class *now* (things Python retrofitted badly)

Not port, not drop — get the shape right up front.

- **Phase model (warmup vs profiling) as a first-class record dimension.** agentx
  shows this is the spine and was painful to retrofit: `MetricsAccumulator` had to
  switch from `session_num` indexing to an append-only `_next_record_idx` (credit
  numbers restart per phase), add a `benchmark_phase` categorical column, and
  thread `ExportContext.phase` + `phase_time_ranges` for server/client
  time-window alignment. **Put `phase` on the record struct; make summarization
  phase-windowed from day one.** Fixes "warmup record 0 overwrites profiling
  record 0" by construction.
- **Config: `clap` derive + `serde` structs, not hand-rolled `argv` scanning.**
  The early spike's positional/flag scan and `std::env::set_var("GRAPH_HTTP2", …)`
  are historical cautions — fine for a throughput spike, wrong for the tool. The
  binary uses `clap` derive with structured arguments; config work preserves the
  Pydantic *ergonomics* (validated, documented fields; YAML/CLI unification) and
  Python's validated configuration semantics without the Pydantic *runtime*. This
  is a redo, not a throw-away.
- **Metric taxonomy — a real decision, not an inertia drop.** The spike replaced
  RECORD/AGGREGATE/DERIVED + registry + topo-sorted dependency order
  (`metric_registry.py`, `graphlib.TopologicalSorter`) with one fixed
  `TraceCollector` struct. For a fixed metric set that is *better*. Decide on the
  axis: **do metrics need to be user/plugin-extensible?** If yes → a `Metric`
  trait + dependency ordering is the clean Rust version. If no → keep the struct.

---

## 5. The trap — deleting "credits" is right; deleting credit *policy* is a silent semantics change

`grep credit rust/**/*.rs` in `ajc/rust` = 0 hits, and admission is a vestigial
no-op (`admit == dispatch`, `http_sink.rs:235`). Fine for synthetic throughput
benchmarks; **wrong for real multi-turn / agentic runs.** The Python credit system
encoded scheduling *policy*, not just an IPC protocol. Kill the ZMQ credit
protocol; re-surface these policies through the unified graph/runtime executor's
`Workload`, `SlotPool`, `RatePool`, and `Gate` seams — not a separate bespoke
scheduler module unless a future design explicitly reintroduces one.

**Checklist against the current `ajc/rust` crates:**

- [ ] **Continuation-turn-before-new-session priority** (`request_rate.py:238`).
  Frees session slots faster, prevents starvation. Missing → latency distribution
  shifts under load. *Where:* the online driver in `rust/runtime/src/run.rs`
  currently just `Semaphore` + `tokio::spawn` per request with no turn priority.
- [ ] **Prefill slot released on TTFT, not on completion** (`issuer.py`,
  `sticky_router.py:476` FirstToken early-return). Models GPU prompt-processing
  pressure; required for disagg realism. *Where:* no prefill-slot concept on the
  online path; graph path has `prefill_concurrency` caps (`bench.rs:57`) but not
  TTFT-release wiring on the HTTP sink.
- [ ] **Session-slot vs request accounting** (concurrency = sessions in flight,
  not requests). *Where:* online `Semaphore` counts requests; agentic needs
  `SessionTreeRegistry`-style tree-scoped slots (slot held until whole tree
  drains, `session_tree.py`).
- [ ] **`--request-count N` recycles the dataset** vs `--num-conversations N`
  single-pass (`gotcha_aiperf_request_count_recycles_dataset`). *Where:*
  `workload.rs` builds a fixed instance list; recycle semantics absent.
- [ ] **Absolute-schedule pacing (cumulative target times, not relative sleeps)**
  with catch-up re-anchoring (`request_rate.py:140,209`). Prevents drift. *Where:*
  arrival offsets exist in the graph path; the online path has no rate pacer, only
  concurrency.
- [ ] **Arrival patterns** — Poisson / Gamma / Constant / Concurrency-burst
  interval generators (`intervals.py`). *Where:* not on the online path yet.
- [ ] **Agentic (if in scope):** cross-stream predecessor barriers
  (`infer_cross_stream_predecessors`), the `t*` snapshot split, and the
  **warmup→profiling handoff state machine** (baseline warmup → cache-pressure
  warmup → drain wire but preserve paused branches → handoff → profiling, with
  warmup-failure abort). Port the *state machine semantics* exactly; redo the
  `getattr`/`execute_async`/`_flush_tasks` plumbing with tokio `JoinSet` +
  structured cancellation.

**Rule:** the `Semaphore` is the *mechanism*; the policy above is the thing that
must survive the credit-system deletion.

---

## 6. Verification method (non-negotiable for the "port near-exact" rows)

Adopt the `graph-rs` proof-of-method for every §2 row:

1. A Python twin harness runs the **real** Python code path over a deterministic
   mock issuer + virtual clock, with `uuid4` monkeypatched to a counter (parity
   only), emitting `*.golden.json`.
2. The Rust side runs the same fixture through the real executor on `SimClock`,
   canonicalizes JSON (recursive key-sort to match `json.dumps(sort_keys=True)`),
   and asserts byte-equality.
3. Corpus grows per bug found. Byte-exact keys: deterministic `seq_no` tie-break,
   integer-ns time, injected deterministic id factory.

This byte-exact fixture harness governs the individual §2 algorithm ports (SSE
parse, timing breakdown, firing-gate arithmetic, reconciliation), where
byte-equality against the Python twin is the correctness gate.

Cross-mode parity is a *different, weaker* contract and must not be read as
byte-identical metric values. Online-real, online-mock, and offline-mock exercise
the **same** workload/gate/slot/collector/exporter code where possible and emit
the **same report schema** — that shared code path plus schema is the parity
anchor. Simulated and real transports are **not** expected to produce
byte-identical metric values; a simulated engine and a real GPU legitimately
differ. The trust guarantee is "same front-end, same schema, one code path across
all three modes," not "same numbers."

---

## 7. One-line summary

Keep AIPerf's **external contracts** and its **earned-in-blood algorithms**; keep
the **Clock + Sink trait seam** as the crown jewel (it is what makes real/mock/
offline free); throw away every **internal artifact of the multiprocess/GIL
model** (ZMQ bus, services, credit protocol, plugins-YAML, shard export, GC
hacks); and consciously **re-design three things Python retrofitted badly** —
phase-scoped metrics, config ergonomics, and scheduling policy. The `ajc/rust`
spike is ~80% aligned; its main gaps are the OFFLINE co-sim sink, phase-first-class
metrics, and re-surfacing credit *policy* through the runtime's workload/slot/rate/
gate seams (§5).
