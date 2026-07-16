# AIPerf-Rust: `AccuracyAccumulator` + `AccuracyResultsAnalyzer`

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** the first-class accuracy accumulator/analyzer and the static-accuracy
evaluator worker seam (`aiperf_runtime::accuracy_core`, `PythonEvaluator` over the
Lighteval/DeepEval worker) are built, but the `http + static_accuracy` pair is NOT
registered on the default product wire — it is a linked-but-off-wire worker seam
today, not a product-reachable pair. The stateful agentic vertical
(Harbor / AgentLab+BrowserGym / MCPMark) has been **REMOVED** from the `ajc/rust`
branch (`rust/runtime/src/accuracy_core/mod.rs`: "The external-evaluator provider-host
and agentic verticals have been removed; only the static lighteval worker path
remains"); its design is retained below as historical record only. The **long-term**
provider-neutral evaluator architecture was superseded by
`2026-07-12-external-evaluator-provider-host-boundary-design.md`, which has itself
since been removed.
**Companion:** analyzer/accuracy findings in memory `#15927949`;
`2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md`;
`2026-07-11-aiperf-runner-only-execution-surface-design.md`.
**Pattern mirrored:** `analysis/energy_analyzer.py` (`EnergyEfficiencyAnalyzer`) —
the reference for a clean analyzer.

---

## The problem

Accuracy started as a self-contained side-pipeline: the metrics accumulator had
**zero** accuracy references; the `accuracy_results` "analyzer" carried the wrong
signature; association was a fragile `session_num % len(tasks)` positional mapping
requiring sequential sampling; and the rich `GradingResult` (confidence / reasoning /
extracted_answer) was discarded — only two floats survived. Because accuracy wasn't an
accumulator, it got no timeslicing, no steady-state windowing, and no sweep/energy
joins.

**Fix (built):** make accuracy a first-class **accumulator + analyzer pair**, exactly
like energy. Accuracy-over-time, accuracy-under-load, and accuracy×perf×energy joins
then fall out for free. This design is implemented natively in Rust — the module paths
below are code truth, not intent.

---

## Data model — carry the full grading result, keyed by a real id

`aiperf_runtime::metrics_core::accuracy` (`rust/runtime/src/metrics_core/accuracy.rs`) owns the
typed record and grading result. The `correlation_id` is the REAL per-request id (from
the scheduling/dataset seam), never `session_num % len(tasks)`.

```rust
/// One graded response. record_type = "accuracy_records".
pub struct AccuracyRecord {
    pub correlation_id: CorrelationId,
    pub task: TaskId,                 // e.g. "mmlu_pro.chemistry"
    pub phase: CreditPhase,           // warmup vs profiling (record dimension)
    pub start_ns: i64,                // enables time-range queries (timeslice/steady-state)
    pub end_ns: i64,
    pub result: GradingResult,        // correct, unparsed, confidence, extracted, ground_truth, reasoning
}

pub struct GradingResult {
    pub correct: bool,
    pub unparsed: bool,               // fell through to a fallback regex tier
    pub confidence: Option<f64>,
    pub extracted: Option<String>,
    pub ground_truth: String,
    pub reasoning: Option<String>,    // kept for per-record export, not aggregation
}
```

Grading is owned by the canonical Python evaluator (see the boundary below), never by a
Rust grader. Rust captures the terminal response text through the generic
`TurnRecordProcessor` seam and hands it to the worker; the worker returns the
`GradingResult`. The graded record is emitted with `record_type = "accuracy_records"`
and the routing table fans it to the accumulator. The one correctness policy — the
`0.5` Lighteval threshold and the correct/incorrect decision — lives in a single
`AccuracyPolicy` constant owned by the worker boundary, not duplicated across grader and
bucketer.

---

## `AccuracyAccumulator` — owns the `accuracy_records` type

```rust
impl Accumulator for AccuracyAccumulator {
    const RECORD_TYPE: &str = "accuracy_records";

    fn process_record(&mut self, r: AccuracyRecord) {
        // columnar, like the metric accumulator: push into per-task GrowableArrays
        let t = self.tasks.entry(r.task).or_default();
        t.correct.push(r.result.correct as u8);
        t.unparsed.push(r.result.unparsed as u8);
        t.start_ns.push(r.start_ns);
        t.end_ns.push(r.end_ns);
        if let Some(c) = r.result.confidence { t.confidence.push(c); }
        self.by_corr.insert(r.correlation_id, /* index */);   // typed association
    }

    /// Time-range mask — this is what buys timeslice + steady-state windowing FOR FREE.
    fn query_time_range(&self, start_ns: i64, end_ns: i64) -> BitVec { /* start>=lo & end<=hi */ }

    fn export_results(&self, ctx: &ExportContext) -> AccuracySummary {
        // overall + per-task accuracy over ctx's window: mean(correct[mask]),
        // unparsed rate, n, Wilson CI on the pass rate.
        AccuracySummary { overall, per_task, unparsed_rate, ci, n }
    }
}
```

The accumulator is columnar (numpy/`ndarray`-style) precisely so
`compute_results_for_mask(window)` works the same way perf metrics do — accuracy
participates in timeslicing and steady-state windowing. Confidence intervals are Wilson
intervals over the pass rate.

---

## `AccuracyResultsAnalyzer` — mirror `EnergyEfficiencyAnalyzer`

```rust
impl Analyzer for AccuracyResultsAnalyzer {
    // REAL dependency declaration — enforced by a toposort, not a dead ClassVar.
    const REQUIRED: &[AccumulatorType] = &[AccumulatorType::Accuracy];
    // Optional joins — present ⇒ richer output, absent ⇒ graceful skip of that block.
    const OPTIONAL: &[AccumulatorType] = &[AccumulatorType::MetricResults, AccumulatorType::GpuTelemetry];

    async fn summarize(&self, ctx: &SummaryContext) -> AccuracyAnalysis {
        let acc = ctx.get_output::<AccuracySummary>(AccumulatorType::Accuracy)?;

        // Base: overall + per-task accuracy, unparsed rate, CIs (over ctx window).
        let mut out = AccuracyAnalysis::from(acc);

        // JOIN 1 — accuracy at the steady-state window (if metrics present):
        if let Some(m) = ctx.get_output::<MetricsSummary>(AccumulatorType::MetricResults) {
            out.accuracy_at_load = Some(quality_vs_throughput(acc, m));   // quality-at-goodput
        }
        // JOIN 2 — accuracy per watt (if telemetry present):
        if let Some(e) = ctx.get_output::<EnergyEfficiencySummary>(AccumulatorType::GpuTelemetry) {
            out.correct_answers_per_kwh = safe_div(acc.correct_count, e.total_energy_j / 3.6e6);
        }
        out
    }
}
```

Because accuracy, `metric_results`, and `gpu_telemetry` are **all accumulators under one
`SummaryContext`**, the analyzer reads them together and emits joins that a side-pipeline
structurally cannot produce:

- **accuracy-over-time** (per timeslice) — is quality degrading as KV-cache fills?
- **accuracy-under-load** (steady-state window only) — quality at the SLA operating point.
- **quality-at-goodput / accuracy-per-watt** — the frontier that unifies the three axes.

Single-input accuracy (no joins) still works when only the accuracy accumulator is
present — the same graceful-degradation contract as energy. Metric goodput /
request-throughput joins are wired end to end; energy joins remain dormant until GPU
telemetry producers exist.

---

## Real dependency enforcement (no dead metadata)

`REQUIRED` / `OPTIONAL` are real. The Rust records-manager equivalent toposorts analyzers
by `REQUIRED` before `summarize`, skips an analyzer whose required accumulators are
absent, and runs analyzer→analyzer dependencies in dependency order (so one analyzer can
consume another's output deterministically). This replaces the earlier "works by luck of
insertion order" behavior with a `petgraph` toposort and removes that whole class of
bugs.

---

## Canonical-provider boundary — Rust schedules, Python scores

Benchmark preparation, answer extraction, symbolic comparison, hidden-test
decoding/execution, agent scaffolds, environments, and grading are **too easy to
duplicate incorrectly** and must have exactly one canonical owner. There is **no
Rust-native benchmark catalog, prompt builder, grader, symbolic-math engine, code
executor, accuracy registry, or Hugging Face dataset client.** The earlier native-Rust
grader/benchmark zoo, `AccuracyDispatcher`/`HttpAccuracyDispatcher`, and the manual
accuracy run loop were all deleted.

The implemented boundary:

- **Rust owns** scheduling, admission, endpoint materialization, inference-server
  HTTP/SSE I/O, response parsing, timing, performance metrics, accuracy accumulation,
  analysis, retries, cancellation, accounting, native-v2 reporting, and every
  upstream/external network operation.
- **One lightweight, long-lived Python worker owns** canonical benchmark/dataset
  preparation, prompts, generation settings, private test material, task execution, and
  scoring. Rust launches and supervises it with `tokio::process::Command` and exchanges
  correlated, versioned JSONL over stdin/stdout; worker diagnostics use stderr only.
- **Data-plane ownership is absolute.** The `aiperf_runtime::accuracy` module never creates or
  calls an HTTP client. Every model request — static or agentic, primary or environment-
  or verifier-originated — flows through the same `ScheduledRuntime` / `TurnDispatcher` /
  `TransportSink`, SSE parser, observers, timing, metrics, credit, and report path used by
  ordinary scheduled runs.

`aiperf_runtime::accuracy` (`rust/runtime/src/accuracy.rs`) is the control-plane integration: the
`AccuracyDataset` lowers opaque evaluator-authored problems into the unified
content-addressed dataset with typed correlation/task/ground-truth associations, and
`AccuracyRecordProcessor` captures terminal text and timestamps through the generic
`TurnRecordProcessor` hook. It runs after transport measurement and credit return, so
slow grading — especially sandboxed code execution — cannot contaminate throughput or
latency. `aiperf_runtime::accuracy_core` (`rust/runtime/src/accuracy_core/`) is the static
evaluator worker seam: only the `AccuracyEvaluator` trait, versioned protocol DTOs, and
the supervised `PythonEvaluator` implementation.

### Static (single-response) evaluator protocol

Problem IDs are opaque. Rust receives only IDs, task labels, prompts/messages, and
generation controls; the worker receives terminal response text only after the ordinary
Rust transport reaches terminal. Expected answers and private tests never cross the
protocol in either direction. The protocol rejects undeclared response fields, and prompt
messages are exactly `{role, content}` rather than an open-ended map. Operations:
`hello`, `load`, `next_problems`, `grade_batch`, `shutdown`.

Grading is batched after the scheduler/transport drain. A worker crash, protocol
violation, or evaluator exception is an infrastructure error that aborts report
construction; it is **never** converted into an incorrect model answer. Inference
transport failures remain explicit failed records in the accuracy denominator, but their
captured partial/empty response text is still scored by the worker; Rust never fabricates
a grading result. The native-v2 report records protocol/worker/Python/package versions,
worker source and dependency-lock SHA-256, optional immutable container digest, canonical
grader name, and dataset repository/subset/revision/splits/task version. The canonical
static worker is the pinned Python/Lighteval evaluator.

### Stateful agentic evaluator vertical — REMOVED (historical record)

**This vertical no longer exists on the `ajc/rust` branch.**
`rust/runtime/src/accuracy_core/mod.rs` states verbatim: "The external-evaluator
provider-host and agentic verticals have been removed; only the static lighteval
worker path remains." There is no stateful evaluator protocol, no
`AgenticHarnessProvider`, no `AgenticWorkload`, no `AgenticInferenceGateway`, no
agentic runner execution, and **no `http + agentic` pair** in the current code. The
description below is retained only as a historical record of what was built and then
removed; nothing in it is present-tense reality.

A capability-gated stateful contract once extended the static protocol with
`load_agentic`, `next_episodes`, `start_episodes`, `poll_agentic`,
`submit_model_results`, `cancel_episodes`, and `finish_agentic`, so an agent could
alternate model inference with evaluator-owned environment work without moving dispatch
into accuracy. The evaluator published model-safe calls with opaque episode/call IDs and
waited for terminal results; the runner issued each call through the ordinary Rust
transport path; Python owned the task, agent scaffold, environment, trajectory, and
verifier. LLM-using environments/verifiers reached the model only through an
authenticated Rust `AgenticInferenceGateway` ingress adapter (never direct Python/sandbox
access), scoped per episode and purpose with a per-run bearer credential.

The stateful worker owned an `AgenticHarnessProvider` factory seam selected by
dataset-name prefix, each provider independently hash-locked and porting **no** benchmark
semantics — replacing only the harness's inner model client with a shared queue-backed
broker/callback LLM:

- **Harbor** (`harbor==0.18.0`, default source form) — Hub packages `org/name@ref`,
  legacy `name@version`, and local task dirs via Harbor's `DatasetConfig`; callback
  `AIPerfCallbackLLM` injected into the Terminus-2 scaffold. Lock digest
  `5ab314ec28af774ed9edf4a6baf5216f8831ecf06eb9bf3b62418bef275b57ef`.
- **AgentLab + BrowserGym** (`agentlab==0.4.2`, `browsergym==0.14.3`; prefix
  `browsergym/`) — ran the exact `ExpArgs.prepare`/`ExpArgs.run` loop with
  `AIPerfAgentLabChatModel`; separately locked (OpenAI-2.x vs <2 conflict with Harbor) at
  digest `2e998cbe869fa6ae21b3ce52264a2cf188316941bb2ebf8e256461a989aedb66`.
- **MCPMark** (commit `cd45b7f57923b9b3985467f5139927575f83141c`, prefix `mcpmark/`) —
  kept MCPMark's evaluator/agent/verifier, replacing only `litellm.acompletion`; lock
  digest `85aed9ad589093de161c8ed00c2dbf64ffea1d06685a96a254c72fa4cf189a59`.

The pinned Harbor canary matrix (SWE-bench Verified, Terminal-Bench 2.1, Aider Polyglot,
GAIA, SkillsBench, BFCL 1.0, `sierra-research/tau3-bench`), a BrowserGym MiniWoB++ run,
and an MCPMark filesystem-service run were once proven end to end. **All of this was
subsequently removed from this branch.** OSWorld / OSWorld-Verified and AppWorld never had
canonical providers.

---

## Product reachability

The runner is **protocol-v2 only**. Runner protocol v1 was fully removed:
`aiperf --execute` advertises `protocol_versions: [2]` and rejects any non-v2 request as a
v2 failure envelope; the v1 `dispatch` entry, `execute_v1`/`execute_run*` chain, the
`RunRequest`/`RunSpec`/`RunTerminal`/`EndpointSpec`/`DatasetSpec`/`AccuracySpec` DTOs, the
`load_protocol_v1` graph-input adapters, the `Legacy` enum variants, and the v1 tests are
gone.

- **Static evaluator-backed accuracy** is defined as the strict protocol-v2
  `http + static_accuracy` pair (`StaticAccuracyWorkloadConfigV2` carries the opaque
  evaluator-authored `accuracy` object; `register_http_static_accuracy_pair` and
  `NativeStaticAccuracyEvaluatorFactory` are present in `rust/runtime/src/engine`). But it is **not
  registered on the default product wire**: `build_default_registry` registers only the
  http/grpc scheduled and graph pairs (`register_http_pairs`,
  `register_http_scheduled_pair`, `register_grpc_pairs`) plus dynosim behind its feature —
  it never calls `register_http_static_accuracy_pair`. Static accuracy is therefore a
  linked-but-off-wire worker seam today, not a product-reachable pair. The
  accumulator/analyzer, the `AccuracyEvaluator`/`PythonEvaluator` seam, and the static
  JSONL protocol all exist; only the default-registry wiring is absent.
- **Stateful agentic accuracy** is **not reachable** — the entire agentic vertical was
  removed (see above). There is no `http + agentic` pair.

The canonical-provider boundary is unchanged: the Python/Lighteval worker owns prompts,
private tests, and grading, and never becomes an alternate inference client. Rust owns
scheduling, transport, measurement, accumulation, analysis, and reporting.

---

## Long-term target (historical)

The long-term evaluator architecture was going to be superseded by
`2026-07-12-external-evaluator-provider-host-boundary-design.md` — a single
provider-neutral evaluation workload in which the selected provider owns dataset, prompt,
agent, environment, verifier, scorer, reducer, and bundle semantics while AIPerf Rust owns
admission, routing, retries, cancellation, accounting, and every external network
operation. That RFC has itself since been removed, and the agentic vertical it would have
subsumed no longer exists on this branch (see above). What remains is the static path: the
accumulator/analyzer and the linked-but-off-wire `aiperf_runtime::accuracy_core` worker seam.

Net: accuracy is no longer a bolted-on side pipeline — it is one more accumulator the
analyzers compose (the same shape as energy, the precondition for every quality×perf×energy
join) — and all benchmark semantics live behind one canonical Python provider boundary rather
than being reimplemented in Rust.
