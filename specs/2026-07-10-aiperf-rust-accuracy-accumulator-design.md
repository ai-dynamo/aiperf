# AIPerf-Rust: `AccuracyAccumulator` + `AccuracyResultsAnalyzer`

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design
**Companion:** analyzer/accuracy findings in memory `#15927949`;
`2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md`
**Pattern to mirror:** `analysis/energy_analyzer.py` (`EnergyEfficiencyAnalyzer`) on
the `ajc/metrics-accumulator` branch — the reference for a clean analyzer.

---

## The problem

Accuracy today is a self-contained side-pipeline (confirmed): `MetricsAccumulator`
has **zero** accuracy references; the `accuracy_results` "analyzer" is a
`NotImplementedError` stub with the wrong signature; association is a fragile
`session_num % len(tasks)` positional mapping requiring sequential sampling; and the
rich `GradingResult` (confidence / reasoning / extracted_answer) is discarded — only
two floats survive. Because accuracy isn't an accumulator, it gets no timeslicing, no
steady-state windowing, and no sweep/energy joins.

**Fix:** make accuracy a first-class **accumulator + analyzer pair**, exactly like
energy. Then accuracy-over-time, accuracy-under-load, and accuracy×perf×energy joins
fall out for free.

---

## Data model — carry the full grading result, keyed by a real id

```rust
/// One graded response. record_type = "accuracy_records".
/// correlation_id is the REAL per-request id (from the SchedulingPolicy /
/// dataset seam), NOT session_num % len(tasks).
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

Grading still happens in the record-processing stage (a `Grader` trait — port the
existing benchmark/grader plugin zoo: exact-match, multiple-choice, math/sympy,
lighteval, code-execution). The processor emits `AccuracyRecord` onto the bus with
`record_type = "accuracy_records"`; the routing table fans it to the accumulator.

**One correctness policy, defined once:** the `0.5` lighteval threshold and the
correct/incorrect decision live in a single `AccuracyPolicy` constant, not
duplicated across grader + bucketer.

---

## `AccuracyAccumulator` — owns the `accuracy_records` type

```rust
impl Accumulator for AccuracyAccumulator {
    const RECORD_TYPE: &str = "accuracy_records";

    fn process_record(&mut self, r: AccuracyRecord) {
        // columnar, like MetricsAccumulator: push into per-task GrowableArrays
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
        // unparsed rate, n, Wilson/Clopper-Pearson CI on the pass rate.
        AccuracySummary { overall, per_task, unparsed_rate, ci, n }
    }
}
```

The accumulator is columnar (numpy/`ndarray`-style) precisely so
`compute_results_for_mask(window)` works the same way perf metrics do — accuracy
now participates in timeslicing and steady-state windowing.

---

## `AccuracyResultsAnalyzer` — mirror `EnergyEfficiencyAnalyzer`

```rust
impl Analyzer for AccuracyResultsAnalyzer {
    // REAL dependency declaration — enforced by a toposort in RecordsManager,
    // NOT the dead ClassVar Python declares-but-never-reads.
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

The key move: because accuracy, `metric_results`, and `gpu_telemetry` are **all
accumulators under one `SummaryContext`**, the analyzer can read them together and
emit joins that the current side-pipeline structurally cannot produce:

- **accuracy-over-time** (per timeslice) — is quality degrading as KV-cache fills?
- **accuracy-under-load** (steady-state window only) — quality at the SLA operating point.
- **quality-at-goodput / accuracy-per-watt** — the frontier that unifies the three axes.

Single-input accuracy (no joins) still works when only the accuracy accumulator is
present — same graceful-degradation contract as energy.

---

## Fix the dead-metadata blemish while porting

Python declares `required_accumulators` / `summary_dependencies` ClassVars but never
reads them (ledger §energy gap): ordering is incidental insertion order, deps are
re-checked imperatively inside each analyzer. **In Rust, make `REQUIRED` real:**
`RecordsManager` (or its Rust equivalent) toposorts analyzers by `REQUIRED` before
`summarize`, skips an analyzer whose required accumulators are absent, and runs
analyzer→analyzer dependencies in dependency order (so an analyzer can consume
another analyzer's output deterministically). This is ~20 lines of `petgraph`
toposort and removes a whole class of "works by luck of insertion order" bugs.

---

## Migration shape (Python side, if landing there first)

1. Add `AccuracyAccumulator` (`record_type = accuracy_records`) to the `accumulator`
   category; grader emits `AccuracyRecord` with a real correlation id.
2. Rewrite `AccuracyResultsProcessor` → `AccuracyResultsAnalyzer(AnalyzerProtocol)`
   with `required_accumulators = {accuracy}` and `summarize(self, ctx)` (delete the
   `NotImplementedError` stub and the no-arg signature).
3. Delete the `session_num % len(tasks)` mapping and the `accuracy.` string-prefix
   coupling; carry `task`/`ground_truth` on the record.
4. Wire the toposort so `required_accumulators` is enforced, not decorative.
5. Exporters read `AccuracyAnalysis` (typed) instead of re-filtering
   `results.records` by tag prefix.

Net: accuracy stops being a bolted-on side pipeline and becomes one more accumulator
the analyzers compose — the same shape as energy, and the precondition for every
quality×perf×energy join.

## Addendum — 2026-07-11

The statement that the Python `accuracy_results` analyzer is a pure
`NotImplementedError` stub is stale. The inherited Python tree now has concrete
accuracy result processing and metric types. The architectural problem this spec
addresses still stands for the native Rust rewrite: accuracy should become a
first-class accumulator/analyzer integrated with phase windows, joins, and the main
reporting pipeline rather than remaining a side pipeline with fragile positional
association.

## Addendum — 2026-07-11 (native Rust implementation)

The native Rust workspace now implements the core first-class accuracy accumulator/analyzer in `crates/aiperf-metrics/src/accuracy.rs`. The built surface includes typed `AccuracyRecord` / `GradingResult`, real `CorrelationId` and `TaskId` association, phase/time-window summaries via `ExportContext`, Wilson confidence intervals, per-task rollups, `Analyzer` / `SummaryContext` dependency enforcement, `AccuracyResultsAnalyzer`, and optional joins to metric goodput/request-throughput and energy telemetry summaries. The grader plugin zoo and runtime record-routing/exporter wiring remain future consumers of this built leaf crate.

## Addendum — 2026-07-11 (runtime and report wiring)

The runtime-wiring caveat in the preceding addendum is now partly superseded.
`aiperf-accuracy` provides the native MMLU-Pro benchmark/source/grader surface, and
`aiperf::accuracy` dispatches its records through the shared transport observer,
feeds both the performance and accuracy accumulators, runs the dependency-enforced
analyzer join, and emits the joined result in the unified native-v2 report. Additional
benchmark/grader implementations remain future extensions behind the built traits;
energy joins remain absent until telemetry producers exist.
