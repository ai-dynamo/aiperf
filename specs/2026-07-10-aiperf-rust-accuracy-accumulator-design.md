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

The native Rust workspace now implements the core first-class accuracy accumulator/analyzer in `rust/aiperf-metrics/src/accuracy.rs`. The built surface includes typed `AccuracyRecord` / `GradingResult`, real `CorrelationId` and `TaskId` association, phase/time-window summaries via `ExportContext`, Wilson confidence intervals, per-task rollups, `Analyzer` / `SummaryContext` dependency enforcement, `AccuracyResultsAnalyzer`, and optional joins to metric goodput/request-throughput and energy telemetry summaries. The grader plugin zoo and runtime record-routing/exporter wiring remain future consumers of this built leaf crate.

## Addendum — 2026-07-11 (runtime and report wiring)

The runtime-wiring caveat in the preceding addendum is now partly superseded.
`aiperf-accuracy` provides the native MMLU-Pro benchmark/source/grader surface, and
`aiperf::accuracy` dispatches its records through the shared transport observer,
feeds both the performance and accuracy accumulators, runs the dependency-enforced
analyzer join, and emits the joined result in the unified native-v2 report. Additional
benchmark/grader implementations remain future extensions behind the built traits;
energy joins remain absent until telemetry producers exist.

## Addendum — 2026-07-11 (native catalog and normal-pipeline ownership)

The preceding runtime addendum is superseded where it says
`aiperf::accuracy` dispatches records. A complete read of the inherited Python
implementation establishes a stricter ownership boundary:

- `dataset/loader/accuracy_dataset_loader.py:1-150` only converts benchmark
  problems into ordinary `Conversation` / `Turn` values with grading metadata;
- the unchanged timing, worker, endpoint, and transport pipeline dispatches them;
- `accuracy/accuracy_record_processor.py:1-147` grades the parsed terminal
  response downstream; and
- `accuracy/accuracy_results_processor.py:1-168` plus the normal exporters own
  aggregation and presentation.

Native Rust now follows that shape. The accuracy-specific
`AccuracyDispatcher`, `HttpAccuracyDispatcher`, `AccuracyDispatchResult`, and
manual accuracy run loop have been deleted. `aiperf::accuracy::AccuracyDataset`
only lowers benchmark problems into the unified content-addressed dataset and
retains typed correlation/task/ground-truth associations.
`SingleTurnDatasetWorkload` issues those ordinary conversations through
`ScheduledRuntime` and the normal `TurnDispatcher` (`TransportSink` online;
future offline sinks use the same path). The generic `TurnRecordProcessor` hook
runs after transport measurement and credit return; `AccuracyRecordProcessor`
implements it to grade responses without holding an inference concurrency slot.
Performance wall time is bounded by the last transport terminal timestamp, so
slow grading—especially sandboxed code execution—cannot contaminate throughput
or latency.

The built catalog is no longer MMLU-Pro-only: `aiperf-accuracy` registers 11
native benchmarks and 9 native graders, including native symbolic math and
bubblewrap-isolated LiveCodeBench execution. Official cached dataset providers,
row-independent benchmark preflight, unified tokenizer/segment lowering,
typed per-problem native-v2 records, performance joins, console output, and the
inherited accuracy-summary CSV are wired end to end. Telemetry-backed energy
joins remain dormant until telemetry producers exist.

## Addendum — 2026-07-11 (canonical Python/Lighteval evaluator boundary)

This addendum supersedes every preceding claim that benchmark preparation,
answer extraction, symbolic comparison, hidden-test decoding/execution, or
grading should be implemented natively in Rust. Those semantics are too easy to
duplicate incorrectly and must have one canonical owner.

The implemented boundary is:

- Rust owns scheduling, admission, endpoint materialization, inference-server
  HTTP/SSE I/O, response parsing, timing, performance metrics, accuracy
  accumulation, analysis, and reporting.
- One lightweight, long-lived Python worker owns canonical benchmark/dataset
  preparation, prompts, generation settings, private test material, Lighteval
  task execution, and scoring.
- Rust launches and supervises the worker with `tokio::process::Command` and
  exchanges correlated, versioned JSONL over stdin/stdout. Worker diagnostics
  use stderr exclusively. The protocol operations are `hello`, `load`,
  `next_problems`, `grade_batch`, and `shutdown`.
- Problem IDs are opaque. Rust receives only IDs, task labels, prompts/messages,
  and generation controls; the worker receives terminal response text only
  after the ordinary Rust transport reaches terminal. Expected answers and
  private tests never cross the protocol in either direction. Protocol v1
  rejects undeclared response fields, and prompt messages are exactly
  `{role, content}` rather than an open-ended map.
- Grading is batched after the scheduler/transport drain. A worker crash,
  protocol violation, or evaluator exception is an infrastructure error that
  aborts report construction; it is never converted into an incorrect model
  answer. Inference transport failures remain explicit failed records in the
  accuracy denominator, but their captured partial or empty response text is
  still scored by the worker; Rust never fabricates a grading result.
- The native-v2 report records protocol/worker/Python/package versions, worker
  source and dependency-lock SHA-256, optional immutable container digest,
  canonical grader name, and dataset repository/subset/revision/splits/task
  version.

`aiperf-accuracy` is consequently only the `AccuracyEvaluator` trait, protocol
DTOs, and supervised `PythonEvaluator` implementation. The Rust-native
benchmark catalog, prompt builders, graders, symbolic math, code executor,
accuracy registry, and manual Hugging Face dataset HTTP/cache client are
deleted. `aiperf::accuracy::AccuracyRecordProcessor` only captures terminal text
and timestamps through the generic `TurnRecordProcessor` seam; it cannot grade
or dispatch. The existing `aiperf-metrics` accumulator/analyzer remains the
trusted Rust owner of aggregation over canonical worker results.

## Addendum — 2026-07-11 (stateful agentic evaluator boundary)

The preceding fixed `load` → `next_problems` → `grade_batch` operation list is
complete for single-response benchmarks but cannot express an agent that
alternates model inference with evaluator-owned environment work. The evaluator
control plane therefore now also has an optional, capability-gated stateful
contract: `load_agentic`, `next_episodes`, `start_episodes`, `poll_agentic`,
`submit_model_results`, `cancel_episodes`, and `finish_agentic`.

This does not move dispatch into accuracy. `AgenticEvaluator` publishes
model-safe calls with opaque episode/call IDs and waits for terminal results;
the future application adapter must issue each call through the ordinary Rust
`ScheduledRuntime` / `TurnDispatcher` / transport path. Python owns the task,
agent scaffold, environment, trajectory, and verifier. Inference failures are
returned explicitly and become episode infrastructure failures rather than
incorrect answers.

The first concrete harness adapter is pinned `harbor==0.18.0`. A queue-backed
`AIPerfCallbackLLM` implements Harbor's `BaseLLM` interface and is injected into
the canonical Terminus-2 scaffold, so Harbor never contacts the inference
server. The worker records exact Harbor and Python package versions, a digest of
all AIPerf evaluator sources, the fully hashed agentic dependency lock, Harbor's
installed source digest, the resolved immutable dataset revision, environment,
agent, verifier, and optional container digest. Harbor task packages retain
their own setup and verifier semantics; AIPerf does not reproduce SWE-bench or
any other task's scorer.

As of this addendum's first implementation commit, the stateful Python/Rust
protocol and Harbor callback adapter are built and tested. The application
`Workload` adapter, CLI/report surface, and real task-container end-to-end proof
remain explicitly unbuilt; the protocol alone must not be described as a
working agentic benchmark run.

## Addendum — 2026-07-11 (normal Rust agentic workload wiring)

The application-workload caveat in the preceding addendum is now partly
superseded. `aiperf::agentic::AgenticWorkload` implements the generic
`Workload` seam: Rust pages opaque episodes, admits evaluator environments,
polls model-call events, enforces a separate Rust model-concurrency `SlotPool`,
and issues every call with `ScheduledRuntime::issue_turn`. Terminal callbacks
return correlated results to the worker only after the ordinary endpoint,
transport, observer, timing, metric, and credit path reaches terminal.

`AgenticTurnBuilder` is the request-lowering extension seam. Its first
implementation uses the unified dataset composer/materializer, preserving full
message history, generation controls, tool schemas, tool choice, and response
format. It assigns the opaque episode ID as the runtime session and the opaque
call ID as request correlation; neither value is interpreted by Rust.

The shared `TurnDispatchOutcome` now retains endpoint-normalized assistant
content, reasoning, cached prompt tokens, response ID, finish reason, and typed
transport error details in addition to the existing continuation text and
usage. That metadata is returned verbatim to the agent harness. Failed,
rejected, and cancelled requests remain infrastructure outcomes; they are never
converted into verifier rewards by Rust.

Executable proof includes exact request-body lowering and a real loopback
HTTP/SSE test in which a stateful evaluator call traverses `TransportSink` and
returns parsed content/ID/finish/usage. The CLI/native-v2 agentic report surface
and a real Harbor task-container run remain unbuilt at this point; those are
still required before describing SWE-bench or another Harbor package as end-to-
end supported.

## Addendum — 2026-07-11 (agentic CLI and native-v2 reporting)

The CLI/report caveat in the preceding addendum is now superseded.
`--agentic-benchmark` launches the supervised worker, resolves and freezes an
immutable Harbor package or local task directory before measurement, and runs
the resulting `AgenticWorkload` through the generic `run_scheduled_online`
composition boundary. Accuracy code still does not create or call an HTTP
client: that runner assembles the same `TransportSink`, phase runner, observers,
credits, timing, metrics, and endpoint path used by ordinary scheduled runs.

The native-v2 report now carries both layers of provenance: the generic worker
block records protocol, Python, package, worker-source, dependency-lock, and
container identities plus the resolved dataset revision; the typed `agentic`
block records exact harness source/version, agent scaffold, environment,
verifier, authored run configuration, reward aggregates, and every opaque
episode result/artifact path. Rust only aggregates finite canonical verifier
values. Completed episodes form reward denominators; environment, harness,
verifier, inference, and cancellation outcomes are counted and emitted as
explicit report errors, never rewritten as zero or incorrect scores.

CLI acceptance proof uses a supervised stateful JSONL worker and a real
loopback OpenAI SSE endpoint. One episode makes multiple correlated turns, all
requests preserve evaluator-authored messages/generation/tools through the
ordinary Rust transport, exact response metadata returns to the worker, and a
second sandbox failure remains outside the primary-score denominator. A real
Harbor task-container run is still required before claiming SWE-bench or any
other packaged benchmark has been proven end to end in this environment.

## Addendum — 2026-07-11 (real Harbor registry, Docker, and verifier proof)

The final generic-Harbor caveat above is now superseded. The opt-in
`rust/aiperf/tests/agentic_harbor_e2e.rs` acceptance test uses the real pinned
worker environment and Harbor's live package registry. It resolves
`harbor/hello-world` to immutable dataset revision
`sha256:d10e96e201d6816b22553504e06e7de0153a26381e808d11404cbca530b9d388`,
starts the package's real Docker environment, runs the inherited Terminus-2
agent, sends every model turn through the compiled Rust CLI's ordinary
OpenAI-SSE transport, executes the package's verifier, and requires a completed
finite canonical reward plus Harbor trial/verifier artifacts in native v2.

The proof also pins the worker dependency-lock digest
`5ab314ec28af774ed9edf4a6baf5216f8831ecf06eb9bf3b62418bef275b57ef`,
`harbor==0.18.0`, the callback agent identity, environment, verifier identity,
and equality between captured Rust HTTP calls and the episode's reported model
call count. Its deliberately non-solving model response can earn zero; the
acceptance criterion is canonical verification, never a fabricated Rust score.
The test is ignored by default because it requires registry access and a Docker
daemon; run it explicitly with:

```bash
AIPERF_AGENTIC_PYTHON=/path/to/pinned/agentic/python \
  cargo test -p aiperf --test agentic_harbor_e2e -- --ignored --nocapture
```

This proves the generic packaged Harbor path through a real sandbox and
verifier. It does not yet prove a SWE-bench task or the cross-family benchmark
matrix; those named canaries remain required before claiming the broader
agentic-evaluation objective complete.

## Addendum — 2026-07-11 (canonical Harbor source resolution)

The preceding package-only wording is superseded. The pinned Harbor 0.18
worker now accepts all three source forms owned by Harbor's public
`DatasetConfig`: content-addressed Hub packages as `org/name@ref`, legacy
registry datasets as `name@version`, and existing local task directories.
Hub refs are resolved to and reported as immutable dataset `sha256:` content
hashes. An omitted legacy version is resolved by Harbor's registry policy and
then written back into `DatasetConfig` before the ordered task list is fetched,
so task enumeration cannot silently move to a different version.

This is a generic source adapter, not a benchmark port. In particular,
`bfcl@1.0` resolves through Harbor to the canonical 3,641-task BFCL dataset and
its Harbor-authored task configs; AIPerf does not decode functions, implement a
tool loop, or score calls. Pinned-worker tests cover package, explicit/default
legacy, and local resolution, and a live registry probe freezes BFCL 1.0's
first opaque episode. A real BFCL sandbox/verifier run and the broader
cross-family canary matrix remain required.

## Addendum — 2026-07-11 (auxiliary inference and Harbor family proof)

This addendum supersedes two stale caveats above: the cross-family Harbor
matrix has now run, and packaged environments/verifiers that themselves use an
LLM no longer require direct Python or sandbox access to the inference server.
The ownership boundary remains strict:

- Primary Terminus calls continue over the versioned JSONL
  `AgenticEvaluator` protocol.
- Canonical task containers and verifier processes that only know the OpenAI
  wire format receive an episode- and purpose-scoped callback URL plus a
  per-run bearer credential. Python only injects `OPENAI_BASE_URL` /
  `OPENAI_API_KEY` through Harbor's `EnvironmentConfig.env` and
  `VerifierConfig.env`; it does not send or receive the model request.
- The authenticated Rust `AgenticInferenceGateway` is an ingress adapter, not
  a forwarding client. It parses lossless message/tool history and generation
  fields into an `AgenticModelCall`, sends that value over an in-process
  channel to `AgenticWorkload`, and waits. The workload acquires the same
  Rust-owned model `SlotPool`, calls the same `AgenticTurnBuilder`, and invokes
  the same `ScheduledRuntime::issue_turn` used by primary agent calls. The
  configured endpoint, `TransportSink`, SSE parser, observers, metrics, timing,
  and report path remain the sole inference data plane.
- Endpoint-native assistant-message reconstruction preserves tool calls and
  tool-call arguments on the return path. Worker capability
  `agentic_inference_gateway` gates the extension; inactive/late episodes,
  gateway failure, transport failure, and caller disconnects fail as
  infrastructure rather than becoming verifier zeros.
- Rust waits for canonical `finish_agentic` results to match the terminal
  events byte-for-byte before adding Rust-owned primary/environment/verifier
  call and token accounting. Native v2 reports the total and each subset while
  omitting the bearer credential.

Executable proof covers each layer. A deterministic CLI test makes real
environment and verifier HTTP calls into the Rust ingress, observes both calls
on the ordinary outbound OpenAI/SSE server, round-trips a fragmented native
tool call, and checks exact purpose/token/report accounting. The pinned real
tau3 task
`sierra-research/tau3-bench__tau3-telecom-mobile-data-issue-bad-network-preference-bad-vpn-user-abroad-roaming-disabled-on-persona-none`
at dataset revision
`sha256:a57304f682894ac061090769af771a3617664f3ff6e5417d4eadf8e30433e4d9`
then invokes its packaged `start_conversation` MCP tool. The resulting
LLM-backed user-simulator request returns through Rust and the report proves
four normal-pipeline calls: three primary, one environment, zero verifier. The
packaged verifier completes with canonical reward `0.0`; Rust does not improve
or reinterpret it.

The broader pinned Harbor 0.18 canary matrix has also completed real package,
Docker, and packaged-verifier runs for SWE-bench Verified, Terminal-Bench 2.1,
Aider Polyglot, GAIA, SkillsBench, and legacy BFCL 1.0. Each report pins the
resolved dataset revision/task, `harbor==0.18.0`, and dependency-lock digest
`5ab314ec28af774ed9edf4a6baf5216f8831ecf06eb9bf3b62418bef275b57ef`.
These are generic Harbor task packages; no SWE-bench, BFCL, tau3, or other
benchmark-specific prompt, tool loop, environment, test decoder, executor, or
scorer exists in Rust. Harness families not present in Harbor 0.18 require a
separate canonical adapter and may not be claimed through this implementation.

## Addendum — 2026-07-11 (AgentLab/BrowserGym canonical browser harness)

The final sentence above remains the governing rule and now has a second
implementation. The stateful worker owns an `AgenticHarnessProvider` factory
seam rather than hard-coding Harbor. Dataset names beginning with
`browsergym/` select pinned `agentlab==0.4.2` (upstream commit
`367d4e8a9c2cd97eab4524f6898ac98010fc99a8`) plus `browsergym==0.14.3`
(upstream commit `0a785fbed075224ae81ca9c1fe924f66050696fe`); every other existing
source form remains Harbor-owned. The generic worker capability is `agentic`,
with `agentic_browsergym` and `agentic_harbor` reporting the installed provider.
Rust retains legacy `agentic_harbor` handshake compatibility.

The adapter does not port browser prompts, action parsing, environments, or
evaluators. It directly instantiates BrowserGym's `DEFAULT_BENCHMARKS`, adapts
AgentLab's official `FLAGS_GPT_4o` / `FLAGS_GPT_4o_VISION` profile through
`GenericAgentArgs.set_benchmark`, and runs the exact `ExpArgs.prepare` /
`ExpArgs.run` loop. AgentLab therefore still owns observation preprocessing,
prompt construction and token fitting, parser retries, action generation,
Playwright reset/step/cleanup, trajectory artifacts, and
`summary_info.json`. BrowserGym still owns task lists, fixed seeds, action sets,
step limits, and rewards; its WebArena-Verified task still calls
`WebArenaVerifiedEvaluatorAPI` over the captured network trace. AIPerf's only
model implementation is `AIPerfAgentLabChatModel`, an `AbstractChatModel` that
blocks the synchronous environment thread on the same queue-backed
`ModelCallBroker` used by Harbor. It has no HTTP client.

Rust remains the sole inference data plane. The complete AgentLab-authored
message list, including screenshot data URLs, becomes a JSONL `model_call`.
`AgenticWorkload` acquires its Rust model slot and lowers it through the normal
dataset materializer, endpoint, `ScheduledRuntime`, `TransportSink`, and SSE
parser. Only Rust's submitted terminal result unblocks AgentLab. Transport or
worker failure becomes an infrastructure result; a canonical completed reward
of zero remains a model score. BrowserGym is run sequentially
(`task_concurrency=1`) in a stable topological order derived from its task
dependency graph, preserving stateful WebArena/VisualWebArena dependencies
without recreating their scheduler semantics.

The pinned BrowserGym registry exposes MiniWoB, WebArena, WebArena-Verified,
WebArena Lite, VisualWebArena, WorkArena L1/L2/L3 curricula, AssistantBench, and
WebLINX through identifiers such as `browsergym/webarena_verified@0.14.3`.
Exact selected `EnvArgs`, metadata, package versions, source files, and task
seeds are content-digested into report provenance. Backend services and
Playwright browsers remain benchmark prerequisites owned by BrowserGym.

Harbor and AgentLab cannot share one Python environment: Harbor 0.18's LiteLLM
requires OpenAI 2.x while AgentLab 0.4.2 requires OpenAI below 2. The two
provider environments are therefore independently hash-locked. Browser runs
record `requirements/browser-agentic-accuracy-worker.txt` digest
`2e998cbe869fa6ae21b3ce52264a2cf188316941bb2ebf8e256461a989aedb66`;
Harbor retains digest
`5ab314ec28af774ed9edf4a6baf5216f8831ecf06eb9bf3b62418bef275b57ef`.
Rust still launches one long-lived worker per run, and reports its exact Python,
package, worker-source, lock, and optional container identity.

Executable proof has two levels. The Python suite uses the real pinned registry
and a real local MiniWoB++ checkout at revision
`7fd85d71a4b60325c6585396ec4f48377d049838`; the canonical environment returns
reward `1.0`. `rust/aiperf/tests/agentic_browsergym_e2e.rs` then runs the
compiled Rust CLI against that environment and a loopback OpenAI-SSE server. It
requires one captured streaming request with `include_usage`, exact full
messages, Rust token/call accounting, AgentLab/BrowserGym provenance, canonical
reward `1.0`, and `summary_info.json`. Its frozen selected task revision is
`sha256:69ae6bd5d03cb41821df06d488bd986d2e041a8f427d385a7a25fb7415f27c27`.

This addendum proves the pinned BrowserGym family, not every external agentic
benchmark. OSWorld/OSWorld-Verified, AppWorld, and MCPMark are not in
BrowserGym 0.14.3 or Harbor 0.18 and still require their own canonical provider
adapters before they may be claimed.

## Addendum — 2026-07-11 (canonical MCPMark Verified provider)

The MCPMark caveat immediately above is now superseded. Dataset names beginning
with `mcpmark/` select a third independently locked `AgenticHarnessProvider`
backed by MCPMark commit `cd45b7f57923b9b3985467f5139927575f83141c`
(`MCPMark==0.0.1`, LiteLLM 1.80.0). The direct source archive is hash-pinned in
`requirements/mcpmark-agentic-accuracy-worker.txt`; the complete lock digest is
`85aed9ad589093de161c8ed00c2dbf64ffea1d06685a96a254c72fa4cf189a59`
and the installed MCPMark `src/**/*.py` digest is
`55bc1d0e43043101d4eed5b76d97c2efb14c3415e9a4c7e7b74cdc8f81fb21f2`.
This worker is separate because MCPMark's Pixi dependency pins Click below 8
and cannot share the static Lighteval/DeepEval environment.

The adapter ports no MCPMark semantics. `MCPEvaluator` still owns setup,
execution, verification, artifact writing, and cleanup
(`src/evaluator.py:181-294`); `MCPMarkAgent` still owns the system prompt,
100-turn loop, 32,768-token generation limit, temperature 1.0, tool history,
retry policy, and MCP calls (`src/agents/mcpmark_agent.py:768-1099`); its exact
stdio/HTTP MCP server definitions remain at
`src/agents/mcpmark_agent.py:1102-1243`; and task selection plus verifier exit
codes remain authoritative in `src/base/task_manager.py:132-245`. The adapter
replaces only `litellm.acompletion` with the shared `ModelCallBroker`. It
initializes MCPMark with the real target model so canonical model-specific tool
schema handling remains active, substitutes a non-secret credential sentinel,
and never gives Python a model-server URL or inference client.

Each MCPMark call therefore crosses JSONL with complete messages, canonical
generation controls, tool schemas, tool choice, and provider extras. Rust alone
admits it through `AgenticWorkload`, materializes the ordinary Chat request,
sends and streams it through `ScheduledRuntime` / `TransportSink`, reconstructs
assistant tool calls in the endpoint parser, records timing/usage/metrics, and
submits the terminal result. MCPMark then invokes its real MCP server and later
its task-local verifier. A Rust transport failure is tagged through the agent
loop and reported as infrastructure; state setup failure, verifier exception,
worker failure, and a zero-call agent failure are also infrastructure. A
normally completed verifier exit 1 remains canonical score `pass=0.0`, never an
infrastructure rewrite.

`mcpmark/<service>[/standard|easy]@<commit>` resolves the exact upstream task
registry and forces sequential task environments because MCPMark uses process
globals. Selected descriptions, metadata, and verifiers are content-digested.
For the filesystem service, MCPMark's own category preparation runs before the
measurement clock and AIPerf additionally digests every concrete environment
file, its relative path, content, and modification timestamp; that state digest
is part of the dataset revision. Hosted MCPMark services retain their canonical
credential and service prerequisites. MCPMark's non-overridable controls are
reported in `canonical_agent_config`; validated common values override generic
CLI defaults in the effective native-v2 `agentic.config`, so reports do not
claim the unused 4,096-token generic default.

Executable proof uses the real pinned worker, MCPMark's
`file_property/size_classification` task, the exact
`@modelcontextprotocol/server-filesystem@2025.12.18` stdio server, and the
shipped `verify.py`. The compiled Rust CLI serves three normal streaming model
turns: discover the allowed directory, issue the canonical MCP tools that
classify all eight files, and finish. The verifier checks directory creation,
exact membership, byte-size ranges, empty root, and total count, then returns
`pass=1.0`. `rust/aiperf/tests/agentic_mcpmark_e2e.rs` requires exactly three
captured Rust HTTP requests with `include_usage`, model/tool history, 32,768
max tokens, temperature 1.0, the exact worker/source/lock identity, concrete
environment digest, MCPMark artifacts, and canonical reward 1.0:

```bash
AIPERF_MCPMARK_AGENTIC_PYTHON=/path/to/pinned/mcpmark/python \
AIPERF_MCPMARK_FILESYSTEM_ROOT=/path/to/prepared/environments \
  cargo test -p aiperf --test agentic_mcpmark_e2e -- --ignored --nocapture
```

This proves MCPMark's canonical provider and filesystem verifier path. It does
not claim OSWorld/OSWorld-Verified or AppWorld; those still require their own
canonical providers and proofs.

## Addendum — 2026-07-11 (accuracy product reachability after native CLI removal)

Static evaluator-backed accuracy is product-reachable through
`aiperf-runner` protocol v1 and retains a real subprocess proof: Rust loads
opaque evaluator-authored problems, owns normal inference transport, and sends
terminal response text to the supervised Python grader.

Stateful agentic execution remains implemented in the `aiperf` library, but it
is not represented in the current runner request DTO. Deleting the native Rust
CLI therefore removes the end-user route and the compiled-CLI Harbor,
BrowserGym, and MCPMark canaries described above. Those provider claims now
mean library/provider implementation evidence, not current product
availability. A future runner protocol addition must restore primary and
callback call routing, provenance/reward reports, cancellation semantics, and
real provider subprocess canaries before agentic accuracy is again described as
product-reachable. This addendum does not weaken the canonical-provider rule or
authorize reimplementing any evaluator in Rust.

## Addendum — 2026-07-11 (runner-only stateful-agentic projection)

`2026-07-11-aiperf-runner-only-execution-surface-design.md` is authoritative
for restoring the product route identified above. It defines `agentic` as a
strict runner workload over the `online_http` backend, including provider-worker
supervision, authenticated callback-gateway lifecycle, primary/environment/
verifier admission through the shared Rust endpoint/transport path, typed
native-v2 results, and real Harbor, BrowserGym, and MCPMark subprocess canaries.

The canonical-provider boundary in this spec remains unchanged. Python workers
continue to own prompts, agents, environments, private tests, tool loops, and
rewards; they do not become an alternate inference client. Agentic capability
is product-reachable only when the selected runner advertises the workload and
passes the provider subprocess matrix.

## Addendum — 2026-07-12 (provider-neutral long-term target)

This specification remains authoritative for the built/current static and
stateful accuracy implementation. Its descriptions of the existing
accumulator, analyzer, supervised evaluator workers, static path, agentic path,
reports, and product-reachability gates remain code truth where the code still
implements them.

The long-term evaluator architecture is superseded by
`2026-07-12-external-evaluator-provider-host-boundary-design.md`. That RFC
converges static and agentic accuracy on one provider-neutral evaluation
workload. The selected evaluator provider owns dataset, prompt, agent,
environment, verifier, scorer, reducer, and canonical bundle semantics. AIPerf
Rust owns admission, routing, retries, cancellation, accounting, and every
upstream/external network operation, reached through typed pipes or an
authenticated, per-run scoped Rust-owned MITM compatibility proxy.

This target does not delete or silently redirect the current implementations.
Legacy static and agentic workloads, providers, protocols, and report
projections remain until the new RFC's provider-specific semantic-parity,
subprocess, report-consumer, and exact deletion gates are satisfied.

## Addendum — 2026-07-12 (runner protocol-v1 fully removed)

The runner's protocol-v1 support has been deleted. `aiperf-runner` now advertises
`protocol_versions: [2]` only and rejects any non-v2 request as a protocol-v2
failure envelope. Removed from `rust/aiperf-runner`: the v1 request `dispatch`
entry, `execute_v1` and the `execute_run*` chain, the `RunRequest` / `RunSpec` /
`RunTerminal` / `EndpointSpec` / `DatasetSpec` / `AccuracySpec` wire DTOs, the
`load_protocol_v1` graph-input adapters, and the `Legacy` enum variants, plus the
v1 tests.

This supersedes the 2026-07-11 "accuracy product reachability" addendum's claim
that static evaluator-backed accuracy is reachable "through `aiperf-runner`
protocol v1": static accuracy is now product-reachable through the same strict
protocol-v2 `online_http + static_accuracy` pair. The canonical-provider
boundary and the supervised Python grader/JSONL protocol are unchanged; only the
runner wire protocol that carries the authored request moved from v1 to v2-only.
