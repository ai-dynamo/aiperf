import type { SlideDefinition } from "../../core/types";

type StepInput = Omit<SlideDefinition, "narration">;

const STEPS: readonly StepInput[] = [
  {
    eyebrow: "1 · System",
    title: "Product landscape: author, execute, target",
    lede:
      "The atlas starts at product boundaries. `aiperf` from `aiperf-cli` authors Config v2, re-execs itself as `--execute`, and optionally launches `--cell` children. Load targets are real servers, `aiperf-mock-server`, or in-process Dynosim.",
    term: {
      word: "execute child",
      meaning: "Same `aiperf` binary started with hidden `--execute` flags; the only process that dispatches measured load.",
    },
    points: [
      "`rust/cli/src/dispatch.rs` routes public commands.",
      "`rust/cli/src/execute_mode.rs` intercepts internal modes before clap.",
      "Mock server is a separate target; Dynosim is a transport feature, not a command.",
    ],
    caption: "Author → execute → target · artifacts fan out after the run.",
  },
  {
    eyebrow: "2 · Processes",
    title: "Crate graph and process roles",
    lede:
      "Compile-time capability flows `aiperf-cli` → `aiperf-runtime` → `loadgen-core`. The entry process and its `--execute` child share one binary; `aiperf-mock-server` links runtime independently; `pyext` and `e2e` stay off the hot path.",
    term: {
      word: "aiperf-runtime",
      meaning: "Library crate that owns clocks, transports, datasets, engine, cellular, and metrics composition.",
    },
    points: [
      "Workspace members: loadgen-core, runtime, cli, mock-server, e2e, pyext.",
      "Solid edges are Cargo deps or self re-exec; dashed edges are network or optional features.",
      "Evidence: `rust/cli/Cargo.toml`, `rust/runtime/Cargo.toml`, `rust/loadgen-core/`.",
    ],
    caption: "cli → runtime → loadgen-core · mock-server is peer, not child.",
  },
  {
    eyebrow: "3 · Runtime",
    title: "One request from bootstrap to commit",
    lede:
      "Startup freezes registries in `Application`, then `Coordinator` validates and prepares factories. Phase runtime paces work; `RequestSink` dispatches transport-native requests; observers accumulate; exporters commit `native-v2.json`.",
    term: {
      word: "Coordinator",
      meaning: "Engine composition root: validate → prepare → execute → persist for one BenchmarkRun.",
    },
    points: [
      "`rust/runtime/src/engine/application.rs` freezes factories once.",
      "`rust/runtime/src/engine/coordinator.rs` owns the run spine.",
      "`rust/loadgen-core/src/sink.rs` defines RequestSink and RequestObserver.",
    ],
    caption: "Registries at startup · Clock + sink on the hot path · merge after drain.",
  },
  {
    eyebrow: "4 · Protocol",
    title: "Self-exec protocol v2 lifecycle",
    lede:
      "Profile resolves `current_exe()`, spawns `aiperf --execute`, writes one strict protocol-v2 envelope to stdin, and waits for exactly one terminal JSONL line on stdout. stderr stays diagnostics-only.",
    term: {
      word: "protocol v2",
      meaning: "Strict serde envelope wrapping the authored BenchmarkRun; unknown fields fail closed.",
    },
    points: [
      "`rust/cli/src/exec_bin.rs` resolves the child binary override.",
      "`rust/cli/src/execute.rs` is the parent protocol client.",
      "Child bootstrap composes Application, then Coordinator validate|execute.",
    ],
    caption: "stdin: one envelope · stdout: one terminal · parent stays the shell.",
  },
  {
    eyebrow: "5 · Scheduled",
    title: "Paced workload path",
    lede:
      "Scheduled runs load conversations, apply arrival policy, admit through SlotPool, and dispatch PreparedTurns over HTTP, gRPC, or Dynosim. Worker topology is one LocalSet co-located sink or OS-thread sub-cells.",
    term: {
      word: "phase_runtime",
      meaning: "Shared warmup → profiling lifecycle with ramp, grace, cancellation, and drain.",
    },
    points: [
      "`rust/runtime/src/phase_runtime.rs` and `scheduled.rs` bridge policy to turns.",
      "Request-rate, concurrency, user-centric, and fixed-schedule share one path.",
      "Transport choice does not fork separate workload registrations.",
    ],
    caption: "dataset → phase → admit → TurnDispatcher → RequestSink.",
  },
  {
    eyebrow: "6 · Graph",
    title: "Trace compile, phase rewrite, execute",
    lede:
      "Graph inputs (`dag_jsonl`, WEKA, Dynamo) compile once into a program plus SegmentStore. Warmup and profiling derive phase programs around a seeded frontier, then the graph executor fires nodes through the same RequestSink seam.",
    term: {
      word: "GraphInputBundle",
      meaning: "Compiled program plus shared SegmentStore produced by the graph input resolver.",
    },
    points: [
      "`rust/runtime/src/engine/graph_input.rs` selects and strictly decodes sources.",
      "`rust/runtime/src/engine/graph_phase_runtime.rs` derives warmup/profiling programs.",
      "`rust/runtime/src/graph/executor.rs` owns firing gates and dependencies.",
    ],
    caption: "Compile once · rewrite phases · dispatch per node.",
  },
  {
    eyebrow: "7 · Endpoints",
    title: "Dialect preparation and wire binding",
    lede:
      "Endpoint registries validate authored profiles into dense EndpointKeys. Each worker calls prepare_worker once to build a local table; dialects format BodyPlans and bind HTTP or gRPC wire forms without per-token registry lookups.",
    term: {
      word: "PreparedEndpointTable",
      meaning: "Worker-local dense lookup from EndpointKey to tokenizer, binder, and parser state.",
    },
    points: [
      "`rust/runtime/src/endpoints/` owns dialect factories and traits.",
      "OpenAI, Anthropic, KServe, Riva, and specialized families register as factories.",
      "HTTP and gRPC share identity, prepare different bindings.",
    ],
    caption: "validate once · prepare per worker · format per turn.",
  },
  {
    eyebrow: "8 · Metrics",
    title: "Observer stream to exporters",
    lede:
      "Transports emit RequestObserver events; worker-local collectors and metrics observers accumulate exact rows or t-digest sketches. After drain, MetricsAccumulator merges stores, joins side channels, commits the report, and exporters fan out.",
    term: {
      word: "native-v2.json",
      meaning: "Authoritative typed report commit before optional exporter side outputs.",
    },
    points: [
      "`rust/runtime/src/metrics.rs` adapts observers into the catalog.",
      "`rust/runtime/src/metrics_core/` accumulates and folds.",
      "`rust/runtime/src/export/` registers JSON, CSV, Parquet, OTLP, MLflow, W&B sinks.",
    ],
    caption: "hot-path events · post-drain merge · durable report + fan-out.",
  },
  {
    eyebrow: "9 · Cellular",
    title: "Controller, cells, hierarchical merge",
    lede:
      "`cells > 1` promotes the execute child into a controller. Cells fetch sliced envelopes over Velo, run the ordinary engine, and return records or folded stores. Optional aggregators merge subtrees; the controller commits once.",
    term: {
      word: "cellular feature",
      meaning: "Cargo feature that enables Velo-backed cross-host cell transport (`aiperf-runtime/cellular`).",
    },
    points: [
      "`rust/runtime/src/engine/cellular_controller.rs` owns launch and merge.",
      "`rust/runtime/src/engine/cellular_cell.rs` runs the sliced execute path.",
      "SLURM and Kubernetes launchers reuse the same controller/cell model.",
    ],
    caption: "promote → slice → ordinary execute → merge → one commit.",
  },
  {
    eyebrow: "10 · Builds",
    title: "Feature graph defines capability",
    lede:
      "Default `aiperf-cli` enables `grpc` and `cellular`. Orthogonal features add `parquet`, `dynosim`, `pyo3-embed`, and `search-pyo3`. `full` unions dynosim + parquet + cellular + grpc. Missing capabilities fail closed at validation.",
    term: {
      word: "fail closed",
      meaning: "Authored transports, artifacts, or cell counts unavailable in the linked image are rejected during validation.",
    },
    points: [
      "Evidence: `rust/cli/Cargo.toml` and `rust/runtime/Cargo.toml`.",
      "Dynosim pulls the pinned Dynamo mocker git dep when the feature is on.",
      "No runtime plugin discovery—capabilities are statically linked factories.",
    ],
    caption: "Cargo features freeze the implementation universe.",
  },
  {
    eyebrow: "11 · Seams",
    title: "Extensions and execution substitution",
    lede:
      "AIPerfExtension registers datasets, endpoints, transports, workloads, and exporters into a frozen AIPerfRegistry. Execution substitutes Clock and RequestSink implementations; cellular mode scales by wrapping the same single-run core.",
    term: {
      word: "AIPerfRegistry",
      meaning: "Transactional, frozen capability table composed once per Application bootstrap.",
    },
    points: [
      "`rust/runtime/src/extensions/` owns registration contracts.",
      "loadgen-core RequestObserver stays transport-neutral and !Send-friendly.",
      "Workloads resolve an execution factory from the prepared transport—no pair matrix.",
    ],
    caption: "compile-time composition · Clock/Sink substitution · cellular wraps the core.",
  },
] as const;

const NARRATION = [
  "The atlas opens at product boundaries. One aiperf binary authors the run, re-executes itself to dispatch load, and targets a real server, the mock server, or Dynosim.",
  "Capability flows from aiperf-cli into aiperf-runtime and then loadgen-core. Mock server and packaging crates stay outside that execute dependency edge.",
  "One request path freezes registries, validates through Coordinator, paces phases, dispatches on RequestSink, observes tokens, and commits native-v2.json.",
  "Protocol version two is a fresh process boundary: parent writes one envelope to stdin and reads one terminal JSONL line from stdout.",
  "Scheduled work loads conversations, applies arrival policy, admits through SlotPool, and places PreparedTurns on worker-local sinks.",
  "Graph traces compile once into a program and segment store, derive warmup and profiling programs, then fire nodes through the same sink seam.",
  "Endpoint dialects validate once, prepare a dense worker table, and format BodyPlans into HTTP or gRPC bindings without registry churn on the hot path.",
  "Measurement is an observer event stream. Workers accumulate locally, merge after drain, join side channels, and exporters fan out from the committed report.",
  "Cellular mode promotes the execute child into a controller. Cells run ordinary execute on sliced envelopes and return mergeable partitions over Velo.",
  "Cargo features define the linked universe. Default builds include gRPC and cellular; parquet, dynosim, and pyo3-embed opt in and fail closed when missing.",
  "The open seams are AIPerfExtension registration at bootstrap and Clock plus RequestSink substitution on the hot path. Cellular wraps that same core.",
] as const;

export const SLIDES: readonly SlideDefinition[] = STEPS.map((slide, index) => ({
  ...slide,
  narration: NARRATION[index] ?? "",
}));
