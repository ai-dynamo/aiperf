import type { SlideDefinition } from "../../core/types";

type StepInput = Omit<SlideDefinition, "narration">;

const STEPS: readonly StepInput[] = [
  {
    eyebrow: "Product shell",
    title: "One binary is both CLI and engine",
    lede:
      "AIPerf ships as a single native `aiperf` executable from crate `aiperf-cli`. There is no separate runner process—the same binary parses public commands and re-executes itself for benchmark runs.",
    term: { word: "aiperf-cli", meaning: "The product entry crate: public commands, Config v2 loading, and self-spawned execution over stdio." },
    points: ["`rust/cli` builds the `aiperf` binary.", "Global allocator is mimalloc on the hot path.", "Internal modes are intercepted before normal CLI parsing."],
    caption: "One executable, two hats: operator CLI and execution child.",
  },
  {
    eyebrow: "Workspace map",
    title: "Six crates, one dependency direction",
    lede:
      "The Rust workspace is intentionally small. Capability flows cli → runtime → loadgen-core. Mock server and e2e are separate targets; pyext is packaging-only.",
    term: { word: "loadgen-core", meaning: "Transport-neutral observer and sink vocabulary with no HTTP, gRPC, or engine dependencies." },
    points: ["Members: loadgen-core, runtime, cli, mock-server, e2e, pyext.", "aiperf-cli depends on aiperf-runtime with the engine feature.", "aiperf-mock-server is a standalone benchmark target."],
    caption: "cli → runtime → loadgen-core",
  },
  {
    eyebrow: "Startup order",
    title: "Intercept internal modes, then dispatch",
    lede:
      "Every process starts the same way: initialize logging, check for hidden execution flags like `--execute`, `--cell`, or `--aggregator`, and only then route public subcommands.",
    term: { word: "execute_mode", meaning: "Hidden argv surface that runs protocol-v2 children before clap ever sees a public command." },
    points: ["`main.rs` calls `execute_mode::is_execution_mode`.", "Internal handlers never return to public dispatch.", "Everything else goes through `dispatch::run`."],
    caption: "Hidden modes short-circuit; public commands fall through.",
  },
  {
    eyebrow: "Command surface",
    title: "Native benchmark commands stay in Rust",
    lede:
      "`profile`, `config`, cellular roles, `slurm run`, and several utility commands are native. Most operational tooling still delegates to the Python package unless built with pyo3-embed.",
    term: { word: "delegate", meaning: "Lean builds shell out to `python -m aiperf`; pyo3-embed runs the same entrypoint in-process." },
    points: ["`dispatch.rs` matches the first argv token.", "Unknown commands delegate to Python.", "Feature gates select gRPC, cellular, dynosim, parquet, and embed modes."],
    caption: "Benchmark hot path native; extended surface delegated.",
  },
  {
    eyebrow: "Configuration",
    title: "Config v2 resolves into BenchmarkRun",
    lede:
      "`aiperf profile --config` reads YAML, expands env vars and Jinja, applies CLI overrides, and materializes a strict `BenchmarkRun` object—the wire payload for execution.",
    term: { word: "BenchmarkRun", meaning: "The authored protocol-v2 request describing workload, transport, endpoints, artifacts, and runtime facts." },
    points: ["`yaml.rs` handles substitution and alias normalization.", "`load.rs` and `model/` build the typed run object.", "`config init|validate|expand` are native helpers."],
    caption: "YAML in → validated BenchmarkRun out.",
  },
  {
    eyebrow: "Self execution",
    title: "profile spawns aiperf --execute",
    lede:
      "Each run is a fresh child of the same binary. Parent writes JSON to stdin; child returns one terminal JSONL envelope on stdout while stderr carries diagnostics.",
    term: { word: "stdio seam", meaning: "The deliberate parent/child boundary that keeps operator UX separate from execution isolation." },
    points: ["`execute::run_once` spawns `current_exe()` with hidden flags.", "Stdout stays reserved for protocol traffic.", "Panics in the child become typed failure envelopes."],
    caption: "Same binary, new process, protocol on stdio.",
  },
  {
    eyebrow: "Wire contract",
    title: "Protocol v2 wraps the bare run",
    lede:
      "The child accepts protocol version 2 only. It composes a stock `Application` once, then `handle_v2` validates, prepares, executes, and persists results.",
    term: { word: "Application", meaning: "Frozen runtime composition: registry, coordinator, and factories selected at bootstrap." },
    points: ["`protocol_v2.rs` defines strict authored types.", "Unknown fields fail closed at serde boundaries.", "Factory-owned config stays opaque until registry decode."],
    caption: "Envelope v2 in → terminal envelope out.",
  },
  {
    eyebrow: "Bootstrap",
    title: "AIPerfRegistry freezes capabilities",
    lede:
      "At startup one registry is built transactionally from explicit extensions: loaders, samplers, endpoints, transports, workloads, exporters, and actuators. There is no runtime plugin scanning.",
    term: { word: "AIPerfExtension", meaning: "Compile-time registration hook that adds implementations into the shared registry during Application construction." },
    points: ["Built-ins register HTTP, gRPC, dry_run, dynosim when features are on.", "Duplicate identifiers are rejected.", "Unknown transport or workload IDs fail closed."],
    caption: "Capabilities are frozen before the first request.",
  },
  {
    eyebrow: "Composition root",
    title: "Coordinator validates, prepares, executes, persists",
    lede:
      "`Coordinator::handle` is the orchestration spine: validate envelope and endpoint profiles, prepare dataset and sidecars, execute the selected program shape, write `native-v2.json`, then run exporters best-effort.",
    term: { word: "Coordinator", meaning: "Engine composition root that owns validate → prepare → execute → persist for a single BenchmarkRun." },
    points: ["Sidecar collectors start only on the primary cell when needed.", "Exporter failures are logged, not run-fatal.", "Cellular and graph runs branch inside the same coordinator."],
    caption: "One coordinator owns the whole run lifecycle.",
  },
  {
    eyebrow: "Time seam",
    title: "Every schedule uses Clock",
    lede:
      "Hot paths never call wall-clock APIs directly. `RealClock` drives online HTTP and gRPC; `SimClock` drives dynosim and graph co-simulation with deterministic ordering.",
    term: { word: "Clock", meaning: "Injectable time source used for scheduling, measurement gates, backoff, and simulation driving." },
    points: ["Measurement must not use `Instant::now` on hot paths.", "Virtual clocks expose integer-nanosecond time.", "Transport retry backoff is clock-driven too."],
    caption: "Real time online, virtual time in simulation.",
  },
  {
    eyebrow: "Inputs",
    title: "Dataset flows load → sample → materialize",
    lede:
      "Loaders ingest files and traces, conversations intern into a BLAKE3-addressed segment pool, samplers choose turns, and materializers build wire payloads per endpoint dialect.",
    term: { word: "Segment pool", meaning: "Content-addressed conversation storage shared read-only across worker threads via Arc." },
    points: ["LoaderRegistry and SamplerRegistry live in the frozen registry.", "Synthetic, JSONL, graph, and trace formats share one substrate.", "Prompt materialization stays endpoint-aware."],
    caption: "Inputs become endpoint-ready requests once, then reuse handles.",
  },
  {
    eyebrow: "Work generation",
    title: "Workloads and phases drive turns",
    lede:
      "Scheduled workloads—request rate, concurrency, user-centric, fixed schedule—emit turn schedules. Phase runtime applies warmup, profiling, ramp, drain, and cancellation policy before transport ever sends.",
    term: { word: "phase_runtime", meaning: "Shared lifecycle orchestration that connects schedulers to turn dispatch and terminal drain behavior." },
    points: ["Graph programs use the separate DAG executor.", "`scheduled.rs` bridges Workload to TurnDispatcher.", "Cancellation and grace propagate through shared phase code."],
    caption: "Scheduling decides when; transport decides how to send.",
  },
  {
    eyebrow: "Observation seam",
    title: "loadgen-core keeps transport neutral",
    lede:
      "Transports implement `RequestSink`; observers receive arrival, admission, token, usage, and terminal events. TTFT is the first token observation—there is no separate first-token event.",
    term: { word: "RequestObserver", meaning: "Worker-local callback surface with no Send bound, allowing LocalSet-friendly state." },
    points: ["HTTP and gRPC share reduction in `transport::reduce`.", "Measurement accumulates in `transport::measure`.", "Endpoint usage is captured verbatim when present."],
    caption: "Transport sends; observer measures; sink owns the lifecycle.",
  },
  {
    eyebrow: "Parallelism",
    title: "Thread-per-core workers, not mutex hot paths",
    lede:
      "When `workers == 1`, scheduler and sink co-locate on a current-thread runtime. When `workers > 1`, each OS thread runs a self-contained sub-cell with its own LocalSet and transport sink.",
    term: { word: "Sub-cell", meaning: "A worker thread that owns scheduling, admission, transport, capture, and local measurement without cross-thread locks on the request path." },
    points: ["Read-only Arc data crosses thread boundaries.", "Per-worker state stays local and merges at boundaries.", "Hyper and SSE paths rely on LocalSet because clients are not Send."],
    caption: "Scale out by adding worker threads, not shared mutable hot state.",
  },
  {
    eyebrow: "Scale-out",
    title: "Cellular mode adds controller and cells",
    lede:
      "`--cells N` promotes the run to a controller that partitions work, launches `aiperf --cell` children, merges records or folded stores over Velo, and writes one authoritative report. SLURM and Kubernetes reuse the same model.",
    term: { word: "Cell partition", meaning: "Deterministic slice of the global request or conversation budget owned by one remote cell process." },
    points: ["Rank 0 or the controller owns merge and artifact shipping.", "Velo carries control only; HTTP and gRPC carry measured load.", "Bulk artifacts can use a separate HTTP/1 + zstd path."],
    caption: "Same engine, more processes, partitioned ownership.",
  },
  {
    eyebrow: "Outputs & gates",
    title: "Metrics merge, exporters fan out, features opt in",
    lede:
      "Workers accumulate metrics locally; the report plane writes `native-v2.json` as authoritative truth, then exporters emit JSON, CSV, Parquet, OTLP, and more. Cargo features gate gRPC, cellular, parquet, dynosim, and pyo3-embed.",
    term: { word: "native-v2.json", meaning: "Authoritative merged report for a run before optional exporter side outputs." },
    points: ["Sketch mode uses mergeable t-digests for approximate percentiles.", "`--steady-state` derives a measurement window from concurrency crossings.", "Mock server stays outside profile supervision for deterministic tests."],
    caption: "Measure locally, merge once, export many formats.",
  },
] as const;

const NARRATION = [
  "AIPerf ships as one native aiperf binary. That same executable is both the public command line and the hidden execution engine.",
  "The Rust workspace stays small. Capability flows from aiperf-cli into aiperf-runtime and then into loadgen-core.",
  "Startup always checks hidden execution modes first. Only after that does the process route public subcommands.",
  "Core benchmark commands stay native in Rust. Most operational tooling still delegates to Python unless the build embeds it.",
  "Profile reads Config v2, expands it, and resolves a strict BenchmarkRun object that describes the whole benchmark.",
  "Each profile run spawns a fresh child of the same binary with aiperf execute over stdio.",
  "Protocol version two wraps the run, composes Application once, and returns one terminal envelope.",
  "At bootstrap, AIPerfRegistry freezes loaders, samplers, endpoints, transports, workloads, and exporters.",
  "Coordinator is the composition root: validate, prepare, execute, persist native-v2.json, then run exporters.",
  "All timing goes through Clock. RealClock drives online traffic; SimClock drives deterministic simulation.",
  "Datasets load, sample, and materialize into endpoint-ready requests through one shared substrate.",
  "Workloads and phase runtime decide when turns fire. Transports only send and observe the wire path.",
  "loadgen-core defines the neutral observer seam. Transports implement RequestSink; measurement stays shared.",
  "Parallelism is thread-per-core. Each worker owns local scheduling, transport, and capture without hot-path mutexes.",
  "Cellular mode adds a controller, remote cells, Velo control traffic, and a single merged report.",
  "Metrics merge into native-v2.json, exporters fan out artifacts, and Cargo features opt into gRPC, cellular, dynosim, and more.",
] as const;

export const SLIDES: readonly SlideDefinition[] = STEPS.map((slide, index) => ({
  ...slide,
  narration: NARRATION[index] ?? "",
}));
