import type { SlideDefinition } from "../../core/types";

export const SLIDES: readonly SlideDefinition[] = [
  {
    eyebrow: "Why Dynosim",
    title: "Co-simulation without a live inference server",
    lede:
      "Dynosim runs AIPerf's full workload loop, observers, and metrics against Dynamo's passive mock engine in-process. No HTTP sockets, no gRPC wire path — the transport ID selects a simulation driver instead.",
    term: {
      word: "dynosim",
      meaning:
        "Feature-gated transport family (`dynosim_offline`, `dynosim_online`) that replaces native HTTP execution with SteppableReplay-driven co-simulation.",
    },
    points: [
      "AIPerf owns scheduling, Clock, RequestObserver, and report merge.",
      "Dynamo mocker owns scheduler and performance-model state behind SteppableReplay.",
      "Same metric vocabulary as online runs when configured for parity.",
    ],
    narration:
      "Dynosim lets AIPerf benchmark without a live inference server. The workload loop and observers stay native; Dynamo's passive engine supplies timing and token events in-process.",
    caption: "No serving socket on the hot path.",
  },
  {
    eyebrow: "Feature gate",
    title: "Cargo feature dynosim pulls the native crates",
    lede:
      "Ordinary lean HTTP builds omit dynosim entirely. Enabling the `dynosim` feature links dynamo-mocker and exposes transport IDs `dynosim_offline` and `dynosim_online` in the frozen registry.",
    term: { word: "transport ID", meaning: "Config v2 transport.type string resolved at Application bootstrap; unknown IDs fail closed." },
    points: [
      "`cargo build -p aiperf-cli --features dynosim` enables the path.",
      "Registry registers dynosim transports only when the feature is on.",
      "Missing feature → unknown transport at validate time.",
    ],
    narration:
      "Dynosim is behind the dynosim Cargo feature. When enabled, the registry exposes dynosim offline and online transport identifiers alongside HTTP and gRPC.",
    caption: "Opt-in at compile time, selected at config time.",
  },
  {
    eyebrow: "Config seam",
    title: "One BenchmarkRun; transport selects the clock",
    lede:
      "The same Config v2 envelope describes workload, endpoints, and artifacts. Changing transport.type from http to dynosim_offline swaps the execution driver and clock implementation without rewriting the benchmark plan.",
    term: { word: "BenchmarkRun", meaning: "Strict protocol-v2 object materialized by profile and passed to the execution child." },
    points: [
      "Workload, phases, and artifact settings stay shared.",
      "`dynosim_offline` → SimClock + virtual pump.",
      "`dynosim_online` → RealClock + wall-time replay.",
    ],
    narration:
      "You author one benchmark configuration. The transport field chooses offline simulation with a virtual clock, or online replay against wall time.",
    caption: "Config is shared; transport picks real vs. virtual time.",
  },
  {
    eyebrow: "Routing",
    title: "Factory bypasses native HTTP execution",
    lede:
      "At bootstrap, dynosim registers its own ExecutionSinkBuilder. The coordinator never constructs Hyper or Tonic clients for these runs — preparation builds an EngineHost and DynosimSink instead.",
    term: { word: "ExecutionSinkBuilder", meaning: "Transport hook that produces worker-local sinks; dynosim's builder wires the replay engine seam." },
    points: [
      "Application freeze includes dynosim factories when feature-gated.",
      "Validate rejects dynosim IDs on builds without the feature.",
      "HTTP/gRPC code paths are not entered for dynosim transport IDs.",
    ],
    narration:
      "Registry routing sends dynosim runs down a dedicated factory path. Native HTTP and gRPC clients are never constructed for these transport identifiers.",
    caption: "Transport ID → dynosim factory, not HTTP stack.",
  },
  {
    eyebrow: "Composition",
    title: "EngineHost + DynosimSink share one observer seam",
    lede:
      "DynosimSink implements RequestSink and forwards token, usage, and terminal events into the same RequestObserver pipeline HTTP uses. EngineHost wraps SteppableReplay and bridges DirectRequest submission.",
    term: { word: "DynosimSink", meaning: "Worker-local sink that drives replay turns through loadgen-core observation without wire encoding." },
    points: [
      "RequestObserver callbacks match online semantics.",
      "TTFT is the first token observation — no separate first-token event.",
      "Terminal coalescing policy matches configured streaming mode.",
    ],
    narration:
      "EngineHost wraps the replay engine. DynosimSink implements the same RequestSink and RequestObserver contract as HTTP transports.",
    caption: "Same observation vocabulary, different dispatch backend.",
  },
  {
    eyebrow: "Offline mode",
    title: "dynosim_offline owns SimClock",
    lede:
      "Offline runs use SimClock: integer-nanosecond virtual time with deterministic event ordering. The sim pump advances virtual time until the workload future completes or the source quiesces.",
    term: { word: "SimClock", meaning: "Virtual Clock implementation; scheduling and measurement must not call wall-clock APIs on hot paths." },
    points: [
      "Arrivals, phase gates, and backoff use SimClock timers.",
      "Engine steps consume virtual time, not wall time.",
      "Runs are reproducible when inputs and seeds are fixed.",
    ],
    narration:
      "Offline dynosim runs on SimClock. Virtual nanoseconds advance under the sim pump until the workload completes.",
    caption: "Deterministic virtual time drives offline replay.",
  },
  {
    eyebrow: "Online mode",
    title: "dynosim_online keeps RealClock",
    lede:
      "Online replay still uses SteppableReplay but drives it with RealClock — wall time between engine steps. This validates live integration paths while preserving the same observer and metric merge logic.",
    term: { word: "RealClock", meaning: "Wall-time Clock used for HTTP, gRPC, and dynosim_online execution." },
    points: [
      "drive_real_with_source is the wall-clock twin of drive_sim_with_source.",
      "Useful when timing must reflect actual scheduler latency.",
      "Metric merge and artifacts follow the same coordinator path.",
    ],
    narration:
      "Online dynosim uses RealClock. The same replay engine runs, but virtual scheduling gives way to wall-clock pacing.",
    caption: "Same engine contract; clock switches to wall time.",
  },
  {
    eyebrow: "Clock compare",
    title: "RealClock vs SimClock semantics",
    lede:
      "Both implement Clock::now_ns and timer registration. SimClock advances only when the driver pumps; RealClock tracks system time. Measurement code must route through Clock in both modes.",
    term: { word: "Clock::is_virtual", meaning: "Runtime selector between real-reactor and simulation drivers; hot paths branch on virtual vs. real timers." },
    points: [
      "Never call Instant::now on request hot paths.",
      "SimClock tie-breaks favor scheduled arrivals before engine passes at equal t.",
      "RealClock online runs behave like ordinary profile executions.",
    ],
    narration:
      "RealClock tracks wall time. SimClock advances only when the driver pumps events. All measurement must go through the Clock abstraction in both modes.",
    caption: "One Clock trait; two time domains.",
  },
  {
    eyebrow: "Event queues",
    title: "Virtual timers and external engine events",
    lede:
      "During offline drive, two queues compete: SimClock's ready timer set and the replay engine's next external event time. The driver always resolves the earliest timestamp, with clock events winning ties.",
    term: { word: "SimEventSource", meaning: "Passive engine adapter exposing next_event_ns and step bounds for the graph runtime pump." },
    points: [
      "Clock queue: arrivals, phase transitions, grace timers.",
      "Source queue: SteppableReplay engine passes and token timing.",
      "Tie at t → clock fires before the engine observes its batch.",
    ],
    narration:
      "Two virtual queues feed the pump: SimClock timers and the replay engine's next event. The driver always picks the earliest timestamp.",
    caption: "Earliest event wins; clock wins ties at the same t.",
  },
  {
    eyebrow: "Sim pump",
    title: "Poll workload to quiescence, then advance time",
    lede:
      "drive_sim_with_source loops: poll the workload future on a current-thread runtime; if Pending and idle, advance to the earliest clock or source event; repeat until Ready or stall/deadlock detection triggers.",
    term: { word: "sim pump", meaning: "The drive_sim_with_source loop in graph/runtime.rs that alternates polling and virtual time advancement." },
    points: [
      "Wakes during a poll re-enter without advancing time.",
      "Source steps can span [at_ns, end_ns) virtual intervals.",
      "Deadlock returns when neither queue has a future event.",
    ],
    narration:
      "The sim pump polls the workload future. When idle, it advances virtual time to the next clock or engine event and repeats until completion.",
    caption: "Poll → idle → advance → poll again.",
  },
  {
    eyebrow: "Ordering rule",
    title: "Earliest event selection; clock wins ties",
    lede:
      "If a request-rate arrival and an engine pass both occur at the same virtual instant, the clock event fires first. Accepting a source step that jumps across a parked clock event would make batching backend-dependent.",
    term: { word: "arrival-before-pass", meaning: "Invariant preserved by clock-wins-ties ordering in drive_sim_inner." },
    points: [
      "validate_event_time rejects time regression.",
      "Overshoot protection errors if a step ends past the next clock event.",
      "Preserves parity with scheduled HTTP arrival ordering.",
    ],
    narration:
      "When timestamps tie, the clock queue wins. That preserves arrival-before-pass ordering required for deterministic batching.",
    caption: "Clock-wins-ties keeps scheduling backend-independent.",
  },
  {
    eyebrow: "Step bounds",
    title: "Engine steps respect virtual interval limits",
    lede:
      "Each source.step(at_ns) returns SimStep with end_ns. The driver rejects steps that regress time or overshoot the next parked clock event — those guards keep the pump monotonic and safe.",
    term: { word: "SimStep", meaning: "One engine advance with end_ns, made_progress, and optional terminal metadata for the pump." },
    points: [
      "TimeRegression if end_ns < now.",
      "OvershotClockEvent if end_ns passes a sooner clock timer.",
      "Stall detection after MAX_NO_PROGRESS_STEPS idle iterations.",
    ],
    narration:
      "Engine steps declare an end time. The pump rejects regressions and overshoots that would skip a parked clock event.",
    caption: "Monotonic virtual time with overshoot protection.",
  },
  {
    eyebrow: "Submission",
    title: "DirectRequest and waiter mailboxes",
    lede:
      "Turns become DirectRequest values submitted to SteppableReplay. Completion signals flow back through per-request waiter mailboxes polled by the sink — no HTTP encoding on the path.",
    term: { word: "DirectRequest", meaning: "Dynamo mocker request descriptor carrying prompt, decode budget, and routing hints for one replay turn." },
    points: [
      "Sink submits when the workload dispatches a turn.",
      "Engine events dequeue into token and terminal observer callbacks.",
      "Waiters bridge async engine state to LocalSet-friendly sinks.",
    ],
    narration:
      "Turns become DirectRequest values submitted to the replay engine. Waiter mailboxes bridge completion back into the sink without HTTP.",
    caption: "In-process submission; mailbox completion.",
  },
  {
    eyebrow: "Token path",
    title: "Engine events become RequestObserver callbacks",
    lede:
      "As SteppableReplay emits token, usage, and terminal engine events, DynosimSink translates them into the same observer vocabulary HTTP reduction produces: classified tokens, batched output tokens, usage fields, and terminal status.",
    term: { word: "RequestObserver", meaning: "Worker-local callback surface; TTFT is the first token observation." },
    points: [
      "First token observation sets TTFT.",
      "Usage fields stay absent when the engine omits them.",
      "Terminal events close the sink lifecycle.",
    ],
    narration:
      "Replay engine events translate into RequestObserver callbacks: tokens, usage, and terminal status match the HTTP reduction path.",
    caption: "Observer seam is transport-neutral.",
  },
  {
    eyebrow: "Metrics",
    title: "TTFT, ITL batches, usage, terminal merge",
    lede:
      "NativeMetricsObserver accumulates per-record timing from observer events. TTFT comes from the first token; inter-token spacing from subsequent token observations; usage_* mirrors endpoint usage when present.",
    term: { word: "NativeMetricsObserver", meaning: "Runtime adapter that turns observer events into metrics_core accumulators and record traces." },
    points: [
      "Generated-token ITL excludes terminal usage frames.",
      "Sketch mode merges digests; exact mode retains records.",
      "Phase windows and steady-state derive from the same curves as HTTP.",
    ],
    narration:
      "Metrics accumulate from observer events: time to first token, inter-token spacing, usage fields, and terminal status merge like online runs.",
    caption: "Same metric plane; source is replay instead of wire.",
  },
  {
    eyebrow: "Delivery modes",
    title: "Incremental tokens vs terminal coalescing",
    lede:
      "Streaming configuration selects whether token observations arrive incrementally as the engine produces them or coalesce at terminal. Dynosim honors the same policy flags HTTP SSE paths respect.",
    term: { word: "terminal coalescing", meaning: "Delivery mode that emits batched output at terminal instead of per-token streaming observations." },
    points: [
      "Incremental: each engine token → observer token event.",
      "Coalesced: sink batches until terminal reduces once.",
      "Policy is endpoint/streaming config, not dynosim-specific.",
    ],
    narration:
      "Streaming policy controls incremental token events versus terminal coalescing. Dynosim honors the same flags as HTTP streaming paths.",
    caption: "Delivery shape follows config, not transport family.",
  },
  {
    eyebrow: "Completion",
    title: "Drain, stall, and deadlock detection",
    lede:
      "When the workload future completes, the pump drains remaining source events. If neither clock nor source can progress, the driver returns deadlocked. Repeated no-progress steps raise SimDriveError::Stalled.",
    term: { word: "RunOutcome", meaning: "Pump result carrying deadlocked flag when virtual time cannot advance and the workload remains pending." },
    points: [
      "Ready workload → drain_source then exit.",
      "No next event with pending work → deadlocked.",
      "Stall guard prevents infinite idle loops.",
    ],
    narration:
      "On completion the pump drains remaining engine events. Stall and deadlock detection prevent infinite idle loops when no progress is possible.",
    caption: "Clean termination with explicit stall signals.",
  },
  {
    eyebrow: "Recap",
    title: "Offline vs online parity and artifacts",
    lede:
      "Both modes write native-v2.json through the same Coordinator persist path. Choose offline for deterministic sweeps and co-simulation; choose online when wall-clock fidelity matters. Artifacts and exporters follow the configured BenchmarkRun.",
    term: { word: "native-v2.json", meaning: "Authoritative merged report before optional exporter fan-out." },
    points: [
      "Offline: SimClock + drive_sim_with_source.",
      "Online: RealClock + drive_real_with_source.",
      "Exporters and cellular modes are orthogonal to dynosim choice.",
    ],
    narration:
      "Offline dynosim gives deterministic virtual time. Online dynosim keeps wall clocks. Both produce the same native report and configured artifacts through the coordinator.",
    caption: "Pick the clock; keep the rest of AIPerf unchanged.",
  },
] as const;
