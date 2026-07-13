// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Narrative model for the guided runtime story.
 *
 * The story tells one AIPerf run end to end in a small number of paged
 * chapters. Where a chapter maps onto one of the extensibility seams, it
 * carries a {@link StoryTrait} whose concrete implementations are drawn from
 * the current Rust workspace (see `crates/`), so the page literally shows the
 * different implementations behind each trait rather than describing them in
 * prose.
 */

/** Colour family used to tint a chapter and its figure. */
export type StoryAccent = "author" | "runner" | "clock" | "workload" | "endpoint" | "transport" | "observe";

/** How a chapter's hub is labelled above the trait name. */
export type StoryHubKind = "trait" | "seam" | "family" | "stage";

/** One concrete implementation behind a trait. */
export interface TraitImpl {
  /** Rust type name, shown verbatim. */
  readonly name: string;
  /** Owning workspace crate, e.g. `aiperf-clock`. */
  readonly crate: string;
  /** Key source file within the crate's `src/`. */
  readonly file: string;
  /** Five-or-so word differentiator; no full sentences. */
  readonly note: string;
  /** Short tag rendered as a chip, e.g. `online` / `offline`. */
  readonly tag: string;
}

/** A trait (or trait-like family) with its shipped implementations. */
export interface StoryTrait {
  readonly kind: StoryHubKind;
  /** Trait name, shown verbatim. */
  readonly name: string;
  /** Terse signature or verb, not a full API dump. */
  readonly signature: string;
  /** Crate that defines the trait. */
  readonly crate: string;
  /** File that defines the trait. */
  readonly file: string;
  readonly impls: readonly TraitImpl[];
}

/** One way to author Config v2 (CLI commands vs YAML file). */
export interface StoryConfigMode {
  readonly id: "cli" | "yaml";
  readonly label: string;
  readonly tag: string;
  readonly lines: readonly string[];
}

/** A single page in the guided flow. */
export interface StoryChapter {
  readonly id: string;
  readonly accent: StoryAccent;
  /** Short label above the title, e.g. `the {clock} seam`. */
  readonly kicker: string;
  /** Two-to-four word chapter title. */
  readonly title: string;
  /** Exactly one sentence. Keep it short. */
  readonly blurb: string;
  /** Optional trait fan; omitted for pure narrative stages. */
  readonly trait?: StoryTrait;
  /** CLI vs YAML compare on the author chapter. */
  readonly configModes?: readonly StoryConfigMode[];
  /** Evidence path shown to maintainers when there is no trait fan. */
  readonly evidence?: string;
}

export const runtimeStory: readonly StoryChapter[] = [
  {
    id: "author",
    accent: "author",
    kicker: "the only human CLI",
    title: "Author the run",
    blurb: "You describe one benchmark in Config v2; Python is the single front door.",
    configModes: [
      {
        id: "cli",
        label: "CLI workflow",
        tag: "commands",
        lines: [
          "aiperf config init --template minimal -o benchmark.yaml",
          "aiperf config validate benchmark.yaml",
          "aiperf profile --config benchmark.yaml",
        ],
      },
      {
        id: "yaml",
        label: "Config v2 file",
        tag: "authored",
        lines: [
          'schemaVersion: "2.0"',
          "benchmark:",
          "  model: meta-llama/Llama-3.1-8B-Instruct",
          "  endpoint:",
          "    url: http://localhost:8000",
          "  dataset:",
          "    type: synthetic",
        ],
      },
    ],
    evidence: "src/aiperf/cli_runner/_single_run.py",
  },
  {
    id: "launch",
    accent: "runner",
    kicker: "one child, protocol v2 only",
    title: "Launch the runner",
    blurb: "Python spawns one aiperf-runner and hands it a strict, authored request.",
    evidence: "crates/runner/src/protocol_v2.rs",
  },
  {
    id: "freeze",
    accent: "runner",
    kicker: "frozen at bootstrap",
    title: "Freeze the application",
    blurb: "One RunnerApplication locks the registries that capabilities, validation, and execution all share.",
    evidence: "crates/runner/src/registry.rs",
  },
  {
    id: "clock",
    accent: "clock",
    kicker: "the {clock} seam",
    title: "Choose a clock",
    blurb: "The same executor runs on wall time or virtual time behind one trait.",
    trait: {
      kind: "trait",
      name: "Clock",
      signature: "now_ns · sleep · is_virtual",
      crate: "aiperf-clock",
      file: "clock.rs",
      impls: [
        {
          name: "RealClock",
          crate: "aiperf-clock",
          file: "real_clock.rs",
          note: "monotonic instant, timerfd ns sleeps",
          tag: "online",
        },
        {
          name: "SimClock",
          crate: "aiperf-clock",
          file: "sim_clock.rs",
          note: "integer-ns discrete-event heap",
          tag: "offline",
        },
      ],
    },
  },
  {
    id: "workload",
    accent: "workload",
    kicker: "one dispatch verb, many schedules",
    title: "Choose a workload",
    blurb: "Arrival strategies are schedule generators over a shared runtime, not bespoke loops.",
    trait: {
      kind: "family",
      name: "Workload",
      signature: "schedule → dispatch",
      crate: "aiperf",
      file: "workload.rs",
      impls: [
        {
          name: "RequestRateWorkload",
          crate: "aiperf",
          file: "request_rate.rs",
          note: "one turn per tick, FIFO",
          tag: "rate",
        },
        {
          name: "UserCentricWorkload",
          crate: "aiperf",
          file: "user_centric.rs",
          note: "per-user steady-state pacing",
          tag: "users",
        },
        {
          name: "FixedScheduleWorkload",
          crate: "aiperf",
          file: "fixed_schedule.rs",
          note: "absolute authored trace replay",
          tag: "replay",
        },
        {
          name: "GraphWorkload",
          crate: "aiperf-graph",
          file: "workload.rs",
          note: "DAG fan-out and fan-in",
          tag: "graph",
        },
      ],
    },
  },
  {
    id: "endpoint",
    accent: "endpoint",
    kicker: "open dialect registry",
    title: "Bind an endpoint",
    blurb: "Each API dialect prepares its own worker-local request and response shape.",
    trait: {
      kind: "trait",
      name: "EndpointFactory",
      signature: "prepare → PreparedEndpoint",
      crate: "aiperf-endpoints",
      file: "registry.rs",
      impls: [
        {
          name: "ChatEndpoint",
          crate: "aiperf-endpoints",
          file: "endpoints.rs",
          note: "OpenAI chat completions, SSE",
          tag: "http",
        },
        {
          name: "MessagesEndpoint",
          crate: "aiperf-endpoints",
          file: "anthropic.rs",
          note: "Anthropic /v1/messages parity",
          tag: "http",
        },
        {
          name: "VllmGenerateEndpoint",
          crate: "aiperf-endpoints",
          file: "vllm_generate.rs",
          note: "raw token-in, token-out",
          tag: "http",
        },
        {
          name: "RivaEndpoint",
          crate: "aiperf-endpoints",
          file: "riva.rs",
          note: "ASR / TTS / NLP RPCs",
          tag: "grpc",
        },
      ],
    },
  },
  {
    id: "dispatch",
    accent: "transport",
    kicker: "the {transport} seam",
    title: "Dispatch the request",
    blurb: "Every transport drives a request to terminal and emits the same observer callbacks.",
    trait: {
      kind: "seam",
      name: "RequestSink<R>",
      signature: "dispatch(R) → observer",
      crate: "loadgen-core",
      file: "sink.rs",
      impls: [
        {
          name: "HttpSink",
          crate: "aiperf-transport-http",
          file: "client/http_client.rs",
          note: "hyper h1 / h2c / UDS / TLS",
          tag: "online",
        },
        {
          name: "GrpcSink",
          crate: "aiperf-transport-grpc",
          file: "transport.rs",
          note: "tonic HTTP/2 streaming",
          tag: "online",
        },
        {
          name: "DynosimSink",
          crate: "aiperf",
          file: "dynosim.rs",
          note: "in-process engine, no sockets",
          tag: "offline",
        },
      ],
    },
  },
  {
    id: "observe",
    accent: "observe",
    kicker: "local-loop callbacks",
    title: "Observe and report",
    blurb: "Timing, tokens, and usage flow through one observer into the metrics report.",
    trait: {
      kind: "trait",
      name: "RequestObserver",
      signature: "on_token · on_usage · on_terminal",
      crate: "loadgen-core",
      file: "sink.rs",
      impls: [
        {
          name: "NativeMetricsObserver",
          crate: "aiperf",
          file: "metrics.rs",
          note: "feeds the native-v2 accumulator",
          tag: "metrics",
        },
        {
          name: "CollectorObserver",
          crate: "aiperf-core",
          file: "observer.rs",
          note: "records into TraceCollector",
          tag: "trace",
        },
      ],
    },
  },
];

/** Total pages in the guided flow. */
export const runtimeStoryLength = runtimeStory.length;
