// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  graphEdgeSchema,
  graphNodeSchema,
  graphSceneSchema,
  type EvidenceReference,
  type ExecutionFlavor,
  type GraphEdge,
  type GraphNode,
  type GraphScene,
  type Ownership,
} from "../../domain/architecture";

const redesignSpec =
  "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md";

function source(
  path: string,
  start: number,
  end: number,
  symbol?: string,
): EvidenceReference {
  return {
    path,
    lines: { start, end },
    role: "source",
    ...(symbol ? { symbol } : {}),
  };
}

function design(path: string): EvidenceReference {
  return { path, role: "design" };
}

const fullAudience: GraphNode["audience"] = {
  visibility: ["executive", "developer", "maintainer"],
  autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
};

const fullSceneAudience: GraphScene["audience"] = {
  visibility: ["executive", "developer", "maintainer"],
  defaultDepth: { executive: 1, developer: 2, maintainer: 3 },
};

const builtJourneyFlavors: ExecutionFlavor[] = [
  "native_http",
  "native_grpc",
  "online_mock",
  "dynamo_offline",
  "dynamo_online",
];

interface JourneyDefinition {
  id: string;
  label: string;
  owner: Ownership;
  childIds: string[];
  evidence: EvidenceReference[];
}

const journeyDefinitions: JourneyDefinition[] = [
  {
    id: "node.journey.python-config-load",
    label: "Python config load",
    owner: "python",
    childIds: [],
    evidence: [source("src/aiperf/config/config.py", 390, 430)],
  },
  {
    id: "node.journey.config-v2-resolution",
    label: "Config-v2 resolution",
    owner: "python",
    childIds: [],
    evidence: [source("src/aiperf/config/resolution/plan.py", 380, 410)],
  },
  {
    id: "node.journey.authored-request-projection",
    label: "Authored request projection",
    owner: "python",
    childIds: [],
    evidence: [source("src/aiperf/orchestrator/rust_executor.py", 30, 80)],
  },
  {
    id: "node.journey.runner-spawn",
    label: "aiperf-runner spawn",
    owner: "python",
    childIds: [],
    evidence: [
      source(
        "src/aiperf/orchestrator/runner_installation.py",
        218,
        247,
        "RunnerInstallation.spawn",
      ),
    ],
  },
  {
    id: "node.journey.strict-jsonl-validation",
    label: "Strict JSONL validation",
    owner: "rust",
    childIds: ["node.runner-protocol-registries"],
    evidence: [
      source(
        "crates/runner/src/protocol_v2.rs",
        101,
        186,
        "RunnerOperationV2",
      ),
    ],
  },
  {
    id: "node.journey.frozen-runner-application",
    label: "Frozen RunnerApplication",
    owner: "rust",
    childIds: ["node.crate-dependency-topology"],
    evidence: [
      source(
        "crates/runner/src/application.rs",
        34,
        65,
        "RunnerApplication",
      ),
    ],
  },
  {
    id: "node.journey.workload-preparation",
    label: "Workload preparation",
    owner: "rust",
    childIds: [],
    evidence: [source("crates/runner/src/execute.rs", 350, 430)],
  },
  {
    id: "node.journey.scheduling-or-graph-ir",
    label: "Scheduling or Graph-IR",
    owner: "rust",
    childIds: [
      "node.runtime-composition",
      "node.scheduling-phase-lifecycle",
      "node.graph-ir-execution",
    ],
    evidence: [
      source(
        "crates/aiperf/src/scheduled.rs",
        421,
        500,
        "ScheduledRuntime",
      ),
    ],
  },
  {
    id: "node.journey.dataset-materialization",
    label: "Dataset materialization",
    owner: "rust",
    childIds: ["node.dataset-segment-pipeline"],
    evidence: [
      source(
        "crates/dataset/src/materialize.rs",
        110,
        180,
        "SegmentItemsMaterializer",
      ),
    ],
  },
  {
    id: "node.journey.endpoint-binding",
    label: "Endpoint binding",
    owner: "rust",
    childIds: ["node.endpoint-bindings-transports"],
    evidence: [
      source(
        "crates/endpoints/src/registry.rs",
        262,
        450,
        "EndpointFactory",
      ),
    ],
  },
  {
    id: "node.journey.http-grpc-dynamo-dispatch",
    label: "HTTP, gRPC, or Dynamo dispatch",
    owner: "rust",
    childIds: ["node.dynamo-offline-runner-backend"],
    evidence: [
      source(
        "crates/loadgen-core/src/sink.rs",
        157,
        160,
        "RequestSink",
      ),
    ],
  },
  {
    id: "node.journey.observer-callbacks",
    label: "Observer callbacks",
    owner: "rust",
    childIds: [],
    evidence: [
      source(
        "crates/loadgen-core/src/sink.rs",
        85,
        133,
        "RequestObserver",
      ),
    ],
  },
  {
    id: "node.journey.metrics-and-reporting",
    label: "Metrics and reporting",
    owner: "rust",
    childIds: ["node.metrics-telemetry", "node.accuracy-evaluator-hosting"],
    evidence: [
      source(
        "crates/metrics/src/report.rs",
        1800,
        1840,
        "NativeReporter",
      ),
    ],
  },
  {
    id: "node.journey.result-returned-to-python",
    label: "Result returned to Python",
    owner: "python",
    childIds: [],
    evidence: [source("src/aiperf/orchestrator/rust_executor.py", 150, 190)],
  },
];

const journeyNodes: GraphNode[] = journeyDefinitions.map(
  ({ id, label, owner, childIds, evidence: nodeEvidence }, index): GraphNode => ({
    id,
    tier: 0,
    parentId: null,
    childIds,
    owner,
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: label,
      developer: `Journey step ${index + 1}: ${label}`,
      maintainer: id,
    },
    summary: {
      executive: "A required stage in the canonical Python-to-result journey.",
      developer: "Connects authored Config v2 to one source-grounded Rust execution stage.",
      maintainer: "A line-ranged implementation fact in the canonical product execution path.",
    },
    evidence: nodeEvidence,
    seamPorts: [
      { id: `${id}.in`, name: "in", channel: "control" },
      { id: `${id}.out`, name: "out", channel: "control" },
      ...(id === "node.journey.result-returned-to-python"
        ? [
            {
              id: `${id}.report`,
              name: "result",
              channel: "report_result" as const,
            },
          ]
        : []),
    ],
    audience: fullAudience,
    footnotes: [],
  }),
);

const graphNodesRaw = [
  ...journeyNodes,
  {
    id: "node.runtime-composition",
    tier: 1,
    parentId: "node.journey.scheduling-or-graph-ir",
    childIds: [
      "node.clock-seam",
      "node.request-sink-seam",
      "node.dynamo-online-library-seam",
    ],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Runtime composition",
      developer: "Rust runtime composition",
      maintainer: "RunnerApplication runtime seam composition",
    },
    summary: {
      executive: "Composes one run from shared time and transport seams.",
      developer: "Connects the frozen runner application to reusable execution primitives.",
      maintainer: "Grounded in RunnerApplication construction and the library runtime seams.",
    },
    evidence: [
      source(
        "crates/runner/src/application.rs",
        34,
        65,
        "RunnerApplication",
      ),
    ],
    seamPorts: [
      { id: "port.runtime.in", name: "entry", channel: "control" },
      { id: "port.runtime.out", name: "dispatch", channel: "request_data" },
      { id: "port.runtime.telemetry", name: "observations", channel: "telemetry" },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.runner-protocol-registries",
    tier: 1,
    parentId: "node.journey.strict-jsonl-validation",
    childIds: ["node.dynamo-online-runner-pair"],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Runner protocol and registries",
      developer: "Protocol-v2 and registry freeze",
      maintainer: "Strict protocol-v2 RunnerApplication registries",
    },
    summary: {
      executive: "Validates authored requests against executable capabilities.",
      developer: "Uses one frozen registry graph for capabilities, validation, and execution.",
      maintainer: "Strict protocol DTOs and immutable runner application composition.",
    },
    evidence: [
      source(
        "crates/runner/src/protocol_v2.rs",
        101,
        186,
        "RunnerOperationV2",
      ),
      source(
        "crates/runner/src/application.rs",
        34,
        65,
        "RunnerApplication",
      ),
    ],
    seamPorts: [
      { id: "port.runner.in", name: "request", channel: "control" },
      { id: "port.runner.out", name: "prepared", channel: "request_data" },
      {
        id: "port.runner.planned",
        name: "planned-pair",
        channel: "control",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.scheduling-phase-lifecycle",
    tier: 1,
    parentId: "node.journey.scheduling-or-graph-ir",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Scheduling and phase lifecycle",
      developer: "Scheduling policy and phases",
      maintainer: "ScheduledRuntime and ClockPhaseOrchestrator",
    },
    summary: {
      executive: "Controls arrivals, admission, and phase transitions.",
      developer: "Combines workload issuance with phase escalation and adaptive controls.",
      maintainer: "Clock-paced scheduling and shared phase lifecycle implementation.",
    },
    evidence: [
      source(
        "crates/aiperf/src/scheduled.rs",
        421,
        500,
        "ScheduledRuntime",
      ),
      source(
        "crates/timing/src/phase/orchestrator.rs",
        137,
        210,
        "ClockPhaseOrchestrator",
      ),
    ],
    seamPorts: [
      { id: "port.schedule.in", name: "prepared-work", channel: "request_data" },
      { id: "port.schedule.out", name: "dispatch", channel: "request_data" },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dataset-segment-pipeline",
    tier: 1,
    parentId: "node.journey.dataset-materialization",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Dataset and segment pipeline",
      developer: "Dataset composition and segment storage",
      maintainer: "SegmentStore and materialization pipeline",
    },
    summary: {
      executive: "Builds repeatable request content from authored data.",
      developer: "Composes loader output into dense content-addressed segment handles.",
      maintainer: "Prefix-dependent segment storage and endpoint-ready materialization.",
    },
    evidence: [
      source(
        "crates/dataset/src/segment.rs",
        195,
        205,
        "SegmentStore",
      ),
      source(
        "crates/dataset/src/materialize.rs",
        110,
        180,
        "SegmentItemsMaterializer",
      ),
    ],
    seamPorts: [
      { id: "port.dataset.in", name: "authored-data", channel: "request_data" },
      {
        id: "port.dataset.out",
        name: "materialized-turns",
        channel: "request_data",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.endpoint-bindings-transports",
    tier: 1,
    parentId: "node.journey.endpoint-binding",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http", "native_grpc", "online_mock"],
    title: {
      executive: "Endpoint bindings and HTTP/gRPC transports",
      developer: "Endpoint registry and native transports",
      maintainer: "EndpointFactory and PreparedEndpoint bindings",
    },
    summary: {
      executive: "Maps one runtime to multiple inference protocol families.",
      developer: "Prepares worker-local HTTP or gRPC endpoint behavior.",
      maintainer: "Open endpoint factories, dense keys, and prepared endpoint contracts.",
    },
    evidence: [
      source(
        "crates/endpoints/src/registry.rs",
        262,
        450,
        "EndpointFactory",
      ),
    ],
    seamPorts: [
      {
        id: "port.endpoint.in",
        name: "prepared-endpoint",
        channel: "request_data",
      },
      {
        id: "port.endpoint.out",
        name: "transport-dispatch",
        channel: "request_data",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.graph-ir-execution",
    tier: 1,
    parentId: "node.journey.scheduling-or-graph-ir",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http", "online_mock", "dynamo_offline"],
    title: {
      executive: "Graph-IR execution",
      developer: "Graph workload and runtime",
      maintainer: "drive_real and drive_sim Graph-IR runtime",
    },
    summary: {
      executive: "Executes branching traces with explicit admission and failure policy.",
      developer: "Runs graph plans through real or virtual-clock execution drivers.",
      maintainer: "Shared graph runtime with source-aware real and simulation pumps.",
    },
    evidence: [
      source("crates/graph/src/runtime.rs", 192, 430, "drive_sim"),
    ],
    seamPorts: [
      { id: "port.graph.in", name: "graph-plan", channel: "request_data" },
      { id: "port.graph.out", name: "graph-dispatch", channel: "request_data" },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.metrics-telemetry",
    tier: 1,
    parentId: "node.journey.metrics-and-reporting",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Metrics and telemetry",
      developer: "Metrics accumulator and telemetry producers",
      maintainer: "MetricsAccumulator and native-v2 Reporter",
    },
    summary: {
      executive: "Converts execution observations into reportable evidence.",
      developer: "Merges request observations and side-channel telemetry.",
      maintainer: "IO-free accumulation with deterministic native-v2 reporting.",
    },
    evidence: [
      source(
        "crates/metrics/src/accumulator.rs",
        396,
        470,
        "MetricsAccumulator",
      ),
      source(
        "crates/metrics/src/report.rs",
        1800,
        1840,
        "NativeReporter",
      ),
    ],
    seamPorts: [
      { id: "port.metrics.in", name: "observer-events", channel: "telemetry" },
      { id: "port.metrics.token", name: "output-token", channel: "token" },
      {
        id: "port.metrics.out",
        name: "native-report",
        channel: "report_result",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.accuracy-evaluator-hosting",
    tier: 1,
    parentId: "node.journey.metrics-and-reporting",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "runtime_conditional" },
    flavors: ["native_http"],
    title: {
      executive: "Accuracy and evaluator hosting",
      developer: "Legacy evaluators and provider host",
      maintainer: "AccuracyEvaluator worker and provider host seams",
    },
    summary: {
      executive: "Combines quality scoring with runtime execution evidence.",
      developer: "Keeps evaluator semantics outside native inference transport.",
      maintainer: "Transport-free evaluator contract with runtime-hosted inference.",
    },
    evidence: [
      source(
        "crates/accuracy/src/worker.rs",
        139,
        190,
        "AccuracyEvaluator",
      ),
    ],
    seamPorts: [
      {
        id: "port.accuracy.in",
        name: "inference-turns",
        channel: "request_data",
      },
      {
        id: "port.accuracy.out",
        name: "score-results",
        channel: "report_result",
      },
    ],
    audience: fullAudience,
    footnotes: [
      {
        executive: "Legacy and migration details remain subordinate footnotes.",
        developer: "Legacy providers remain while bounded host migration proceeds.",
        maintainer: "Pinned evaluator providers retain benchmark-owned semantics.",
      },
    ],
  },
  {
    id: "node.crate-dependency-topology",
    tier: 1,
    parentId: "node.journey.frozen-runner-application",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Crate dependency topology",
      developer: "Workspace dependency topology",
      maintainer: "Cargo metadata dependency validation",
    },
    summary: {
      executive: "Shows major dependency boundaries in the Rust workspace.",
      developer: "Validates catalog dependency claims against Cargo metadata.",
      maintainer: "Exact package identities, dependency kinds, and workspace coverage.",
    },
    evidence: [
      source(
        "apps/architecture-atlas/src/domain/integrity.ts",
        404,
        455,
        "validateWorkspaceCrates",
      ),
    ],
    seamPorts: [
      {
        id: "port.crates.in",
        name: "workspace-manifests",
        channel: "control",
      },
      {
        id: "port.crates.out",
        name: "validated-topology",
        channel: "report_result",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.clock-seam",
    tier: 2,
    parentId: "node.runtime-composition",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Clock seam",
      developer: "Clock trait seam",
      maintainer: "Clock with RealClock and SimClock",
    },
    summary: {
      executive: "Provides shared timing semantics.",
      developer: "Injects real or virtual time into runtime behavior.",
      maintainer: "Clock now_ns, sleep, and is_virtual contract.",
    },
    evidence: [
      source("crates/clock/src/clock.rs", 20, 41, "Clock"),
    ],
    seamPorts: [{ id: "port.clock.out", name: "time", channel: "control" }],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.request-sink-seam",
    tier: 2,
    parentId: "node.runtime-composition",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    title: {
      executive: "Transport seam",
      developer: "RequestSink seam",
      maintainer: "RequestSink, RequestObserver, and Dispatchable",
    },
    summary: {
      executive: "Carries dispatch and observation through one seam.",
      developer: "Normalizes tokens, usage, and terminal callbacks.",
      maintainer: "Transport-neutral dispatch and observer contracts.",
    },
    evidence: [
      source(
        "crates/loadgen-core/src/sink.rs",
        157,
        160,
        "RequestSink",
      ),
    ],
    seamPorts: [
      { id: "port.sink.out", name: "observer", channel: "telemetry" },
      { id: "port.sink.token", name: "output-token", channel: "token" },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dynamo-online-library-seam",
    tier: 3,
    parentId: "node.runtime-composition",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "library_seam" },
    flavors: ["dynamo_online"],
    title: {
      executive: "Dynamo online replay helper",
      developer: "Library-owned online replay helper",
      maintainer: "run_scheduled_backend_online",
    },
    summary: {
      executive: "The existing runner path invokes this shared library helper.",
      developer: "The helper is invoked by the existing feature-gated dynamo_offline runner pair.",
      maintainer: "The existing feature-gated pair calls run_scheduled_backend_online; only a distinct dynamo_online backend ID remains planned.",
    },
    evidence: [
      source(
        "crates/aiperf/src/dynosim.rs",
        4159,
        4192,
        "run_scheduled_backend_online",
      ),
    ],
    seamPorts: [
      {
        id: "port.dynamo.online.library.control",
        name: "runner-integration",
        channel: "control",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dynamo-online-runner-pair",
    tier: 3,
    parentId: "node.runner-protocol-registries",
    childIds: [],
    owner: "rust",
    status: { state: "planned", delivery: "runner_pair" },
    flavors: ["dynamo_online"],
    title: {
      executive: "Dynamo online runner pair",
      developer: "Planned runner backend and pair",
      maintainer: "Planned aiperf-runner dynamo_online integration",
    },
    summary: {
      executive: "A distinct Dynamo-online backend identity remains planned.",
      developer: "Tracks a future dedicated dynamo_online backend ID and registered pair.",
      maintainer: "Design-only dedicated identity; the built path uses dynamo_offline with replay_mode online.",
    },
    evidence: [design(redesignSpec)],
    seamPorts: [
      {
        id: "port.dynamo.online.runner",
        name: "planned-runner-pair",
        channel: "control",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dynamo-offline-runner-backend",
    tier: 1,
    parentId: "node.journey.http-grpc-dynamo-dispatch",
    childIds: [
      "node.dynamo-offline-sim-clock",
      "node.dynamo-offline-steppable-replay",
      "node.dynamo-offline-report-gate",
      "node.dynamo-online-replay-mode",
    ],
    owner: "rust",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline", "dynamo_online"],
    title: {
      executive: "Dynamo replay runner backend",
      developer: "Feature-gated Dynamo offline backend and pairs",
      maintainer: "DynosimBackendFactory with replay_mode",
    },
    summary: {
      executive: "Runs deterministic or wall-clock in-process replay in special builds.",
      developer: "The dynosim backend selects offline or online replay_mode.",
      maintainer: "One registered backend and pair family owns both clock axes.",
    },
    evidence: [
      source(
        "crates/runner/src/offline_execution.rs",
        98,
        103,
        "DYNOSIM_BACKEND_ID",
      ),
      source(
        "crates/runner/src/offline_execution.rs",
        830,
        846,
        "DynosimBackendFactory",
      ),
    ],
    seamPorts: [
      {
        id: "port.dynamo.offline.runner.control",
        name: "offline-control",
        channel: "control",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dynamo-online-replay-mode",
    tier: 2,
    parentId: "node.dynamo-offline-runner-backend",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_online"],
    title: {
      executive: "Dynamo online in-process replay",
      developer: "dynamo_offline replay_mode online",
      maintainer: "DynamoReplayModeSpec::Online execution branch",
    },
    summary: {
      executive: "Drives the in-process Dynamo engine under a real wall clock.",
      developer: "Uses the existing feature-gated backend and pair with replay_mode online.",
      maintainer: "Dispatches to run_scheduled_backend_online without registering a new backend ID.",
    },
    evidence: [
      source(
        "crates/runner/src/offline_execution.rs",
        229,
        249,
        "DynamoReplayModeSpec",
      ),
      source(
        "crates/runner/src/offline_execution.rs",
        1894,
        1923,
        "DynosimExecutor::execute_scheduled",
      ),
    ],
    seamPorts: [
      {
        id: "port.dynamo.online.replay.control",
        name: "wall-clock-replay",
        channel: "control",
      },
    ],
    audience: fullAudience,
    footnotes: [
      {
        executive: "This is not a dedicated Dynamo-online registry backend.",
        developer: "The registered backend ID remains dynosim.",
        maintainer: "A distinct dynamo_online backend/pair remains design-only.",
      },
    ],
  },
  {
    id: "node.dynamo-offline-sim-clock",
    tier: 2,
    parentId: "node.dynamo-offline-runner-backend",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline"],
    title: {
      executive: "Dynamo offline virtual time",
      developer: "SimClock event scheduler",
      maintainer: "Heap-ordered SimClock controls",
    },
    summary: {
      executive: "Advances deterministic simulation time without wall-clock delay.",
      developer: "Orders sleepers and engine events on integer nanoseconds.",
      maintainer: "Concrete SimClock owns next-event and advance controls.",
    },
    evidence: [
      source(
        "crates/clock/src/sim_clock.rs",
        56,
        100,
        "SimClock",
      ),
    ],
    seamPorts: [
      {
        id: "port.dynamo.offline.clock.control",
        name: "virtual-time",
        channel: "control",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dynamo-offline-steppable-replay",
    tier: 2,
    parentId: "node.dynamo-offline-runner-backend",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline"],
    title: {
      executive: "Dynamo SteppableReplay",
      developer: "In-process passive replay engine",
      maintainer: "SteppableReplay adapter and topology factory",
    },
    summary: {
      executive: "Runs the Dynamo performance model without sockets.",
      developer: "Builds single, aggregate, or disaggregate replay engines.",
      maintainer: "Injected SteppableReplay engine behind the offline factory seam.",
    },
    evidence: [
      source(
        "crates/aiperf/src/dynosim.rs",
        575,
        649,
        "OfflineEngineConfig::build_native",
      ),
    ],
    seamPorts: [
      {
        id: "port.dynamo.offline.replay.control",
        name: "engine-events",
        channel: "control",
      },
      {
        id: "port.dynamo.offline.replay.report",
        name: "dynamo-summary",
        channel: "report_result",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
  {
    id: "node.dynamo-offline-report-gate",
    tier: 2,
    parentId: "node.dynamo-offline-runner-backend",
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline"],
    title: {
      executive: "Dynamo offline report gate",
      developer: "Common-summary byte parity gate",
      maintainer: "finish_shared_metrics_enforcing",
    },
    summary: {
      executive: "Rejects simulations whose common summaries disagree.",
      developer: "Compares AIPerf and Dynamo common report bytes before return.",
      maintainer: "Exact serialization parity is enforced on every product return path.",
    },
    evidence: [
      source(
        "crates/aiperf/src/dynosim.rs",
        950,
        1019,
        "finish_shared_metrics_enforcing",
      ),
    ],
    seamPorts: [
      {
        id: "port.dynamo.offline.gate.report",
        name: "validated-summary",
        channel: "report_result",
      },
    ],
    audience: fullAudience,
    footnotes: [],
  },
] satisfies GraphNode[];

export const graphNodes = graphNodeSchema.array().parse(graphNodesRaw);

const journeyEdges: GraphEdge[] = journeyDefinitions
  .slice(0, -1)
  .map((definition, index): GraphEdge => {
    const next = journeyDefinitions[index + 1];
    return {
      id: `edge.journey.${index + 1}`,
      source: {
        nodeId: definition.id,
        portId: `${definition.id}.out`,
      },
      target: { nodeId: next.id, portId: `${next.id}.in` },
      channel: "control",
      status: { state: "built", delivery: "unconditional" },
      flavors: [...builtJourneyFlavors],
      protocol: "Canonical product lifecycle transition",
      evidence: next.evidence,
      footnotes: [],
    };
  });

const graphEdgesRaw = [
  ...journeyEdges,
  {
    id: "edge.runtime.dispatch.metrics",
    source: {
      nodeId: "node.runtime-composition",
      portId: "port.runtime.telemetry",
    },
    target: {
      nodeId: "node.metrics-telemetry",
      portId: "port.metrics.in",
    },
    channel: "telemetry",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    protocol: "RequestObserver callbacks",
    evidence: [
      source(
        "crates/loadgen-core/src/sink.rs",
        85,
        133,
        "RequestObserver",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.request-sink.token.metrics",
    source: {
      nodeId: "node.request-sink-seam",
      portId: "port.sink.token",
    },
    target: {
      nodeId: "node.metrics-telemetry",
      portId: "port.metrics.token",
    },
    channel: "token",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    protocol: "RequestObserver::on_token",
    evidence: [
      source(
        "crates/loadgen-core/src/sink.rs",
        81,
        104,
        "RequestObserver::on_token",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.dataset.to.endpoint",
    source: {
      nodeId: "node.dataset-segment-pipeline",
      portId: "port.dataset.out",
    },
    target: {
      nodeId: "node.endpoint-bindings-transports",
      portId: "port.endpoint.in",
    },
    channel: "request_data",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http", "native_grpc", "online_mock"],
    protocol: "Materialized request payloads",
    evidence: [
      source(
        "crates/dataset/src/materialize.rs",
        110,
        180,
        "SegmentItemsMaterializer",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.metrics.to.result",
    source: {
      nodeId: "node.metrics-telemetry",
      portId: "port.metrics.out",
    },
    target: {
      nodeId: "node.journey.result-returned-to-python",
      portId: "node.journey.result-returned-to-python.report",
    },
    channel: "report_result",
    status: { state: "built", delivery: "unconditional" },
    flavors: [...builtJourneyFlavors],
    protocol: "native-v2 report payload",
    evidence: [
      source(
        "crates/metrics/src/report.rs",
        1800,
        1840,
        "NativeReporter",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.dynamo.online.runner.plan",
    source: {
      nodeId: "node.dynamo-online-library-seam",
      portId: "port.dynamo.online.library.control",
    },
    target: {
      nodeId: "node.dynamo-online-runner-pair",
      portId: "port.dynamo.online.runner",
    },
    channel: "control",
    status: { state: "planned", delivery: "runner_pair" },
    flavors: ["dynamo_online"],
    protocol: "Planned runner pair registration path",
    evidence: [design(redesignSpec)],
    footnotes: [],
  },
  {
    id: "edge.dynamo.online.replay-mode",
    source: {
      nodeId: "node.dynamo-offline-runner-backend",
      portId: "port.dynamo.offline.runner.control",
    },
    target: {
      nodeId: "node.dynamo-online-replay-mode",
      portId: "port.dynamo.online.replay.control",
    },
    channel: "control",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_online"],
    protocol: "Existing dynosim pair with replay_mode=online",
    evidence: [
      source(
        "crates/runner/src/offline_execution.rs",
        229,
        249,
        "DynamoReplayModeSpec",
      ),
      source(
        "crates/runner/src/offline_execution.rs",
        1894,
        1923,
        "DynosimExecutor::execute_scheduled",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.dynamo.offline.runner.sim-clock",
    source: {
      nodeId: "node.dynamo-offline-runner-backend",
      portId: "port.dynamo.offline.runner.control",
    },
    target: {
      nodeId: "node.dynamo-offline-sim-clock",
      portId: "port.dynamo.offline.clock.control",
    },
    channel: "control",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline"],
    protocol: "Feature-gated virtual clock construction",
    evidence: [
      source(
        "crates/runner/src/offline_execution.rs",
        830,
        846,
        "DynosimBackendFactory",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.dynamo.offline.sim-clock.replay",
    source: {
      nodeId: "node.dynamo-offline-sim-clock",
      portId: "port.dynamo.offline.clock.control",
    },
    target: {
      nodeId: "node.dynamo-offline-steppable-replay",
      portId: "port.dynamo.offline.replay.control",
    },
    channel: "control",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline"],
    protocol: "Discrete-event engine advancement",
    evidence: [
      source(
        "crates/graph/src/runtime.rs",
        192,
        280,
        "drive_sim_with_source",
      ),
    ],
    footnotes: [],
  },
  {
    id: "edge.dynamo.offline.replay.report-gate",
    source: {
      nodeId: "node.dynamo-offline-steppable-replay",
      portId: "port.dynamo.offline.replay.report",
    },
    target: {
      nodeId: "node.dynamo-offline-report-gate",
      portId: "port.dynamo.offline.gate.report",
    },
    channel: "report_result",
    status: { state: "built", delivery: "feature_gated" },
    flavors: ["dynamo_offline"],
    protocol: "Complete common-summary byte comparison",
    evidence: [
      source(
        "crates/aiperf/src/dynosim.rs",
        950,
        1019,
        "finish_shared_metrics_enforcing",
      ),
    ],
    footnotes: [],
  },
] satisfies GraphEdge[];

export const graphEdges = graphEdgeSchema.array().parse(graphEdgesRaw);

const graphScenesRaw = [
  {
    id: "scene.runtime-composition",
    title: "Runtime composition",
    rustScene: true,
    nodeIds: [
      "node.journey.python-config-load",
      "node.journey.config-v2-resolution",
      "node.journey.authored-request-projection",
      "node.journey.runner-spawn",
      "node.journey.strict-jsonl-validation",
      "node.journey.frozen-runner-application",
      "node.journey.workload-preparation",
      "node.journey.scheduling-or-graph-ir",
      "node.journey.dataset-materialization",
      "node.journey.endpoint-binding",
      "node.journey.http-grpc-dynamo-dispatch",
      "node.journey.observer-callbacks",
      "node.journey.metrics-and-reporting",
      "node.journey.result-returned-to-python",
      "node.runtime-composition",
      "node.clock-seam",
      "node.request-sink-seam",
      "node.dataset-segment-pipeline",
      "node.endpoint-bindings-transports",
      "node.metrics-telemetry",
      "node.dynamo-online-library-seam",
      "node.dynamo-offline-runner-backend",
      "node.dynamo-online-replay-mode",
      "node.dynamo-offline-sim-clock",
      "node.dynamo-offline-steppable-replay",
      "node.dynamo-offline-report-gate",
    ],
    edgeIds: [
      "edge.journey.1",
      "edge.journey.2",
      "edge.journey.3",
      "edge.journey.4",
      "edge.journey.5",
      "edge.journey.6",
      "edge.journey.7",
      "edge.journey.8",
      "edge.journey.9",
      "edge.journey.10",
      "edge.journey.11",
      "edge.journey.12",
      "edge.journey.13",
      "edge.dataset.to.endpoint",
      "edge.runtime.dispatch.metrics",
      "edge.request-sink.token.metrics",
      "edge.metrics.to.result",
      "edge.dynamo.online.replay-mode",
      "edge.dynamo.offline.runner.sim-clock",
      "edge.dynamo.offline.sim-clock.replay",
      "edge.dynamo.offline.replay.report-gate",
    ],
    audience: fullSceneAudience,
  },
  {
    id: "scene.runner-protocol-registries",
    title: "Runner protocol and registries",
    rustScene: true,
    nodeIds: [
      "node.runner-protocol-registries",
      "node.dynamo-online-library-seam",
      "node.dynamo-online-runner-pair",
      "node.dynamo-offline-runner-backend",
    ],
    edgeIds: ["edge.dynamo.online.runner.plan"],
    audience: fullSceneAudience,
  },
  {
    id: "scene.scheduling-phase-lifecycle",
    title: "Scheduling and phase lifecycle",
    rustScene: true,
    nodeIds: [
      "node.scheduling-phase-lifecycle",
      "node.journey.scheduling-or-graph-ir",
      "node.journey.dataset-materialization",
    ],
    edgeIds: ["edge.journey.8"],
    audience: fullSceneAudience,
  },
  {
    id: "scene.dataset-segment-pipeline",
    title: "Dataset and segment pipeline",
    rustScene: true,
    nodeIds: [
      "node.dataset-segment-pipeline",
      "node.journey.dataset-materialization",
      "node.journey.endpoint-binding",
      "node.endpoint-bindings-transports",
    ],
    edgeIds: ["edge.journey.9", "edge.dataset.to.endpoint"],
    audience: fullSceneAudience,
  },
  {
    id: "scene.endpoint-bindings-transports",
    title: "Endpoint bindings and HTTP/gRPC transports",
    rustScene: true,
    nodeIds: [
      "node.endpoint-bindings-transports",
      "node.journey.endpoint-binding",
      "node.journey.http-grpc-dynamo-dispatch",
      "node.dataset-segment-pipeline",
    ],
    edgeIds: ["edge.journey.10", "edge.dataset.to.endpoint"],
    audience: fullSceneAudience,
  },
  {
    id: "scene.graph-ir-execution",
    title: "Graph-IR execution",
    rustScene: true,
    nodeIds: [
      "node.graph-ir-execution",
      "node.journey.scheduling-or-graph-ir",
      "node.journey.dataset-materialization",
    ],
    edgeIds: ["edge.journey.8"],
    audience: fullSceneAudience,
  },
  {
    id: "scene.metrics-telemetry",
    title: "Metrics and telemetry",
    rustScene: true,
    nodeIds: [
      "node.metrics-telemetry",
      "node.journey.metrics-and-reporting",
      "node.journey.result-returned-to-python",
      "node.runtime-composition",
    ],
    edgeIds: [
      "edge.journey.13",
      "edge.runtime.dispatch.metrics",
      "edge.metrics.to.result",
    ],
    audience: fullSceneAudience,
  },
  {
    id: "scene.accuracy-evaluator-hosting",
    title: "Accuracy and evaluator hosting",
    rustScene: true,
    nodeIds: [
      "node.accuracy-evaluator-hosting",
      "node.metrics-telemetry",
      "node.journey.result-returned-to-python",
    ],
    edgeIds: ["edge.metrics.to.result"],
    audience: fullSceneAudience,
  },
  {
    id: "scene.crate-dependency-topology",
    title: "Crate dependency topology",
    rustScene: true,
    nodeIds: [
      "node.crate-dependency-topology",
      "node.runner-protocol-registries",
      "node.runtime-composition",
      "node.graph-ir-execution",
      "node.metrics-telemetry",
      "node.dataset-segment-pipeline",
      "node.endpoint-bindings-transports",
    ],
    edgeIds: ["edge.runtime.dispatch.metrics", "edge.dataset.to.endpoint"],
    audience: fullSceneAudience,
  },
] satisfies GraphScene[];

export const graphScenes = graphSceneSchema.array().parse(graphScenesRaw);
