// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureStatus,
  CargoDependencyKind,
  CrateReference,
  ExecutionMode,
} from "../domain/architecture";
import { copy, evidence } from "./helpers";

interface CrateDefinition {
  name: string;
  value: string;
  integration: string;
  maintenance: string;
  key?: string[];
  dependencies?: Array<
    | string
    | {
        name: string;
        kind: CargoDependencyKind;
      }
  >;
  contracts: string[];
  modes?: ExecutionMode[];
  scars?: string[];
  status?: ArchitectureStatus;
}

function crateReference(definition: CrateDefinition): CrateReference {
  // Cargo package names keep the `aiperf-` prefix, but the on-disk crate
  // directories were shortened to `crates/<capability>`; strip the prefix for
  // the source path while leaving the package name / ids intact.
  const directory = definition.name.startsWith("aiperf-")
    ? definition.name.slice("aiperf-".length)
    : definition.name;
  const path = `crates/${directory}`;
  return {
    id: `crate.${definition.name}`,
    kind: "crate",
    packageName: definition.name,
    path,
    status: definition.status ?? "built",
    title: copy(
      `${definition.value} capability`,
      `${definition.name} integration`,
      `${definition.name} package`,
    ),
    summary: copy(
      definition.value,
      definition.integration,
      definition.maintenance,
    ),
    responsibility: copy(
      `Delivers ${definition.value.toLocaleLowerCase()}`,
      `Provides ${definition.integration.toLocaleLowerCase()}`,
      `Owns ${definition.maintenance}`,
    ),
    keySourcePaths: (definition.key ?? ["src/lib.rs"]).map(
      (source) => `${path}/${source}`,
    ),
    dependencies: (definition.dependencies ?? []).map((dependency) => ({
      crateId: `crate.${typeof dependency === "string" ? dependency : dependency.name}`,
      kind: typeof dependency === "string" ? "normal" : dependency.kind,
    })),
    contracts: definition.contracts,
    modes: definition.modes ?? [],
    parityScars: definition.scars ?? [],
    evidence: [evidence(`${path}/Cargo.toml`), evidence(`${path}/src/lib.rs`)],
  };
}

export const crateCatalog: CrateReference[] = [
  crateReference({
    name: "aiperf-clock",
    value: "Consistent real and virtual time",
    integration: "Injects monotonic timing into schedulers and transports",
    maintenance: "Clock, RealClock, SimClock, and integer-nanosecond sleeper ordering",
    key: ["src/lib.rs", "src/clock.rs", "src/sim_clock.rs"],
    contracts: ["Clock"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["Virtual advance controls remain inherent on SimClock"],
  }),
  crateReference({
    name: "loadgen-core",
    value: "Transport-neutral request observation",
    integration: "Defines dispatch and terminal measurement callbacks",
    maintenance: "Dispatchable, RequestSink, RequestObserver, ObservedUsage, and TraceCollector",
    key: ["src/lib.rs", "src/sink.rs", "src/collector.rs"],
    contracts: ["RequestSink<R>", "RequestObserver", "Dispatchable"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["Observers are deliberately !Send and !Sync compatible"],
  }),
  crateReference({
    name: "aiperf-timing",
    value: "Load pacing and bounded run lifecycle",
    integration: "Supplies arrivals, slots, stops, phases, ramps, cancellation, and URL selection",
    maintenance: "IntervalGenerator, SlotPool, StopChecker, RampDriver, and phase modules",
    key: ["src/lib.rs", "src/slots.rs", "src/ramping.rs", "src/phase/mod.rs"],
    dependencies: ["aiperf-clock", "aiperf-rng"],
    contracts: ["phase lifecycle", "admission policy"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
  }),
  crateReference({
    name: "aiperf-adaptive",
    value: "SLA-driven scale discovery",
    integration: "Adjusts live sessions, prefill, rate, or users from metric windows",
    maintenance: "ControlActuator, SlaEvaluator, StepPolicy, RampUntilFailController, and artifacts",
    key: ["src/lib.rs", "src/actuator.rs", "src/controller.rs"],
    dependencies: ["aiperf-clock", "aiperf-metrics", "aiperf-timing", "loadgen-core"],
    contracts: ["adaptive actuator", "schema-v2 adaptive artifacts"],
    modes: ["online_http", "dynamo_offline", "online_mock"],
  }),
  crateReference({
    name: "aiperf-rng",
    value: "Order-independent reproducibility",
    integration: "Derives named deterministic sampling streams",
    maintenance: "RngRoot BLAKE3 derivation, Pcg64 generator, and sampler traits",
    key: ["src/lib.rs", "src/derive.rs", "src/generator.rs"],
    contracts: ["RngRoot", "SamplingRng"],
    modes: ["online_http", "dynamo_offline", "online_mock"],
    scars: ["Semantics are deterministic but not cross-language byte parity"],
  }),
  crateReference({
    name: "aiperf-dataset",
    value: "Unified request content and segment storage",
    integration: "Loads, composes, validates, samples, and materializes endpoint inputs",
    maintenance: "loader registry, SegmentStore, PromptMaterializer, tokenizer, fetch, and media seams",
    key: ["src/lib.rs", "src/compose.rs", "src/segment.rs", "src/materialize.rs"],
    dependencies: ["aiperf-clock", "aiperf-endpoints", "aiperf-rng", "aiperf-transport-http"],
    contracts: ["DatasetLoader", "SegmentStore", "PromptMaterializer"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["Recorded graph formats bypass the linear loader registry"],
  }),
  crateReference({
    name: "aiperf-content-server",
    value: "Confined generated-media delivery",
    integration: "Publishes image and video files for eligible HTTP workloads",
    maintenance: "SyntheticMediaPublisher, Axum server, response tracker, and shutdown lifecycle",
    key: ["src/lib.rs", "src/publisher.rs", "src/server.rs"],
    dependencies: ["aiperf-dataset"],
    contracts: ["SyntheticMediaPublisher"],
    modes: ["online_http", "online_mock"],
    scars: ["gRPC, offline, agentic, and evaluation reject the sidecar"],
  }),
  crateReference({
    name: "aiperf-mock-rs",
    value: "Repeatable online inference target",
    integration: "Exercises ordinary HTTP and SSE network paths",
    maintenance: "standalone Axum routes, latency model, batch scheduler, cache, and synthetic telemetry",
    key: ["src/lib.rs", "src/main.rs", "src/app.rs"],
    dependencies: ["aiperf-rng"],
    contracts: ["ordinary inference server process"],
    modes: ["online_mock"],
    scars: ["Never supervised as a runner backend process"],
  }),
  crateReference({
    name: "aiperf-endpoints",
    value: "Open inference dialect portfolio",
    integration: "Prepares endpoint-specific HTTP and gRPC request/response behavior",
    maintenance: "EndpointId, factories, descriptors, aliases, PreparedEndpoint, and dialect implementations",
    key: ["src/lib.rs", "src/registry.rs", "src/endpoints.rs", "src/vllm_generate.rs"],
    contracts: ["EndpointFactory", "PreparedEndpoint"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["Protocol-v1 enum adapters are compatibility-only"],
  }),
  crateReference({
    name: "aiperf-metrics",
    value: "Native performance and quality evidence",
    integration: "Computes sparse metrics, sweep-lines, windows, and typed reports",
    maintenance: "MetricSpec catalog, ColumnStore, MetricsAccumulator, sweep kernels, and Reporter",
    key: ["src/lib.rs", "src/accumulator.rs", "src/catalog.rs", "src/report.rs"],
    contracts: ["MetricsAccumulator", "native-v2 Reporter"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["genai-perf-v1 and general metric CSV remain unbuilt"],
  }),
  crateReference({
    name: "aiperf-gpu-telemetry",
    value: "Accelerator energy and utilization context",
    integration: "Feeds DCGM and supervised Python GPU observations into native metrics",
    maintenance: "telemetry sources, parser, field scaling, collector, and energy joins",
    key: ["src/lib.rs", "src/collector.rs", "src/accumulator.rs"],
    dependencies: ["aiperf-clock", "aiperf-metrics", "aiperf-transport-http"],
    contracts: ["GPU telemetry accumulator"],
    modes: ["online_http", "online_grpc", "online_mock"],
  }),
  crateReference({
    name: "aiperf-prometheus",
    value: "Bounded lossless metrics exposition parsing",
    integration: "Parses Prometheus 0.0.4 and OpenMetrics 1.0 text for telemetry consumers",
    maintenance: "syntax, semantic, number, limits, compatibility, and role-matrix parsers",
    key: ["src/lib.rs", "src/parser.rs", "src/semantic.rs"],
    contracts: ["bounded exposition parser"],
    modes: ["online_http", "online_grpc", "online_mock"],
  }),
  crateReference({
    name: "aiperf-server-metrics",
    value: "Inference server metric context",
    integration: "Collects and derives Prometheus/OpenMetrics server observations",
    maintenance: "source, parser integration, histogram learning, atlas, units, and accumulator",
    key: ["src/lib.rs", "src/accumulator.rs", "src/histogram.rs"],
    dependencies: ["aiperf-clock", "aiperf-metrics", "aiperf-prometheus", "aiperf-transport-http"],
    contracts: ["server telemetry accumulator"],
    modes: ["online_http", "online_grpc", "online_mock"],
    scars: ["TRT-LLM fallback disables terminally when unsupported"],
  }),
  crateReference({
    name: "aiperf-network-latency",
    value: "Fresh network baseline context",
    integration: "Calibrates TCP-connect latency per target",
    maintenance: "Clock-injected probe, model, and side-channel accumulator",
    key: ["src/lib.rs", "src/probe.rs", "src/accumulator.rs"],
    dependencies: ["aiperf-clock", "aiperf-metrics"],
    contracts: ["network latency accumulator"],
    modes: ["online_http", "online_grpc", "online_mock"],
  }),
  crateReference({
    name: "aiperf-telemetry-archive",
    value: "Durable telemetry retention and recovery",
    integration: "Archives versioned telemetry frames with explicit loss and receipt accounting",
    maintenance: "WAL, manifests, recovery, Arrow/Parquet projection, stores, policy, and lifecycle",
    key: ["src/lib.rs", "src/wal.rs", "src/manifest.rs", "src/loss_ledger.rs"],
    dependencies: ["aiperf-clock", "aiperf-prometheus"],
    contracts: ["archive WAL", "manifest and loss ledger"],
    modes: ["online_http", "online_grpc", "online_mock"],
  }),
  crateReference({
    name: "aiperf-accuracy",
    value: "Controlled evaluator process boundary",
    integration: "Supervises legacy and provider-neutral evaluation workers",
    maintenance: "provider registry, protocol v2, isolation, artifacts, projection, and legacy worker seams",
    key: ["src/lib.rs", "src/provider.rs", "src/isolation.rs", "src/worker.rs"],
    contracts: ["AccuracyEvaluator", "evaluator-worker protocol v2"],
    modes: ["online_http"],
    scars: ["Legacy Python benchmark semantics remain canonical"],
  }),
  crateReference({
    name: "aiperf-extensions",
    value: "Compile-time product extension composition",
    integration: "Transactionally links dataset, sampler, and endpoint registrations",
    maintenance: "AiperfRegistry and AiperfExtension freeze path",
    key: ["src/lib.rs"],
    dependencies: [
      "aiperf-dataset",
      "aiperf-endpoints",
      { name: "aiperf-rng", kind: "dev" },
    ],
    contracts: ["AiperfExtension", "AiperfRegistry"],
    modes: ["online_http", "online_grpc", "online_mock"],
    scars: ["No runtime discovery, plugins.yaml, or dynamic-library ABI"],
  }),
  crateReference({
    name: "aiperf-core",
    value: "Shared request measurement helpers",
    integration: "Supplies chat body, SSE chunk, and collector observer utilities",
    maintenance: "ChatChunk, chat_request_body, and CollectorObserver",
    key: ["src/lib.rs", "src/chat.rs", "src/sse.rs", "src/observer.rs"],
    dependencies: ["loadgen-core"],
    contracts: ["CollectorObserver"],
    modes: ["online_http", "online_mock"],
    scars: ["Owns no transport client or clock"],
  }),
  crateReference({
    name: "aiperf-transport-http",
    value: "Native web inference transport",
    integration: "Executes prepared HTTP and SSE lifecycle bindings",
    maintenance: "Hyper connection clients, endpoint bindings, SSE parser, traces, reuse, and cancellation",
    key: ["src/lib.rs", "src/transport/endpoint_binding.rs"],
    dependencies: ["aiperf-clock", "aiperf-endpoints"],
    contracts: ["HttpEndpointBinding", "RequestSink<HttpRequest>"],
    modes: ["online_http", "online_mock"],
  }),
  crateReference({
    name: "aiperf-transport-grpc",
    value: "Native KServe and Riva RPC transport",
    integration: "Executes prepared unary and streaming gRPC bindings",
    maintenance: "Tonic transport, binding factories, KServe/Riva codecs, status, traces, and cancellation",
    key: ["src/lib.rs", "src/binding.rs", "src/transport.rs"],
    dependencies: [
      "aiperf-clock",
      "aiperf-endpoints",
      { name: "aiperf-clock", kind: "dev" },
    ],
    contracts: ["GrpcEndpointBinding", "RequestSink<GrpcRequest>"],
    modes: ["online_grpc"],
    scars: ["Readiness retries, every sidecar, and the graph pair are rejected"],
  }),
  crateReference({
    name: "aiperf-graph",
    value: "Branching asynchronous workload execution",
    integration: "Compiles and runs DAG, WEKA, and Dynamo trace plans",
    maintenance: "GraphTracePlan, lowering, executor, placement, materialization, and real/sim drivers",
    key: ["src/lib.rs", "src/input.rs", "src/executor.rs", "src/runtime.rs"],
    dependencies: ["aiperf-core", "aiperf-dataset", "aiperf-metrics", "aiperf-rng", "aiperf-timing", "aiperf-transport-http", "aiperf-clock", "loadgen-core"],
    contracts: ["GraphSink", "GraphTracePlan"],
    modes: ["online_http", "dynamo_offline", "online_mock"],
    scars: ["Online graph is chat-shaped HTTP and lacks raw-token handles"],
  }),
  crateReference({
    name: "aiperf-runner",
    value: "Strict one-run product executable",
    integration: "Validates and executes exact-image backend/workload pairs",
    maintenance: "RunnerApplication, protocol v2 coordinator, registries, pair factories, and execution adapters",
    key: ["src/lib.rs", "src/main.rs", "src/protocol_v2.rs", "src/registry.rs"],
    dependencies: ["aiperf", "aiperf-accuracy", "aiperf-adaptive", "aiperf-clock", "aiperf-content-server", "aiperf-dataset", "aiperf-endpoints", "aiperf-extensions", "aiperf-gpu-telemetry", "aiperf-graph", "aiperf-metrics", "aiperf-network-latency", "aiperf-prometheus", "aiperf-rng", "aiperf-server-metrics", "aiperf-telemetry-archive", "aiperf-timing", "aiperf-transport-http", "aiperf-transport-grpc", "loadgen-core"],
    contracts: ["strict JSONL protocol v2", "RunnerApplication"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["Protocol v1 remains compatibility-only"],
  }),
  crateReference({
    name: "aiperf",
    value: "Reusable run-time composition",
    integration: "Connects scheduling, transports, datasets, metrics, accuracy, and evaluation",
    maintenance: "library-only online/offline dispatch, workloads, observers, evaluation, and report persistence",
    key: ["src/lib.rs", "src/scheduled.rs", "src/http.rs", "src/grpc.rs", "src/report.rs"],
    dependencies: ["aiperf-accuracy", "aiperf-adaptive", "aiperf-core", "aiperf-dataset", "aiperf-endpoints", "aiperf-graph", "aiperf-metrics", "aiperf-rng", "aiperf-timing", "aiperf-transport-http", "aiperf-transport-grpc", "aiperf-clock", "loadgen-core"],
    contracts: ["scheduled runtime composition", "evaluation arbiter"],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    scars: ["Library package intentionally has no binary target"],
  }),
];
