// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ArchitectureRisk } from "../domain/architecture";
import { copy, evidence, rangedEvidence } from "./helpers";

export const parityLedger: ArchitectureRisk[] = [
  {
    id: "risk.graph-transport-limits",
    kind: "risk",
    status: "unbuilt",
    severity: "medium",
    title: copy("Graph protocol limits", "Graph transport gaps", "Chat-shaped HTTP GraphSink constraints"),
    summary: copy(
      "Branching journeys cannot yet use every protocol or exact-content path.",
      "Graph-IR is executable over HTTP and offline, but not gRPC; raw-token and broad multimodal graph inputs remain unavailable.",
      "Direct Graph-IR has no raw-token handle, rejects requires_raw_token_ids endpoints, and its online binding remains chat-shaped HTTP.",
    ),
    componentIds: ["component.graph-ir", "component.grpc-transport", "component.exact-token-ids"],
    evidence: [evidence("crates/graph/src/materialize.rs"), evidence("crates/runner/src/registry.rs")],
  },
  {
    id: "risk.grpc-sidecar-readiness",
    kind: "risk",
    status: "unbuilt",
    severity: "medium",
    title: copy("RPC feature boundary", "gRPC lifecycle limits", "No readiness retries, sidecars, or graph pair"),
    summary: copy(
      "Native RPC covers scheduled inference but not all web-path features.",
      "Readiness retries, every authored sidecar, and graph workloads are rejected.",
      "online_grpc requires wait_for_model_timeout <= 0 and rejects content, GPU, network, server-metrics, and live-streaming sidecars before execution.",
    ),
    componentIds: ["component.grpc-transport", "component.content-server"],
    evidence: [
      evidence("crates/transport-grpc/src/binding.rs"),
      rangedEvidence("crates/runner/src/grpc_execution.rs", 164, 195),
    ],
  },
  {
    id: "risk.offline-semantic-limits",
    kind: "risk",
    status: "feature-gated",
    severity: "medium",
    title: copy("Simulation boundary", "Offline mode constraints", "Single-worker timing-only SteppableReplay"),
    summary: copy(
      "Offline studies model timing and topology, not answer quality or external delivery behavior.",
      "The feature-gated backend is single-worker, has one in-process endpoint, and rejects semantic accuracy and sidecars.",
      "dynamo_offline uses SimClock without sockets; no evaluator text, content server, telemetry sidecar, or multiworker placement is implied.",
    ),
    componentIds: ["component.dynamo-offline", "component.worker-placement", "component.static-accuracy"],
    evidence: [evidence("crates/aiperf/src/dynosim.rs"), evidence("crates/runner/src/offline_execution.rs")],
  },
  {
    id: "risk.provider-evaluation-scope",
    kind: "risk",
    status: "runtime-conditional",
    severity: "high",
    title: copy("Evaluation preview scope", "Conditional evaluator availability", "Two attested five-record GSM8K manifests only"),
    summary: copy(
      "The neutral evaluator path is intentionally too narrow to replace established benchmark systems.",
      "It appears only with verified provider roots and Linux isolation and supports two bounded canaries.",
      "NeMo 0.4.0 and OpenBench 0.5.3 plus Inspect 0.3.141 each expose the frozen five-record GSM8K canary; arbitrary tasks and effects fail closed.",
    ),
    componentIds: ["component.provider-evaluation"],
    evidence: [evidence("crates/runner/src/stock_evaluation.rs"), evidence("crates/accuracy/src/isolation.rs")],
  },
  {
    id: "risk.legacy-evaluation",
    kind: "risk",
    status: "legacy-parallel",
    severity: "medium",
    title: copy("Dual evaluation ownership", "Legacy Python semantics remain", "AccuracyEvaluator and AgenticHarness migration gate"),
    summary: copy(
      "Quality semantics remain split by design until exact parity evidence exists.",
      "Static and agentic Python workers remain canonical for benchmark datasets, prompts, loops, tools, hidden tests, and scoring.",
      "Rust brokers model I/O and metrics but deliberately does not reimplement benchmark graders or environment semantics.",
    ),
    componentIds: ["component.static-accuracy", "component.agentic-evaluation", "component.python-evaluators"],
    evidence: [evidence("crates/accuracy/src/lib.rs"), evidence("crates/aiperf/src/agentic.rs")],
  },
  {
    id: "risk.protocol-v1",
    kind: "risk",
    status: "compatibility-only",
    severity: "low",
    title: copy("Historical wire format", "Protocol-v1 isolation", "Compatibility decoder outside product path"),
    summary: copy(
      "Older native integrations are preserved without complicating the supported product route.",
      "Protocol v1 can decode compatibility inputs, but Python never resolves, converts, or falls back to it.",
      "The v1 authority remains isolated in runner compatibility code while Config v2 launches strict authored protocol v2 only.",
    ),
    componentIds: ["component.python-frontend", "component.rust-runner"],
    evidence: [evidence("crates/runner/src/protocol.rs"), evidence("crates/runner/src/protocol_v2.rs")],
  },
  {
    id: "risk.compatibility-export",
    kind: "risk",
    status: "unbuilt",
    severity: "medium",
    title: copy("Export compatibility gap", "Legacy output gaps", "No genai-perf-v1, general Rust CSV, or insights exporter"),
    summary: copy(
      "Some downstream formats and automated guidance still require Python or future work.",
      "Native-v2 JSON is built, while general metric CSV, genai-perf-v1 compatibility, warnings, insights, and console replay are not.",
      "aiperf-metrics Reporter is typed and IO-free; exporter-specific serializers beyond runner native-v2 persistence remain absent.",
    ),
    componentIds: ["component.native-metrics", "component.python-frontend"],
    evidence: [evidence("crates/metrics/src/report.rs"), evidence("AGENTS.md")],
  },
];
