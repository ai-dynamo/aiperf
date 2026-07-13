// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureComponent,
  ArchitectureEdge,
} from "../domain/architecture";
import { copy, evidence } from "./helpers";

export const ownershipComponents: ArchitectureComponent[] = [
  {
    id: "component.python-frontend",
    kind: "component",
    owner: "python",
    lifecycleBand: "authoring",
    status: "built",
    title: copy(
      "Product control room",
      "Python configuration and orchestration",
      "Config-v2 Python frontend",
    ),
    summary: copy(
      "Owns the human workflow, configuration choices, outer run loops, and presentation.",
      "Validates authored configuration, preflights runner capabilities, launches one child per run, and presents results.",
      "The canonical path is src/aiperf/config plus orchestrator and cli_runner; it projects strict protocol-v2 input with no v1 fallback.",
    ),
    evidence: [evidence("AGENTS.md"), evidence("src/aiperf/cli_runner/_single_run.py")],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    contracts: ["Config v2", "strict runner protocol v2"],
    crateIds: [],
  },
  {
    id: "component.rust-runner",
    kind: "component",
    owner: "rust",
    lifecycleBand: "validation",
    status: "built",
    title: copy(
      "Single-run execution engine",
      "Strict native runner boundary",
      "aiperf-runner application",
    ),
    summary: copy(
      "Owns one run’s high-performance execution and returns source-grounded measurements.",
      "Freezes registries at bootstrap, validates one backend/workload pair, executes it, and writes native-v2 results.",
      "RunnerApplication binds protocol_v2, pair factories, endpoint/input registries, execution adapters, and mimalloc in crates/runner.",
    ),
    evidence: [
      evidence("crates/runner/src/main.rs"),
      evidence("crates/runner/src/lib.rs"),
    ],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    contracts: ["JSONL protocol v2", "exact-image capability inventory"],
    crateIds: ["crate.aiperf-runner"],
  },
  {
    id: "component.rust-runtime",
    kind: "component",
    owner: "rust",
    lifecycleBand: "execution",
    status: "built",
    title: copy(
      "Reusable performance core",
      "Clock and transport-neutral runtime",
      "aiperf library composition",
    ),
    summary: copy(
      "Keeps scheduling, measurement, and request execution aligned across real and simulated runs.",
      "Composes workload scheduling, prepared transports, observers, reporting, and evaluation behind injected seams.",
      "The library-only crates/aiperf package consumes Clock and RequestSink boundaries and deliberately has no binary target.",
    ),
    evidence: [evidence("crates/aiperf/src/lib.rs"), evidence("crates/loadgen-core/src/sink.rs")],
    modes: ["online_http", "online_grpc", "dynamo_offline", "online_mock"],
    contracts: ["Clock", "RequestSink and RequestObserver"],
    crateIds: ["crate.aiperf", "crate.loadgen-core"],
  },
  {
    id: "component.inference-target",
    kind: "component",
    owner: "external",
    lifecycleBand: "execution",
    status: "built",
    title: copy(
      "Inference destination",
      "HTTP, gRPC, or mock inference target",
      "Prepared endpoint transport peer",
    ),
    summary: copy(
      "Supplies model responses while AIPerf remains responsible for load and measurement.",
      "Accepts OpenAI-compatible, Anthropic, KServe, or Riva requests over the selected transport.",
      "Online targets are external peers; aiperf-mock-rs is a standalone ordinary HTTP target and is never runner-supervised.",
    ),
    evidence: [evidence("crates/mock-rs/src/main.rs"), evidence("crates/endpoints/src/lib.rs")],
    modes: ["online_http", "online_grpc", "online_mock"],
    contracts: ["endpoint dialect", "HTTP/SSE or gRPC"],
    crateIds: ["crate.aiperf-mock-rs", "crate.aiperf-endpoints"],
  },
  {
    id: "component.python-evaluators",
    kind: "component",
    owner: "legacy",
    lifecycleBand: "measurement",
    status: "legacy-parallel",
    title: copy(
      "Established evaluation semantics",
      "Legacy Python accuracy and agent loops",
      "AccuracyEvaluator and AgenticHarness providers",
    ),
    summary: copy(
      "Preserves benchmark-owned prompts, tools, hidden tests, and scoring while migration remains deliberately bounded.",
      "A supervised Python worker owns static and stateful benchmark semantics; Rust owns normal inference scheduling and measurement.",
      "Lighteval static accuracy and pinned Harbor, AgentLab/BrowserGym, and MCPMark agentic providers remain canonical legacy-parallel paths.",
    ),
    evidence: [evidence("crates/accuracy/src/lib.rs"), evidence("crates/aiperf/src/accuracy.rs")],
    modes: ["online_http"],
    contracts: ["correlated JSONL evaluator protocol", "opaque task identity"],
    crateIds: ["crate.aiperf-accuracy", "crate.aiperf"],
  },
];

export const ownershipEdges: ArchitectureEdge[] = [
  {
    id: "edge.python-launches-runner",
    kind: "message",
    from: "component.python-frontend",
    to: "component.rust-runner",
    label: "Launch one authored run",
    protocol: "strict JSONL protocol v2",
    status: "built",
    evidence: [evidence("crates/runner/src/protocol_v2.rs")],
  },
  {
    id: "edge.runner-composes-runtime",
    kind: "dependency",
    from: "component.rust-runner",
    to: "component.rust-runtime",
    label: "Compose run-local execution",
    contract: "frozen registries and injected runtime seams",
    status: "built",
    evidence: [evidence("crates/runner/src/lib.rs")],
  },
  {
    id: "edge.runtime-dispatches-target",
    kind: "message",
    from: "component.rust-runtime",
    to: "component.inference-target",
    label: "Dispatch and observe requests",
    protocol: "prepared HTTP/SSE or gRPC binding",
    status: "built",
    evidence: [evidence("crates/loadgen-core/src/sink.rs")],
  },
  {
    id: "edge.runtime-evaluator-broker",
    kind: "control",
    from: "component.python-evaluators",
    to: "component.rust-runtime",
    label: "Broker model turns through Rust",
    control: "opaque correlated evaluator requests",
    status: "legacy-parallel",
    evidence: [evidence("crates/aiperf/src/agentic_gateway.rs")],
  },
];
