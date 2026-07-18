// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  architectureCatalogSchema,
  type ArchitectureView,
} from "../domain/architecture";
import { crateCatalog } from "./crates";
import { dataPlaneComponents, dataPlaneEdges } from "./data-plane";
import {
  executionComponents,
  executionEdges,
  executionPairSupport,
} from "./execution";
import { copy } from "./helpers";
import { oneRunLifecycle } from "./journey";
import {
  observabilityComponents,
  observabilityEdges,
} from "./observability";
import { ownershipComponents, ownershipEdges } from "./ownership";
import { parityLedger } from "./parity";
import { graphEdges, graphNodes, graphScenes } from "./scenes/graph-catalog";

const allComponents = [
  ...ownershipComponents,
  ...executionComponents,
  ...dataPlaneComponents,
  ...observabilityComponents,
];
const allArchitectureEdges = [
  ...ownershipEdges,
  ...executionEdges,
  ...dataPlaneEdges,
  ...observabilityEdges,
];

const views: ArchitectureView[] = [
  {
    id: "view.ownership",
    kind: "view",
    route: "/",
    title: copy("Who owns what", "System ownership", "Canonical product boundary"),
    summary: copy(
      "Shows accountable product, execution, external, and retained evaluation boundaries.",
      "Maps Python authoring to the strict runner, reusable runtime, inference peers, and legacy evaluator workers.",
      "Separates Config-v2 product truth, RunnerApplication execution, library seams, and compatibility providers.",
    ),
    componentIds: ownershipComponents.map(({ id }) => id),
    edgeIds: ownershipEdges.map(({ id }) => id),
    riskIds: ["risk.legacy-evaluation", "risk.protocol-v1"],
  },
  {
    id: "view.journey",
    kind: "view",
    route: "/journey",
    title: copy("One run from choice to evidence", "Protocol-v2 run journey", "Authored-v2 validate and execute lifecycle"),
    summary: copy(
      "Follows one decision through availability checks, execution, measurement, and presentation.",
      "Connects Config v2, capability preflight, strict validation, pair execution, and native reporting.",
      "Pins the exact-image RunnerApplication lifecycle without protocol-v1 conversion or fallback.",
    ),
    componentIds: ["component.python-frontend", "component.rust-runner", "component.rust-runtime", "component.inference-target", "component.native-metrics"],
    edgeIds: ["edge.python-launches-runner", "edge.runner-composes-runtime", "edge.runtime-dispatches-target", "edge.runtime-native-metrics"],
    riskIds: ["risk.protocol-v1"],
  },
  {
    id: "view.execution",
    kind: "view",
    route: "/execution",
    title: copy("Ways to run", "Execution modes and controls", "Clock, scheduling, transport, and placement matrix"),
    summary: copy(
      "Compares real web, native RPC, online mock, and deterministic simulation choices.",
      "Shows shared scheduling and lifecycle contracts alongside mode-specific transports and limits.",
      "Details RealClock/SimClock selection, LocalSet placement, feature gates, pair registration, and actuator reach.",
    ),
    componentIds: executionComponents.map(({ id }) => id),
    edgeIds: executionEdges.map(({ id }) => id),
    riskIds: ["risk.grpc-sidecar-readiness", "risk.offline-semantic-limits"],
  },
  {
    id: "view.data-plane",
    kind: "view",
    route: "/data-plane",
    title: copy("How requests take shape", "Dataset and endpoint data plane", "Segment, materializer, endpoint, graph, media, and token paths"),
    summary: copy(
      "Explains how authored content becomes exact requests for diverse inference services.",
      "Connects loaders, content-addressed segments, endpoint preparation, graph execution, media publication, and token arrays.",
      "Surfaces prefix-dependent hashes, frozen registries, GraphTracePlan constraints, sidecar eligibility, and raw-token bypass.",
    ),
    componentIds: dataPlaneComponents.map(({ id }) => id),
    edgeIds: dataPlaneEdges.map(({ id }) => id),
    riskIds: ["risk.graph-transport-limits", "risk.grpc-sidecar-readiness"],
  },
  {
    id: "view.observability",
    kind: "view",
    route: "/observability",
    title: copy("Evidence and quality", "Observability and evaluation", "Native accumulators, telemetry producers, archive, and evaluator boundaries"),
    summary: copy(
      "Links performance, infrastructure context, durable telemetry, and model quality.",
      "Shows native request metrics, side channels, archive durability, legacy evaluators, and conditional neutral providers.",
      "Pins observer facts, sweep kernels, telemetry finalize joins, WAL contracts, evaluator isolation, and migration gates.",
    ),
    componentIds: observabilityComponents.map(({ id }) => id),
    edgeIds: observabilityEdges.map(({ id }) => id),
    riskIds: ["risk.provider-evaluation-scope", "risk.legacy-evaluation", "risk.compatibility-export"],
  },
  {
    id: "view.parity",
    kind: "view",
    route: "/parity",
    title: copy("What is ready and what is not", "Parity and migration ledger", "Built, conditional, compatibility, legacy, and unbuilt surfaces"),
    summary: copy(
      "Makes capability boundaries and investment risks visible without overstating roadmap intent.",
      "Separates executable paths from feature gates, runtime attestation, retained Python semantics, compatibility decoders, and gaps.",
      "Grounds graph, gRPC, offline, evaluator, protocol, and exporter status in current source evidence.",
    ),
    componentIds: allComponents.map(({ id }) => id),
    edgeIds: [],
    riskIds: parityLedger.map(({ id }) => id),
  },
  {
    id: "view.atlas",
    kind: "view",
    route: "/atlas",
    title: copy("Complete system map", "Unified architecture atlas", "Validated source and crate topology"),
    summary: copy(
      "Combines ownership, execution, content, evidence, and risk in one navigable source of truth.",
      "Provides linked entities, pair support, lifecycle stages, and a complete workspace crate catalog.",
      "The schema validates cross-module IDs, source evidence, Cargo identities, route coverage, vocabularies, and audience copy.",
    ),
    componentIds: allComponents.map(({ id }) => id),
    edgeIds: allArchitectureEdges.map(({ id }) => id),
    riskIds: parityLedger.map(({ id }) => id),
  },
];

export const architectureCatalog = architectureCatalogSchema.parse({
  schemaVersion: 2,
  components: allComponents,
  edges: allArchitectureEdges,
  risks: parityLedger,
  lifecycleStages: oneRunLifecycle,
  views,
  crates: crateCatalog,
  pairSupport: executionPairSupport,
  graphNodes,
  graphEdges,
  graphScenes,
});
