// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { architectureCatalog } from "../content";
import {
  architectureCatalogSchema,
  executionFlavorSchema,
  flowChannelSchema,
  tierSchema,
  type ArchitectureCatalog,
  type GraphEdge,
  type GraphNode,
} from "./architecture";
import { validateArchitectureCatalog } from "./integrity";

const repositoryRoot = pathToFileURL(
  `${resolve(dirname(fileURLToPath(import.meta.url)), "../../../../")}/`,
);

function minimalGraphCatalog(): ArchitectureCatalog {
  return architectureCatalogSchema.parse({
    schemaVersion: 2,
    components: [
      {
        id: "component.python",
        kind: "component",
        owner: "python",
        lifecycleBand: "authoring",
        status: "built",
        title: {
          executive: "Configuration front door",
          developer: "Python configuration boundary",
          maintainer: "Config-v2 Python projection",
        },
        summary: {
          executive: "Owns the product controls and presentation boundary.",
          developer: "Authors one strict run request and launches the native runner.",
          maintainer:
            "Projects protocol-v2 input without protocol-v1 fallback or resolved state.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 12 } }],
        modes: ["online_http"],
        contracts: ["protocol-v2"],
      },
    ],
    edges: [],
    risks: [],
    lifecycleStages: [],
    views: [
      {
        id: "view.journey",
        kind: "view",
        route: "/journey",
        title: {
          executive: "Journey map",
          developer: "Run journey",
          maintainer: "Journey route coverage",
        },
        summary: {
          executive: "Route coverage placeholder for tests.",
          developer: "Ensures required route checks pass in focused tests.",
          maintainer: "Minimal view to satisfy integrity route coverage.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
      {
        id: "view.execution",
        kind: "view",
        route: "/execution",
        title: {
          executive: "Execution map",
          developer: "Execution route",
          maintainer: "Execution route coverage",
        },
        summary: {
          executive: "Route coverage placeholder for tests.",
          developer: "Ensures required route checks pass in focused tests.",
          maintainer: "Minimal view to satisfy integrity route coverage.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
      {
        id: "view.data-plane",
        kind: "view",
        route: "/data-plane",
        title: {
          executive: "Data map",
          developer: "Data route",
          maintainer: "Data route coverage",
        },
        summary: {
          executive: "Route coverage placeholder for tests.",
          developer: "Ensures required route checks pass in focused tests.",
          maintainer: "Minimal view to satisfy integrity route coverage.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
      {
        id: "view.observability",
        kind: "view",
        route: "/observability",
        title: {
          executive: "Observability map",
          developer: "Observability route",
          maintainer: "Observability route coverage",
        },
        summary: {
          executive: "Route coverage placeholder for tests.",
          developer: "Ensures required route checks pass in focused tests.",
          maintainer: "Minimal view to satisfy integrity route coverage.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
      {
        id: "view.parity",
        kind: "view",
        route: "/parity",
        title: {
          executive: "Parity map",
          developer: "Parity route",
          maintainer: "Parity route coverage",
        },
        summary: {
          executive: "Route coverage placeholder for tests.",
          developer: "Ensures required route checks pass in focused tests.",
          maintainer: "Minimal view to satisfy integrity route coverage.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
      {
        id: "view.atlas",
        kind: "view",
        route: "/atlas",
        title: {
          executive: "Atlas map",
          developer: "Atlas route",
          maintainer: "Atlas route coverage",
        },
        summary: {
          executive: "Route coverage placeholder for tests.",
          developer: "Ensures required route checks pass in focused tests.",
          maintainer: "Minimal view to satisfy integrity route coverage.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
      {
        id: "view.ownership",
        kind: "view",
        route: "/",
        title: {
          executive: "Ownership map",
          developer: "System ownership",
          maintainer: "Product ownership boundary",
        },
        summary: {
          executive: "Shows who owns each product decision.",
          developer: "Connects authoring, execution, and presentation.",
          maintainer: "Pins ownership claims to implementation evidence.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
    ],
    crates: [],
    pairSupport: [],
    graphNodes: [
      {
        id: "node.root",
        tier: 0,
        parentId: null,
        childIds: ["node.runner"],
        owner: "rust",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http"],
        title: {
          executive: "Rust architecture root",
          developer: "Canonical graph root",
          maintainer: "Root graph entity for validation",
        },
        summary: {
          executive: "Captures the architecture entry point.",
          developer: "Provides one graph node for integrity tests.",
          maintainer: "Used as the parent of node.runner in focused tests.",
        },
        evidence: [
          {
            path: "AGENTS.md",
            lines: { start: 1, end: 12 },
            role: "source",
          },
        ],
        seamPorts: [{ id: "port.root.out", name: "Dispatch", channel: "control" }],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          autoExpandDepth: { executive: 0, developer: 1, maintainer: 3 },
        },
      },
      {
        id: "node.runner",
        tier: 1,
        parentId: "node.root",
        childIds: [],
        owner: "rust",
        status: { state: "planned", delivery: "runner_pair" },
        flavors: ["dynamo_online"],
        title: {
          executive: "Runner pair",
          developer: "Dynamo online runner pair",
          maintainer: "Runner backend/pair planned status anchor",
        },
        summary: {
          executive: "Planned runner pair remains clearly unbuilt.",
          developer: "Design-backed planned node for integrity tests.",
          maintainer: "Ensures planned entities include explicit design evidence.",
        },
        evidence: [
          {
            path: "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
            role: "design",
          },
        ],
        seamPorts: [{ id: "port.runner.in", name: "Input", channel: "control" }],
        audience: {
          visibility: ["developer", "maintainer"],
          autoExpandDepth: { executive: 0, developer: 2, maintainer: 3 },
        },
      },
    ],
    graphEdges: [
      {
        id: "edge.root.runner",
        source: { nodeId: "node.root", portId: "port.root.out" },
        target: { nodeId: "node.runner", portId: "port.runner.in" },
        channel: "control",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http"],
        protocol: "strict JSONL validate/execute",
        evidence: [
          {
            path: "AGENTS.md",
            lines: { start: 1, end: 12 },
            role: "source",
          },
        ],
      },
    ],
    graphScenes: [
      {
        id: "scene.runtime",
        title: "Runtime composition",
        rustScene: true,
        nodeIds: ["node.root", "node.runner"],
        edgeIds: ["edge.root.runner"],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          defaultDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
    ],
  });
}

describe("graph-first catalog", () => {
  it("supports required tiers, channels, and execution flavors", () => {
    expect([0, 1, 2, 3].every((tier) => tierSchema.safeParse(tier).success)).toBe(
      true,
    );
    expect(tierSchema.safeParse(4).success).toBe(false);
    expect(flowChannelSchema.options).toEqual([
      "control",
      "request_data",
      "token",
      "telemetry",
      "report_result",
    ]);
    expect(executionFlavorSchema.options).toEqual([
      "native_http",
      "native_grpc",
      "online_mock",
      "dynamo_offline",
      "dynamo_online",
    ]);
  });

  it("publishes all nine approved Rust scenes", () => {
    const rustScenes = architectureCatalog.graphScenes.filter(
      (scene) => scene.rustScene,
    );
    expect(rustScenes.map((scene) => scene.title)).toEqual([
      "Runtime composition",
      "Runner protocol and registries",
      "Scheduling and phase lifecycle",
      "Dataset and segment pipeline",
      "Endpoint bindings and HTTP/gRPC transports",
      "Graph-IR execution",
      "Metrics and telemetry",
      "Accuracy and evaluator hosting",
      "Crate dependency topology",
    ]);
  });

  it("includes the complete tier-0 Python-to-result journey", () => {
    expect(
      architectureCatalog.graphNodes
        .filter((node) => node.tier === 0)
        .map((node) => node.id),
    ).toEqual([
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
    ]);
  });

  it("keeps runtime composition as the complete canonical default scene", () => {
    const scene = architectureCatalog.graphScenes.find(
      ({ id }) => id === "scene.runtime-composition",
    );

    expect(scene).toBeDefined();
    expect(scene?.nodeIds).toEqual(
      expect.arrayContaining([
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
        "node.endpoint-bindings-transports",
        "node.metrics-telemetry",
        "node.dynamo-online-library-seam",
        "node.dynamo-offline-runner-backend",
        "node.dynamo-online-replay-mode",
        "node.dynamo-offline-sim-clock",
        "node.dynamo-offline-steppable-replay",
        "node.dynamo-offline-report-gate",
      ]),
    );
    expect(scene?.edgeIds).toEqual(
      expect.arrayContaining([
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
      ]),
    );
  });

  it("models Dynamo online as planned runner integration", () => {
    const librarySeam = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-library-seam",
    );
    const onlineReplay = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-replay-mode",
    );
    const runnerPair = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-runner-pair",
    );
    expect(librarySeam?.status.state).toBe("built");
    expect(onlineReplay?.status).toEqual({
      state: "built",
      delivery: "feature_gated",
    });
    expect(onlineReplay?.flavors).toContain("dynamo_online");
    expect(runnerPair?.status.state).toBe("planned");
    expect(runnerPair?.flavors).toContain("dynamo_online");
  });

  it("maps built online replay through the dynamo_offline runner backend", () => {
    const backend = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-offline-runner-backend",
    );
    const replay = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-replay-mode",
    );
    const edge = architectureCatalog.graphEdges.find(
      (candidate) => candidate.id === "edge.dynamo.online.replay-mode",
    );

    expect(backend?.flavors).toEqual(
      expect.arrayContaining(["dynamo_offline", "dynamo_online"]),
    );
    expect(backend?.childIds).toContain("node.dynamo-online-replay-mode");
    expect(replay?.parentId).toBe("node.dynamo-offline-runner-backend");
    expect(edge?.status).toEqual({
      state: "built",
      delivery: "feature_gated",
    });
    expect(edge?.flavors).toEqual(["dynamo_online"]);
    expect(
      replay?.evidence.some(
        ({ path, lines, role }) =>
          path === "crates/runner/src/offline_execution.rs" &&
          role === "source" &&
          lines !== undefined,
      ),
    ).toBe(true);
  });

  it("includes built Dynamo online replay across the shared Tier-0 journey", () => {
    const builtJourneyNodes = architectureCatalog.graphNodes.filter(
      (node) => node.tier === 0 && node.status.state === "built",
    );
    const builtJourneyEdges = architectureCatalog.graphEdges.filter(
      (edge) => edge.id.startsWith("edge.journey.") && edge.status.state === "built",
    );

    expect(
      builtJourneyNodes.every((node) => node.flavors.includes("dynamo_online")),
    ).toBe(true);
    expect(
      builtJourneyEdges.every((edge) => edge.flavors.includes("dynamo_online")),
    ).toBe(true);
  });

  it("distinguishes the invoked library helper from the planned dedicated pair", () => {
    const librarySeam = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-library-seam",
    );
    const dedicatedPair = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-runner-pair",
    );
    const dedicatedPairEdge = architectureCatalog.graphEdges.find(
      (edge) => edge.id === "edge.dynamo.online.runner.plan",
    );

    expect(librarySeam?.summary.developer).toMatch(
      /invoked.*dynamo_offline.*runner pair/i,
    );
    expect(librarySeam?.summary.maintainer).toMatch(
      /existing.*feature-gated.*pair/i,
    );
    expect(dedicatedPair?.status).toEqual({
      state: "planned",
      delivery: "runner_pair",
    });
    expect(dedicatedPairEdge?.status).toEqual({
      state: "planned",
      delivery: "runner_pair",
    });
  });

  it("models the complete feature-gated Dynamo offline path", () => {
    const requiredNodeIds = [
      "node.dynamo-offline-runner-backend",
      "node.dynamo-offline-sim-clock",
      "node.dynamo-offline-steppable-replay",
      "node.dynamo-offline-report-gate",
    ];
    const nodes = requiredNodeIds.map((id) =>
      architectureCatalog.graphNodes.find((node) => node.id === id),
    );

    expect(nodes.every(Boolean)).toBe(true);
    expect(
      nodes.every(
        (node) =>
          node?.status.state === "built" &&
          node.status.delivery === "feature_gated" &&
          node.flavors.includes("dynamo_offline"),
      ),
    ).toBe(true);
    expect(
      architectureCatalog.graphEdges.filter((edge) =>
        edge.id.startsWith("edge.dynamo.offline."),
      ),
    ).toHaveLength(3);
  });

  it("anchors DynoSim graph evidence to current source symbols", () => {
    const dynosimEvidence = [
      ...architectureCatalog.graphNodes.flatMap(({ evidence }) => evidence),
      ...architectureCatalog.graphEdges.flatMap(({ evidence }) => evidence),
    ].filter(({ path }) => path === "crates/aiperf/src/dynosim.rs");

    expect(dynosimEvidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          lines: { start: 4159, end: 4192 },
          symbol: "run_scheduled_backend_online",
        }),
        expect.objectContaining({
          lines: { start: 575, end: 649 },
          symbol: "OfflineEngineConfig::build_native",
        }),
        expect.objectContaining({
          lines: { start: 950, end: 1019 },
          symbol: "finish_shared_metrics_enforcing",
        }),
      ]),
    );
    expect(
      [
        ...architectureCatalog.components.flatMap(({ evidence }) => evidence),
        ...architectureCatalog.risks.flatMap(({ evidence }) => evidence),
        ...architectureCatalog.graphNodes.flatMap(({ evidence }) => evidence),
        ...architectureCatalog.graphEdges.flatMap(({ evidence }) => evidence),
      ].some(({ path }) => path === "crates/aiperf/src/dynamo_offline.rs"),
    ).toBe(false);
  });

  it("pins audited graph-catalog evidence ranges to current symbols", () => {
    const nodeById = new Map(
      architectureCatalog.graphNodes.map((node) => [node.id, node]),
    );

    expect(nodeById.get("node.runtime-composition")?.evidence).toContainEqual({
      path: "crates/runner/src/application.rs",
      lines: { start: 34, end: 65 },
      role: "source",
      symbol: "RunnerApplication",
    });
    expect(nodeById.get("node.dataset-segment-pipeline")?.evidence).toContainEqual({
      path: "crates/dataset/src/segment.rs",
      lines: { start: 195, end: 205 },
      role: "source",
      symbol: "SegmentStore",
    });
    expect(nodeById.get("node.request-sink-seam")?.evidence).toContainEqual({
      path: "crates/loadgen-core/src/sink.rs",
      lines: { start: 157, end: 160 },
      role: "source",
      symbol: "RequestSink",
    });
    expect(nodeById.get("node.crate-dependency-topology")?.evidence).toContainEqual({
      path: "apps/architecture-atlas/src/domain/integrity.ts",
      lines: { start: 404, end: 455 },
      role: "source",
      symbol: "validateWorkspaceCrates",
    });

    expect(
      nodeById.get("node.dynamo-online-library-seam")?.evidence,
    ).toContainEqual({
      path: "crates/aiperf/src/dynosim.rs",
      lines: { start: 4159, end: 4192 },
      role: "source",
      symbol: "run_scheduled_backend_online",
    });
    expect(
      nodeById.get("node.dynamo-offline-steppable-replay")?.evidence,
    ).toContainEqual({
      path: "crates/aiperf/src/dynosim.rs",
      lines: { start: 575, end: 649 },
      role: "source",
      symbol: "OfflineEngineConfig::build_native",
    });
    expect(nodeById.get("node.dynamo-offline-report-gate")?.evidence).toContainEqual({
      path: "crates/aiperf/src/dynosim.rs",
      lines: { start: 950, end: 1019 },
      role: "source",
      symbol: "finish_shared_metrics_enforcing",
    });
    expect(
      nodeById.get("node.dynamo-offline-runner-backend")?.evidence,
    ).toEqual(
      expect.arrayContaining([
        {
          path: "crates/runner/src/offline_execution.rs",
          lines: { start: 98, end: 103 },
          role: "source",
          symbol: "DYNOSIM_BACKEND_ID",
        },
        {
          path: "crates/runner/src/offline_execution.rs",
          lines: { start: 830, end: 846 },
          role: "source",
          symbol: "DynosimBackendFactory",
        },
      ]),
    );
    expect(nodeById.get("node.dynamo-online-replay-mode")?.evidence).toEqual(
      expect.arrayContaining([
        {
          path: "crates/runner/src/offline_execution.rs",
          lines: { start: 229, end: 249 },
          role: "source",
          symbol: "DynamoReplayModeSpec",
        },
        {
          path: "crates/runner/src/offline_execution.rs",
          lines: { start: 1894, end: 1923 },
          role: "source",
          symbol: "DynosimExecutor::execute_scheduled",
        },
      ]),
    );
  });

  it("models RequestObserver on_token as a source-grounded token edge", () => {
    const edge = architectureCatalog.graphEdges.find(
      ({ id }) => id === "edge.request-sink.token.metrics",
    );
    const scene = architectureCatalog.graphScenes.find(
      ({ id }) => id === "scene.runtime-composition",
    );

    expect(edge).toMatchObject({
      source: {
        nodeId: "node.request-sink-seam",
        portId: "port.sink.token",
      },
      target: {
        nodeId: "node.metrics-telemetry",
        portId: "port.metrics.token",
      },
      channel: "token",
      protocol: "RequestObserver::on_token",
    });
    expect(edge?.evidence).toContainEqual({
      path: "crates/loadgen-core/src/sink.rs",
      lines: { start: 81, end: 104 },
      role: "source",
      symbol: "RequestObserver::on_token",
    });
    expect(scene?.nodeIds).toEqual(
      expect.arrayContaining([
        "node.request-sink-seam",
        "node.metrics-telemetry",
      ]),
    );
    expect(scene?.edgeIds).toContain("edge.request-sink.token.metrics");
  });

  it("requires reciprocal parent and child declarations", async () => {
    const catalog = minimalGraphCatalog();
    catalog.graphNodes[0].childIds = [];

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/parent.*child|child.*parent/i);
  });

  it("populates Tier-0 children for every hierarchical descendant", () => {
    const nodesById = new Map(
      architectureCatalog.graphNodes.map((node) => [node.id, node]),
    );

    for (const node of architectureCatalog.graphNodes) {
      if (node.parentId) {
        expect(nodesById.get(node.parentId)?.childIds).toContain(node.id);
      }
    }
  });

  it("rejects dangling scene references", async () => {
    const catalog = minimalGraphCatalog();
    catalog.graphScenes[0].nodeIds.push("node.missing");
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/scene.*node\.missing/i);
  });

  it("rejects scene edges whose endpoints are outside the scene", async () => {
    const catalog = minimalGraphCatalog();
    catalog.graphScenes[0].nodeIds = ["node.root"];

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/scene.*edge.*endpoint/i);
  });

  it("keeps every canonical scene closed over edge endpoints", () => {
    const edges = new Map(
      architectureCatalog.graphEdges.map((edge) => [edge.id, edge]),
    );

    for (const scene of architectureCatalog.graphScenes) {
      const nodeIds = new Set(scene.nodeIds);
      for (const edgeId of scene.edgeIds) {
        const edge = edges.get(edgeId);
        expect(edge).toBeDefined();
        expect(nodeIds.has(edge!.source.nodeId)).toBe(true);
        expect(nodeIds.has(edge!.target.nodeId)).toBe(true);
      }
    }
  });

  it("rejects dangling port endpoints", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphEdges[0] as GraphEdge).source.portId = "port.root.missing";
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/missing port/i);
  });

  it("rejects built entities backed only by design evidence", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphNodes[0] as GraphNode).evidence = [
      {
        path: "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
        role: "design",
      },
    ];
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/built.*design evidence/i);
  });

  it.each(["node", "edge"] as const)(
    "rejects %s source evidence without line ranges",
    async (entityKind) => {
      const catalog = minimalGraphCatalog();
      const entity =
        entityKind === "node" ? catalog.graphNodes[0] : catalog.graphEdges[0];
      entity.evidence = [{ path: "AGENTS.md", role: "source" }];

      await expect(
        validateArchitectureCatalog(catalog, repositoryRoot),
      ).rejects.toThrow(/source evidence.*line range/i);
    },
  );

  it.each(["node", "edge"] as const)(
    "rejects %s spec evidence without an explicit design role",
    async (entityKind) => {
      const catalog = minimalGraphCatalog();
      const entity =
        entityKind === "node" ? catalog.graphNodes[0] : catalog.graphEdges[0];
      entity.evidence = [
        {
          path:
            "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
          lines: { start: 1, end: 12 },
        },
      ];

      await expect(
        validateArchitectureCatalog(catalog, repositoryRoot),
      ).rejects.toThrow(/design evidence.*role/i);
    },
  );

  it.each(["node", "edge"] as const)(
    "rejects %s spec evidence declared as source",
    async (entityKind) => {
      const catalog = minimalGraphCatalog();
      const entity =
        entityKind === "node" ? catalog.graphNodes[0] : catalog.graphEdges[0];
      entity.evidence = [
        {
          path:
            "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
          lines: { start: 1, end: 12 },
          role: "source",
        },
      ];

      await expect(
        validateArchitectureCatalog(catalog, repositoryRoot),
      ).rejects.toThrow(/design.*cannot.*source/i);
    },
  );

  it("rejects planned entities without design evidence", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphNodes[1] as GraphNode).evidence = [
      {
        path: "AGENTS.md",
        lines: { start: 1, end: 12 },
        role: "source",
      },
    ];
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/planned.*design evidence/i);
  });

  it.each(["node", "edge"] as const)(
    "rejects dedicated Dynamo-online runner-pair %s facts marked built",
    async (entityKind) => {
      const catalog = minimalGraphCatalog();
      const sourceEvidence = [
        {
          path: "AGENTS.md",
          lines: { start: 1, end: 12 },
          role: "source" as const,
        },
      ];
      if (entityKind === "node") {
        (catalog.graphNodes[1] as GraphNode).status = {
          state: "built",
          delivery: "runner_pair",
        };
        (catalog.graphNodes[1] as GraphNode).evidence = sourceEvidence;
      } else {
        catalog.graphEdges.push({
          id: "edge.dynamo.online.dedicated-pair",
          source: { nodeId: "node.root", portId: "port.root.out" },
          target: { nodeId: "node.runner", portId: "port.runner.in" },
          channel: "control",
          status: { state: "built", delivery: "runner_pair" },
          flavors: ["dynamo_online"],
          protocol: "Dedicated Dynamo-online runner pair",
          evidence: sourceEvidence,
          footnotes: [],
        });
      }

      await expect(
        validateArchitectureCatalog(catalog, repositoryRoot),
      ).rejects.toThrow(/planned.*dynamo[_ ]online.*runner/i);
    },
  );

  it("rejects invalid parent-child relationships and cycles", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphNodes[0] as GraphNode).parentId = "node.runner";
    (catalog.graphNodes[1] as GraphNode).childIds.push("node.root");
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/cycle|parent/i);
  });

  it("rejects edge channels that differ from endpoint ports", async () => {
    const catalog = minimalGraphCatalog();
    catalog.graphEdges[0].channel = "telemetry";

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/channel.*port/i);
  });
});
