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
        evidence: [{ path: "AGENTS.md", role: "source" }],
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
        evidence: [{ path: "AGENTS.md", role: "source" }],
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

  it("models Dynamo online as planned runner integration", () => {
    const librarySeam = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-library-seam",
    );
    const runnerPair = architectureCatalog.graphNodes.find(
      (node) => node.id === "node.dynamo-online-runner-pair",
    );
    expect(librarySeam?.status.state).toBe("built");
    expect(runnerPair?.status.state).toBe("planned");
    expect(runnerPair?.flavors).toContain("dynamo_online");
  });

  it("rejects dangling scene references", async () => {
    const catalog = minimalGraphCatalog();
    catalog.graphScenes[0].nodeIds.push("node.missing");
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/scene.*node\.missing/i);
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

  it("rejects planned entities without design evidence", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphNodes[1] as GraphNode).evidence = [{ path: "AGENTS.md", role: "source" }];
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/planned.*design evidence/i);
  });

  it("rejects Dynamo-online runner entities marked built", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphNodes[1] as GraphNode).status = {
      state: "built",
      delivery: "runner_pair",
    };
    (catalog.graphNodes[1] as GraphNode).evidence = [{ path: "AGENTS.md", role: "source" }];
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/planned.*dynamo[_ ]online.*runner/i);
  });

  it("rejects invalid parent-child relationships and cycles", async () => {
    const catalog = minimalGraphCatalog();
    (catalog.graphNodes[0] as GraphNode).parentId = "node.runner";
    (catalog.graphNodes[1] as GraphNode).childIds.push("node.root");
    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/cycle|parent/i);
  });
});
