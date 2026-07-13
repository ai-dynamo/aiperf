// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { architectureCatalog } from "../content";
import {
  architectureCatalogSchema,
  type ArchitectureCatalog,
} from "./architecture";
import {
  collapseExpandedNode,
  deriveGraphDerivation,
  selectSceneById,
  toggleExpandedNode,
} from "./graph-derivation";

function createGraphCatalog(): ArchitectureCatalog {
  return architectureCatalogSchema.parse({
    schemaVersion: 2,
    components: [
      {
        id: "component.placeholder",
        kind: "component",
        owner: "rust",
        lifecycleBand: "execution",
        status: "built",
        title: {
          executive: "Placeholder component",
          developer: "Placeholder component",
          maintainer: "Placeholder component",
        },
        summary: {
          executive: "Ensures schema completeness in focused tests.",
          developer: "Ensures schema completeness in focused tests.",
          maintainer: "Ensures schema completeness in focused tests.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
        modes: ["online_http"],
        contracts: ["placeholder-contract"],
      },
    ],
    edges: [],
    risks: [],
    lifecycleStages: [],
    views: [
      {
        id: "view.atlas",
        kind: "view",
        route: "/atlas",
        title: {
          executive: "Atlas View",
          developer: "Atlas View",
          maintainer: "Atlas View",
        },
        summary: {
          executive: "Atlas view for test graph derivation.",
          developer: "Atlas view for test graph derivation.",
          maintainer: "Atlas view for test graph derivation.",
        },
        componentIds: ["component.placeholder"],
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
        childIds: ["node.shared", "node.http-only", "node.offline-only"],
        owner: "rust",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http", "dynamo_offline"],
        title: {
          executive: "Root Node",
          developer: "Root Node",
          maintainer: "Root Node",
        },
        summary: {
          executive: "Root summary with no transport lock-in.",
          developer: "Root summary with no transport lock-in.",
          maintainer: "Root summary with no transport lock-in.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
        seamPorts: [{ id: "port.root.out", name: "Root out", channel: "control" }],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
      {
        id: "node.shared",
        tier: 1,
        parentId: "node.root",
        childIds: ["node.shared-deep"],
        owner: "rust",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http", "dynamo_offline"],
        title: {
          executive: "Shared Node",
          developer: "Shared Node",
          maintainer: "Shared Node",
        },
        summary: {
          executive: "Shared branch.",
          developer: "Shared branch.",
          maintainer: "Shared branch.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
        seamPorts: [
          { id: "port.shared.in", name: "Shared in", channel: "control" },
          { id: "port.shared.out", name: "Shared out", channel: "request_data" },
        ],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
      {
        id: "node.shared-deep",
        tier: 2,
        parentId: "node.shared",
        childIds: [],
        owner: "rust",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http", "dynamo_offline"],
        title: {
          executive: "Deep node",
          developer: "Deep node",
          maintainer: "Deep node",
        },
        summary: {
          executive: "Deep descendant for reveal tests.",
          developer: "Deep descendant for reveal tests.",
          maintainer: "Deep descendant for reveal tests.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
        seamPorts: [{ id: "port.deep.in", name: "Deep in", channel: "request_data" }],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
      {
        id: "node.http-only",
        tier: 1,
        parentId: "node.root",
        childIds: [],
        owner: "rust",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http"],
        title: {
          executive: "HTTP only",
          developer: "HTTP only",
          maintainer: "HTTP only",
        },
        summary: {
          executive: "Only appears in native_http.",
          developer: "Only appears in native_http.",
          maintainer: "Only appears in native_http.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
        seamPorts: [{ id: "port.http.in", name: "HTTP in", channel: "request_data" }],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
      {
        id: "node.offline-only",
        tier: 1,
        parentId: "node.root",
        childIds: [],
        owner: "rust",
        status: { state: "planned", delivery: "feature_gated" },
        flavors: ["dynamo_offline"],
        title: {
          executive: "Offline only",
          developer: "Offline only",
          maintainer: "Offline only",
        },
        summary: {
          executive: "Only appears in dynamo_offline.",
          developer: "Only appears in dynamo_offline.",
          maintainer: "Only appears in dynamo_offline.",
        },
        evidence: [
          {
            path: "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
            role: "design",
          },
        ],
        seamPorts: [
          { id: "port.offline.in", name: "Offline in", channel: "request_data" },
        ],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
    ],
    graphEdges: [
      {
        id: "edge.root.shared",
        source: { nodeId: "node.root", portId: "port.root.out" },
        target: { nodeId: "node.shared", portId: "port.shared.in" },
        channel: "control",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http", "dynamo_offline"],
        protocol: "shared",
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
      },
      {
        id: "edge.shared.deep",
        source: { nodeId: "node.shared", portId: "port.shared.out" },
        target: { nodeId: "node.shared-deep", portId: "port.deep.in" },
        channel: "request_data",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http", "dynamo_offline"],
        protocol: "shared",
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
      },
      {
        id: "edge.root.http",
        source: { nodeId: "node.root", portId: "port.root.out" },
        target: { nodeId: "node.http-only", portId: "port.http.in" },
        channel: "request_data",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["native_http"],
        protocol: "http",
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
      },
      {
        id: "edge.root.offline",
        source: { nodeId: "node.root", portId: "port.root.out" },
        target: { nodeId: "node.offline-only", portId: "port.offline.in" },
        channel: "request_data",
        status: { state: "planned", delivery: "feature_gated" },
        flavors: ["dynamo_offline"],
        protocol: "offline",
        evidence: [
          {
            path: "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
            role: "design",
          },
        ],
      },
      {
        id: "edge.pruned.missing-endpoint",
        source: { nodeId: "node.shared-deep", portId: "port.deep.in" },
        target: { nodeId: "node.offline-only", portId: "port.offline.in" },
        channel: "request_data",
        status: { state: "built", delivery: "unconditional" },
        flavors: ["dynamo_offline"],
        protocol: "should-prune-when-endpoints-hidden",
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 4 } }],
      },
    ],
    graphScenes: [
      {
        id: "scene.runtime",
        title: "Runtime scene",
        rustScene: true,
        nodeIds: [
          "node.root",
          "node.shared",
          "node.shared-deep",
          "node.http-only",
          "node.offline-only",
        ],
        edgeIds: [
          "edge.root.shared",
          "edge.shared.deep",
          "edge.root.http",
          "edge.root.offline",
          "edge.pruned.missing-endpoint",
        ],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          defaultDepth: { executive: 1, developer: 2, maintainer: 3 },
        },
      },
      {
        id: "scene.secondary",
        title: "Secondary scene",
        rustScene: true,
        nodeIds: ["node.root"],
        edgeIds: [],
        audience: {
          visibility: ["executive", "developer", "maintainer"],
          defaultDepth: { executive: 0, developer: 1, maintainer: 2 },
        },
      },
    ],
  });
}

describe("graph derivation", () => {
  const catalog = createGraphCatalog();

  it("selects one scene by stable id", () => {
    const scene = selectSceneById(catalog, "scene.secondary");
    expect(scene.id).toBe("scene.secondary");
    expect(() => selectSceneById(catalog, "scene.missing")).toThrow(
      /unknown scene/i,
    );
  });

  it("applies audience default topology depth", () => {
    const executive = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "executive",
      primaryFlavor: "native_http",
    });
    const developer = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "developer",
      primaryFlavor: "native_http",
    });

    expect(executive.visibleNodeIds).toEqual([
      "node.root",
      "node.shared",
      "node.http-only",
    ]);
    expect(developer.visibleNodeIds).toEqual([
      "node.root",
      "node.shared",
      "node.shared-deep",
      "node.http-only",
    ]);
  });

  it("supports expand/collapse with descendant cleanup and breadcrumbs", () => {
    const expanded = toggleExpandedNode([], "node.shared");
    expect(expanded).toEqual(["node.shared"]);

    const withDescendants = ["node.shared", "node.shared-deep"];
    const collapsed = collapseExpandedNode(catalog, withDescendants, "node.shared");
    expect(collapsed).toEqual([]);

    const derived = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "executive",
      primaryFlavor: "native_http",
      expandedNodeIds: expanded,
      focusedEntityId: "node.shared-deep",
    });
    expect(derived.visibleNodeIds).toContain("node.shared-deep");
    expect(derived.breadcrumbNodeIds).toEqual([
      "node.root",
      "node.shared",
      "node.shared-deep",
    ]);
  });

  it("reveals hidden descendants when search matches", () => {
    const derived = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "executive",
      primaryFlavor: "native_http",
      searchQuery: "deep descendant",
    });

    expect(derived.visibleNodeIds).toContain("node.shared-deep");
    expect(derived.revealedAncestorNodeIds).toEqual(["node.shared"]);
  });

  it("overlays primary and compare flavors with shared entities exactly once", () => {
    const derived = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "developer",
      primaryFlavor: "native_http",
      compareFlavor: "dynamo_offline",
    });

    expect(derived.visibleNodeIds).toEqual([
      "node.root",
      "node.shared",
      "node.shared-deep",
      "node.http-only",
      "node.offline-only",
    ]);
    expect(derived.overlay).toEqual({
      sharedNodeIds: ["node.root", "node.shared", "node.shared-deep"],
      primaryOnlyNodeIds: ["node.http-only"],
      compareOnlyNodeIds: ["node.offline-only"],
      sharedEdgeIds: ["edge.root.shared", "edge.shared.deep"],
      primaryOnlyEdgeIds: ["edge.root.http"],
      compareOnlyEdgeIds: ["edge.root.offline", "edge.pruned.missing-endpoint"],
    });
  });

  it("prunes edges when one endpoint is not visible", () => {
    const derived = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "executive",
      primaryFlavor: "native_http",
    });

    expect(derived.visibleEdgeIds).toEqual(["edge.root.shared", "edge.root.http"]);
    expect(derived.visibleEdgeIds).not.toContain("edge.shared.deep");
    expect(derived.visibleEdgeIds).not.toContain("edge.pruned.missing-endpoint");
  });

  it("derives directed upstream and downstream neighborhoods", () => {
    const derived = deriveGraphDerivation(catalog, {
      sceneId: "scene.runtime",
      audience: "developer",
      primaryFlavor: "native_http",
      focusedEntityId: "node.shared",
    });

    expect(derived.neighborhood).toEqual({
      upstreamNodeIds: ["node.root"],
      downstreamNodeIds: ["node.shared-deep"],
    });
  });

  it("derives canonical runtime topology with audience depth from the real catalog", () => {
    const executive = deriveGraphDerivation(architectureCatalog, {
      sceneId: "scene.runtime-composition",
      audience: "executive",
      primaryFlavor: "native_http",
    });
    const maintainer = deriveGraphDerivation(architectureCatalog, {
      sceneId: "scene.runtime-composition",
      audience: "maintainer",
      primaryFlavor: "native_http",
    });

    expect(executive.visibleNodeIds).toContain("node.runtime-composition");
    expect(maintainer.visibleNodeIds).toContain("node.request-sink-seam");
    expect(maintainer.visibleNodeIds.length).toBeGreaterThan(
      executive.visibleNodeIds.length,
    );
  });

  it("partitions canonical flavor overlay without duplicate node or edge classification", () => {
    const derived = deriveGraphDerivation(architectureCatalog, {
      sceneId: "scene.runtime-composition",
      audience: "developer",
      primaryFlavor: "native_http",
      compareFlavor: "dynamo_online",
    });

    const nodeOverlayIds = [
      ...derived.overlay.sharedNodeIds,
      ...derived.overlay.primaryOnlyNodeIds,
      ...derived.overlay.compareOnlyNodeIds,
    ];
    const edgeOverlayIds = [
      ...derived.overlay.sharedEdgeIds,
      ...derived.overlay.primaryOnlyEdgeIds,
      ...derived.overlay.compareOnlyEdgeIds,
    ];

    expect(new Set(nodeOverlayIds).size).toBe(nodeOverlayIds.length);
    expect(new Set(edgeOverlayIds).size).toBe(edgeOverlayIds.length);
    expect(new Set(nodeOverlayIds)).toEqual(new Set(derived.visibleNodeIds));
    expect(new Set(edgeOverlayIds)).toEqual(new Set(derived.visibleEdgeIds));
    expect(derived.overlay.compareOnlyNodeIds.length).toBeGreaterThan(0);
    expect(derived.overlay.primaryOnlyNodeIds.length).toBeGreaterThan(0);
  });
});
