/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused contracts for canonical connector resolution.

import { describe, expect, it } from "vitest";

import type {
  SceneGeometryLike,
  SceneNodeLike,
  ScenePointLike,
} from "../scene-types.js";
import {
  isRoutingObstacle,
  resolveConnectors,
  svgPathEndpoints,
} from "./resolve-connectors.js";

const PANEL_BOUNDS: Readonly<Record<string, SceneGeometryLike>> = {
  a: { x: 0, y: 20, width: 80, height: 60 },
  b: { x: 220, y: 20, width: 80, height: 60 },
};

function panel(id: string, geometry: SceneGeometryLike): SceneNodeLike {
  return {
    id,
    kind: "group",
    capabilityId: "core.panel",
    geometry,
    children: [],
  };
}

function edge(
  props: Readonly<{
    id?: string;
    from?: ScenePointLike;
    to?: ScenePointLike;
    d?: string;
    path?: string;
    points?: readonly ScenePointLike[];
    arrowhead?: boolean;
    route?: string;
  }> = {},
): SceneNodeLike {
  return {
    id: props.id ?? "edge",
    kind: "connector",
    capabilityId: "core.connector",
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    from: props.from ?? { nodeId: "a", anchor: "e" },
    to: props.to ?? { nodeId: "b", anchor: "w" },
    ...(props.d !== undefined ? { d: props.d } : {}),
    ...(props.path !== undefined ? { path: props.path } : {}),
    ...(props.points !== undefined ? { points: props.points } : {}),
    style: {
      ...(props.arrowhead === false
        ? { markerEnd: "none", arrowhead: false }
        : {}),
      ...(props.route !== undefined ? { route: props.route } : {}),
    },
  };
}

function resolve(
  connector: SceneNodeLike,
  extraNodes: readonly SceneNodeLike[] = [],
  options: Readonly<{ generatedPartIds?: ReadonlySet<string> }> = {},
) {
  const nodes = [
    panel("a", PANEL_BOUNDS.a!),
    panel("b", PANEL_BOUNDS.b!),
    ...extraNodes,
    connector,
  ];
  return resolveConnectors({
    nodesById: new Map(nodes.map((node) => [node.id, node])),
    worldGeometryById: new Map(
      nodes.map((node) => [
        node.id,
        node.geometry ?? { x: 0, y: 0, width: 0, height: 0 },
      ]),
    ),
    ancestorIdsById: new Map(nodes.map((node) => [node.id, []])),
    ...(options.generatedPartIds === undefined
      ? {}
      : { generatedPartIds: options.generatedPartIds }),
  });
}

function decorativeNode(
  id: string,
  capabilityId: string,
  geometry: SceneGeometryLike,
): SceneNodeLike {
  return {
    id,
    kind: "group",
    capabilityId,
    geometry,
    children: [],
  };
}

describe("svgPathEndpoints", () => {
  it("tracks absolute and relative SVG endpoint semantics without rewriting", () => {
    expect(
      svgPathEndpoints(
        "M10 20 l5 0 h10 v10 c1 2 3 4 5 6 s7 8 9 10 q1 2 3 4 t5 6 a8 9 0 0 1 10 12",
      ),
    ).toEqual({
      start: { x: 10, y: 20 },
      end: { x: 57, y: 68 },
    });
    expect(svgPathEndpoints("M0 0 L10 nope")).toBeUndefined();
    expect(svgPathEndpoints("M0 0 LInfinity 1")).toBeUndefined();
  });
});

describe("resolveConnectors", () => {
  it("defaults edges to visible direction and honors explicit opt-out", () => {
    expect(resolve(edge()).connectorsById.get("edge")).toMatchObject({
      directed: true,
      showArrowhead: true,
      source: { x: 80, y: 50 },
      target: { x: 220, y: 50 },
    });
    expect(resolve(edge({ arrowhead: false })).connectorsById.get("edge"))
      .toMatchObject({
        directed: false,
        showArrowhead: false,
      });
  });

  it("preserves authored path precedence and reports reversed direction", () => {
    const authored = "M220 50 C180 10 120 90 80 50";
    const result = resolve(
      edge({
        d: authored,
        path: "M80 50 L220 50",
        points: [
          { x: 80, y: 50 },
          { x: 220, y: 50 },
        ],
      }),
    );

    expect(result.connectorsById.get("edge")?.d).toBe(authored);
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_AUTHORED_PATH_REVERSED",
        nodeIds: ["edge"],
      }),
    );
  });

  it("reports detached authored endpoints without changing the path", () => {
    const authored = "M100 100 L150 100";
    const result = resolve(edge({ path: authored }));

    expect(result.connectorsById.get("edge")?.d).toBe(authored);
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_CONNECTOR_ENDPOINT_DETACHED",
        severity: "error",
      }),
    );
  });

  it("skips obstacle-noise diagnostics for decorative dividers but keeps them for content connectors", () => {
    const blocker = panel("blocker", {
      x: 130,
      y: 35,
      width: 40,
      height: 30,
    });
    const props = {
      from: { x: 80, y: 50 },
      to: { x: 220, y: 50 },
      path: "M80 50 L220 50",
    };
    const divider: SceneNodeLike = {
      ...edge({ id: "divider", ...props }),
      capabilityId: "core.divider",
    };

    const dividerResult = resolve(divider, [blocker]);
    const contentResult = resolve(edge({ id: "content-edge", ...props }), [blocker]);
    const obstacleDiagnosticCodes = new Set([
      "SCENE_CONNECTOR_INTERSECTION",
      "SCENE_CONNECTOR_VISUALLY_AMBIGUOUS",
    ]);

    expect(
      dividerResult.diagnostics.filter(({ code }) => obstacleDiagnosticCodes.has(code)),
    ).toEqual([]);
    expect(
      contentResult.diagnostics.map(({ code }) => code),
    ).toEqual(
      expect.arrayContaining([
        "SCENE_CONNECTOR_INTERSECTION",
        "SCENE_CONNECTOR_VISUALLY_AMBIGUOUS",
      ]),
    );
  });

  it("reports missing endpoint geometry instead of silently collapsing to the origin", () => {
    const result = resolve(edge({ from: { nodeId: "missing", anchor: "e" } }));
    const resolved = result.connectorsById.get("edge");

    expect(resolved?.source).toEqual({ x: 0, y: 0 });
    expect(resolved?.sourceId).toBe("missing");
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_CONNECTOR_ENDPOINT_MISSING_GEOMETRY",
        severity: "error",
        nodeIds: ["edge", "missing"],
      }),
    );
  });

  it("still allows explicit x/y coordinates without requiring a nodeId", () => {
    const result = resolve(
      edge({ from: { x: 12, y: 34 }, to: { x: 56, y: 78 } }),
    );
    const resolved = result.connectorsById.get("edge");

    expect(resolved?.source).toEqual({ x: 12, y: 34 });
    expect(resolved?.target).toEqual({ x: 56, y: 78 });
    expect(resolved?.sourceId).toBeUndefined();
    expect(resolved?.targetId).toBeUndefined();
    expect(
      result.diagnostics.some(
        (diagnostic) =>
          diagnostic.code === "SCENE_CONNECTOR_ENDPOINT_MISSING_GEOMETRY",
      ),
    ).toBe(false);
  });

  it("resolves edge-bound motion signals from the referenced connector path", () => {
    const credit = edge({ id: "request-credit" });
    const motion: SceneNodeLike = {
      id: "motion",
      kind: "connector",
      capabilityId: "motion.signal",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      edgeRef: "request-credit",
      style: {},
    };
    const result = resolve(credit, [motion]);
    const edgePath = result.connectorsById.get("request-credit");
    const motionPath = result.connectorsById.get("motion");

    expect(edgePath?.d).toBeDefined();
    expect(motionPath?.d).toBe(edgePath?.d);
    expect(motionPath).toMatchObject({
      directed: false,
      showArrowhead: false,
      source: edgePath?.source,
      target: edgePath?.target,
    });
    expect(motionPath?.d).not.toBe("M0 0 L0 0");
  });

  it("reports missing edge references for edge-bound motion signals", () => {
    const motion: SceneNodeLike = {
      id: "motion",
      kind: "connector",
      capabilityId: "motion.signal",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      edgeRef: "missing-edge",
      style: {},
    };
    const result = resolve(edge(), [motion]);

    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_SIGNAL_EDGE_NOT_FOUND",
        severity: "error",
        nodeIds: expect.arrayContaining(["motion", "missing-edge"]),
      }),
    );
  });

  it("reports standalone motion signals that duplicate an existing connector", () => {
    const authoredPath = "M80 50 L220 50";
    const credit = edge({
      id: "request-credit",
      d: authoredPath,
      from: { nodeId: "a", anchor: "e" },
      to: { nodeId: "b", anchor: "w" },
    });
    const duplicate: SceneNodeLike = {
      id: "motion",
      kind: "connector",
      capabilityId: "motion.signal",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      d: authoredPath,
      from: { nodeId: "a", anchor: "e" },
      to: { nodeId: "b", anchor: "w" },
      style: {},
    };
    const result = resolve(credit, [duplicate]);

    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_SIGNAL_DUPLICATES_EDGE",
        severity: "error",
        nodeIds: ["motion", "request-credit"],
        repair: 'Reference the existing edge with edge = "request-credit".',
      }),
    );
  });

  it("carries penetrating curve fallback metadata into diagnostics", () => {
    const blocker = panel("blocker", {
      x: 75,
      y: -100,
      width: 150,
      height: 300,
    });
    const result = resolve(edge({ route: "curve" }), [blocker]);
    const resolved = result.connectorsById.get("edge");

    expect(resolved?.usedFallback).toBe(true);
    expect(resolved?.penetratedObstacleIds).toContain("blocker");
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_ROUTE_FALLBACK",
        severity: "warning",
        nodeIds: expect.arrayContaining(["edge", "blocker"]),
      }),
    );
  });

  it("excludes decorative bands, brackets, and thin labels from route obstacles", () => {
    const band = decorativeNode("map-band", "core.band", {
      x: 0,
      y: 0,
      width: 300,
      height: 200,
    });
    const bracket = decorativeNode("dispatch-bracket", "core.bracket", {
      x: 40,
      y: 10,
      width: 20,
      height: 80,
    });
    const label = decorativeNode("rx2-ew-label", "core.text", {
      x: 100,
      y: 40,
      width: 200,
      height: 20,
    });
    const result = resolve(edge({ route: "curve" }), [band, bracket, label]);

    expect(
      result.diagnostics.some(({ code }) => code === "SCENE_ROUTE_FALLBACK"),
    ).toBe(false);
    expect(result.connectorsById.get("edge")?.penetratedObstacleIds).toEqual([]);
  });

  it("still treats content panels as routing obstacles", () => {
    const blocker = panel("blocker", {
      x: 75,
      y: -100,
      width: 150,
      height: 300,
    });
    const result = resolve(edge({ route: "curve" }), [blocker]);

    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_ROUTE_FALLBACK",
        nodeIds: expect.arrayContaining(["edge", "blocker"]),
      }),
    );
  });

  it("excludes generated semantic parts from route obstacles", () => {
    const generated = panel("note__caption", {
      x: 130,
      y: 10,
      width: 40,
      height: 80,
    });
    const result = resolve(edge({ route: "curve" }), [generated], {
      generatedPartIds: new Set(["note__caption"]),
    });

    expect(
      result.diagnostics.some(({ code }) => code === "SCENE_ROUTE_FALLBACK"),
    ).toBe(false);
  });

  it("does not emit route fallback warnings when search is infeasible but the path does not penetrate", () => {
    const crowding = [
      panel("row-1", { x: 110, y: 30, width: 30, height: 40 }),
      panel("row-2", { x: 110, y: 80, width: 30, height: 40 }),
      panel("row-3", { x: 110, y: 130, width: 30, height: 40 }),
    ];
    const result = resolve(
      {
        ...edge({
          route: "curve",
          from: { nodeId: "a", anchor: "e" },
          to: { nodeId: "b", anchor: "w" },
        }),
        style: { route: "curve", clearance: 24 },
      },
      crowding,
    );
    const resolved = result.connectorsById.get("edge");

    if (resolved?.usedFallback === true && resolved.penetratedObstacleIds.length === 0) {
      expect(
        result.diagnostics.some(({ code }) => code === "SCENE_ROUTE_FALLBACK"),
      ).toBe(false);
    }
  });
});

describe("isRoutingObstacle", () => {
  it("matches overlap-validation decorative policy for bands and annotation text", () => {
    const band = decorativeNode("band", "core.band", {
      x: 0,
      y: 0,
      width: 100,
      height: 100,
    });
    const label = decorativeNode("label", "core.text", {
      x: 0,
      y: 0,
      width: 100,
      height: 20,
    });
    const panelNode = panel("panel", { x: 0, y: 0, width: 100, height: 72 });

    expect(isRoutingObstacle(band, band.geometry!)).toBe(false);
    expect(isRoutingObstacle(label, label.geometry!)).toBe(false);
    expect(isRoutingObstacle(panelNode, panelNode.geometry!)).toBe(true);
    expect(
      isRoutingObstacle(panelNode, panelNode.geometry!, new Set(["panel"])),
    ).toBe(false);
  });
});
