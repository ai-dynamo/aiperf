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
  });
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
});
