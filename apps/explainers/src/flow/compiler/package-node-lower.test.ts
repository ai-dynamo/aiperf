/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { capabilityKind, lowerFirstClassPackageNode } from "./package-node-lower.js";

const compilerModules = import.meta.glob("./*.ts", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

describe("package node lowering module", () => {
  it("contains only the live package-node lowering surface", () => {
    expect(compilerModules).not.toHaveProperty("./desugar-scene-primitives.ts");

    const source = compilerModules["./package-node-lower.ts"];
    expect(source).toBeDefined();
    expect(source).toContain("export function asRecord");
    expect(source).toContain("export function capabilityKind");
    expect(source).toContain("export function isSupportedPackageCapability");
    expect(source).toContain("export function lowerFirstClassPackageNode");
    expect(source).not.toContain("desugarPackageNode");
  });
});

function lowerConnector(node: Record<string, unknown>) {
  return lowerFirstClassPackageNode(node, {
    id: "edge",
    capability: "core.connector",
    kind: capabilityKind("core.connector"),
    children: [],
    label: "edge",
    fallback: "edge",
  });
}

describe("styleOf preserves theme-role style values", () => {
  it("keeps structured { kind: \"theme-role\", role } fill/stroke on rect nodes", () => {
    const lowered = lowerFirstClassPackageNode(
      {
        style: {
          fill: { kind: "theme-role", role: "surface.panel" },
          stroke: { kind: "theme-role", role: "line.structural" },
        },
        layout: { x: 0, y: 0, width: 10, height: 10 },
      },
      {
        id: "panel",
        capability: "core.rect",
        kind: capabilityKind("core.rect"),
        children: [],
        label: "panel",
        fallback: "panel",
      },
    );
    expect(lowered.style.fill).toEqual({
      kind: "theme-role",
      role: "surface.panel",
    });
    expect(lowered.style.stroke).toEqual({
      kind: "theme-role",
      role: "line.structural",
    });
  });
});

describe("connector endpoint lowering fails closed on malformed from/to", () => {
  it("omits `from`/`to` entirely rather than inventing a {x:0,y:0} origin when both are absent", () => {
    const lowered = lowerConnector({});
    expect(lowered.kind).toBe("connector");
    expect(lowered).not.toHaveProperty("from");
    expect(lowered).not.toHaveProperty("to");
  });

  it("omits a malformed `from` (no nodeId, no x/y) while keeping a valid `to`", () => {
    const lowered = lowerConnector({
      from: { anchor: "n" },
      to: { nodeId: "target" },
    });
    expect(lowered.kind).toBe("connector");
    expect(lowered).not.toHaveProperty("from");
    expect(lowered).toMatchObject({ to: { nodeId: "target" } });
  });

  it("omits a malformed `to` (non-object value) while keeping a valid `from`", () => {
    const lowered = lowerConnector({
      from: { nodeId: "source" },
      to: "not-an-endpoint-object",
    });
    expect(lowered.kind).toBe("connector");
    expect(lowered).toMatchObject({ from: { nodeId: "source" } });
    expect(lowered).not.toHaveProperty("to");
  });

  it("omits an endpoint with only a partial coordinate (x without y)", () => {
    const lowered = lowerConnector({
      from: { x: 5 },
      to: { nodeId: "target" },
    });
    expect(lowered).not.toHaveProperty("from");
  });

  it("resolves valid coordinate and nodeId endpoints without alteration", () => {
    const lowered = lowerConnector({
      from: { x: 5, y: 10 },
      to: { nodeId: "target", anchor: "w" },
    });
    expect(lowered).toMatchObject({
      from: { x: 5, y: 10 },
      to: { nodeId: "target", anchor: "w" },
    });
  });

  it("falls back to authored geometry instead of throwing when an endpoint cannot be resolved", () => {
    const lowered = lowerConnector({
      from: { anchor: "n" },
      to: { x: 10, y: 20 },
      geometry: { x: 0, y: 0, width: 0, height: 0 },
    });
    expect(lowered.geometry).toEqual({ x: 0, y: 0, width: 0, height: 0 });
  });

  it("skips malformed polyline points instead of injecting a synthetic origin point", () => {
    const lowered = lowerConnector({
      points: [{ x: 1, y: 2 }, { garbage: true }, { nodeId: "mid" }],
    });
    expect(lowered.points).toEqual([{ x: 1, y: 2 }, { nodeId: "mid" }]);
  });

  it("omits `from`/`to` for a path-based connector that never authored endpoints", () => {
    const lowered = lowerFirstClassPackageNode(
      { path: "M0 0 L10 10" },
      {
        id: "path-edge",
        capability: "core.path",
        kind: capabilityKind("core.path"),
        children: [],
        label: "path edge",
        fallback: "path edge",
      },
    );
    expect(lowered.kind).toBe("connector");
    expect(lowered).not.toHaveProperty("from");
    expect(lowered).not.toHaveProperty("to");
    expect(lowered).toMatchObject({ path: "M0 0 L10 10" });
  });
});

function lowerRect(node: Record<string, unknown>) {
  return lowerFirstClassPackageNode(node, {
    id: "box",
    capability: "core.rect",
    kind: capabilityKind("core.rect"),
    children: [],
    label: "box",
    fallback: "box",
  });
}

describe("geometryOf rejects non-finite and negative extents", () => {
  it("falls back to 0 for non-numeric geometry fields instead of emitting NaN", () => {
    const lowered = lowerRect({
      geometry: { x: "not-a-number", y: {}, width: "wide", height: null },
    });
    expect(lowered.geometry).toEqual({ x: 0, y: 0, width: 0, height: 0 });
    expect(Number.isFinite(lowered.geometry.x)).toBe(true);
    expect(Number.isFinite(lowered.geometry.y)).toBe(true);
    expect(Number.isFinite(lowered.geometry.width)).toBe(true);
    expect(Number.isFinite(lowered.geometry.height)).toBe(true);
  });

  it("clamps negative width and height to 0 while preserving finite negative x/y", () => {
    const lowered = lowerRect({
      geometry: { x: -10, y: -20, width: -5, height: -8 },
    });
    expect(lowered.geometry).toEqual({ x: -10, y: -20, width: 0, height: 0 });
  });

  it("preserves valid finite nonnegative geometry unchanged", () => {
    const lowered = lowerRect({
      geometry: { x: 12, y: 34, width: 56, height: 78 },
    });
    expect(lowered.geometry).toEqual({ x: 12, y: 34, width: 56, height: 78 });
  });

  it("falls back when width/height are Infinity", () => {
    const lowered = lowerRect({
      geometry: { x: 1, y: 2, width: Number.POSITIVE_INFINITY, height: Number.NaN },
    });
    expect(lowered.geometry).toEqual({ x: 1, y: 2, width: 0, height: 0 });
  });
});

function lowerArrow(node: Record<string, unknown>) {
  return lowerFirstClassPackageNode(node, {
    id: "arrow",
    capability: "core.arrow",
    kind: capabilityKind("core.arrow"),
    children: [],
    label: "arrow",
    fallback: "arrow",
  });
}

describe("core.arrow default arrowhead applies for all endpoint forms", () => {
  it("injects arrowhead defaults for nodeId-to-nodeId arrows", () => {
    const lowered = lowerArrow({
      from: { nodeId: "a" },
      to: { nodeId: "b" },
    });
    expect(lowered.kind).toBe("connector");
    expect(lowered.style).toMatchObject({
      arrowhead: true,
      markerEnd: "arrow",
    });
  });

  it("injects arrowhead defaults for absolute-coordinate arrows", () => {
    const lowered = lowerArrow({
      from: { x: 0, y: 0 },
      to: { x: 10, y: 10 },
    });
    expect(lowered.style).toMatchObject({
      arrowhead: true,
      markerEnd: "arrow",
    });
  });

  it("does not override an authored markerEnd / arrowhead", () => {
    const lowered = lowerArrow({
      from: { nodeId: "a" },
      to: { nodeId: "b" },
      style: { markerEnd: "none", arrowhead: false },
    });
    expect(lowered.style).toMatchObject({
      arrowhead: false,
      markerEnd: "none",
    });
  });
});
