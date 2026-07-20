/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure deterministic traversal from authored Scene IR to canonical world layout.

import { resolveCapabilityLayout } from "../capabilities/registry.js";
import type { CapabilityLayoutDiagnostic } from "../capabilities/types.js";
import type {
  SceneGeometryLike,
  SceneIrLike,
  SceneNodeLike,
  SceneSourceRangeLike,
} from "../scene-types.js";
import type { ResolvedScene, SceneResolutionDiagnostic } from "./types.js";

/** Stable fallback source range for structural scenes without authored metadata. */
export const UNKNOWN_SCENE_RANGE: SceneSourceRangeLike = Object.freeze({
  source: "<scene>",
  start: Object.freeze({ offset: 0, line: 1, column: 1 }),
  end: Object.freeze({ offset: 0, line: 1, column: 1 }),
});

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function geometryOf(node: SceneNodeLike): SceneGeometryLike {
  const geometry = node.geometry ?? node.layout;
  return {
    x: finiteNumber(geometry?.x),
    y: finiteNumber(geometry?.y),
    width: finiteNumber(geometry?.width),
    height: finiteNumber(geometry?.height),
  };
}

function capabilityOf(node: SceneNodeLike): string {
  if (typeof node.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  return typeof node.kind === "string" && node.kind.length > 0
    ? `core.${node.kind}`
    : "";
}

function styleCoordinateSpace(
  style: SceneNodeLike["style"],
): "local" | "absolute" | undefined {
  const value = style?.coordinateSpace;
  return value === "absolute" || value === "local" ? value : undefined;
}

function childrenFitParentLocalBox(
  parentGeometry: SceneGeometryLike,
  children: readonly SceneNodeLike[],
): boolean {
  if (parentGeometry.width <= 0 || parentGeometry.height <= 0) {
    return false;
  }
  let sawGeometry = false;
  for (const child of children) {
    if (child.geometry === undefined && child.layout === undefined) {
      continue;
    }
    sawGeometry = true;
    const childGeometry = geometryOf(child);
    if (childGeometry.x < -0.5 || childGeometry.y < -0.5) {
      return false;
    }
    if (
      childGeometry.x + childGeometry.width > parentGeometry.width + 0.5 ||
      childGeometry.y + childGeometry.height > parentGeometry.height + 0.5
    ) {
      return false;
    }
  }
  return sawGeometry;
}

function childrenUseLocalLayout(
  node: SceneNodeLike,
  parentGeometry: SceneGeometryLike,
  children: readonly SceneNodeLike[] | undefined,
): boolean {
  if (!Array.isArray(children) || children.length === 0) {
    return false;
  }
  const space = styleCoordinateSpace(node.style);
  if (space === "absolute") {
    return false;
  }
  if (space === "local") {
    return true;
  }
  const capability = capabilityOf(node);
  if (
    capability === "core.panel" ||
    capability === "core.header" ||
    capability === "core.callout" ||
    capability === "core.lane" ||
    capability === "core.band" ||
    capability === "core.swimlane" ||
    capability === "core.stepper" ||
    capability.startsWith("layout.")
  ) {
    return true;
  }
  if (childrenFitParentLocalBox(parentGeometry, children)) {
    return true;
  }
  if (capability !== "core.group" && node.kind !== "group") {
    return false;
  }
  if (parentGeometry.width > 0 && parentGeometry.height > 0) {
    return false;
  }
  if (parentGeometry.x === 0 && parentGeometry.y === 0) {
    return false;
  }
  for (const child of children) {
    if (child.geometry === undefined && child.layout === undefined) {
      continue;
    }
    const childGeometry = geometryOf(child);
    if (
      childGeometry.x >= parentGeometry.x - 0.5 &&
      childGeometry.y >= parentGeometry.y - 0.5
    ) {
      return false;
    }
  }
  return true;
}

function nodeAnchorPoint(
  geometry: SceneGeometryLike,
  anchor: string | undefined,
): Readonly<{ x: number; y: number }> {
  const left = geometry.x;
  const right = geometry.x + geometry.width;
  const top = geometry.y;
  const bottom = geometry.y + geometry.height;
  const center = {
    x: geometry.x + geometry.width / 2,
    y: geometry.y + geometry.height / 2,
  };
  switch ((anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: left, y: center.y };
    case "right":
    case "east":
    case "e":
      return { x: right, y: center.y };
    case "top":
    case "north":
    case "n":
      return { x: center.x, y: top };
    case "bottom":
    case "south":
    case "s":
      return { x: center.x, y: bottom };
    case "ne":
      return { x: right, y: top };
    case "nw":
      return { x: left, y: top };
    case "se":
      return { x: right, y: bottom };
    case "sw":
      return { x: left, y: bottom };
    default:
      return center;
  }
}

function promoteLayoutDiagnostics(
  node: SceneNodeLike,
  diagnostics: readonly CapabilityLayoutDiagnostic[] | undefined,
  out: SceneResolutionDiagnostic[],
): void {
  if (diagnostics === undefined || diagnostics.length === 0) {
    return;
  }
  const range = node.sourceMap ?? UNKNOWN_SCENE_RANGE;
  for (const diagnostic of diagnostics) {
    out.push({
      code: diagnostic.code,
      severity: diagnostic.severity,
      message: diagnostic.message,
      range,
      nodeIds: diagnostic.nodeIds,
    });
  }
}

function resolveLayoutChildren(
  children: readonly SceneNodeLike[] | undefined,
  diagnostics: SceneResolutionDiagnostic[],
): readonly SceneNodeLike[] | undefined {
  if (!Array.isArray(children)) {
    return undefined;
  }
  return children.map((child) => {
    const grandchildren = resolveLayoutChildren(child.children, diagnostics);
    const members = grandchildren ?? [];
    const layout = resolveCapabilityLayout(child, members);
    promoteLayoutDiagnostics(child, layout.diagnostics, diagnostics);
    return {
      ...child,
      geometry: layout.bounds,
      children:
        grandchildren === undefined
          ? child.children
          : grandchildren.map((grandchild, index) => ({
              ...grandchild,
              geometry:
                layout.childGeometries[index] ?? geometryOf(grandchild),
            })),
    };
  });
}

/**
 * Resolve authored scene nodes into final world geometry in document order.
 *
 * The authored scene and nodes are retained by reference and never mutated.
 */
export function resolveScene(scene: SceneIrLike): ResolvedScene {
  const nodesById = new Map<string, SceneNodeLike>();
  const worldGeometryById = new Map<string, SceneGeometryLike>();
  const ancestorIdsById = new Map<string, readonly string[]>();
  const diagnostics: SceneResolutionDiagnostic[] = [];

  const visit = (
    node: SceneNodeLike,
    originX: number,
    originY: number,
    coordinatesAreLocal: boolean,
    geometryOverride: SceneGeometryLike | undefined,
    ancestors: readonly string[],
  ): void => {
    let authored = geometryOverride ?? geometryOf(node);
    if (geometryOverride === undefined && node.relativePosition !== undefined) {
      const target = worldGeometryById.get(node.relativePosition.nodeId);
      if (target !== undefined) {
        const anchor = nodeAnchorPoint(target, node.relativePosition.anchor);
        const worldX = anchor.x + finiteNumber(node.relativePosition.dx);
        const worldY = anchor.y + finiteNumber(node.relativePosition.dy);
        authored = {
          ...authored,
          x: coordinatesAreLocal ? worldX - originX : worldX,
          y: coordinatesAreLocal ? worldY - originY : worldY,
        };
      }
    }

    const children = node.children;
    const members = resolveLayoutChildren(children, diagnostics) ?? [];
    const layout = resolveCapabilityLayout(
      { ...node, geometry: authored },
      members,
    );
    promoteLayoutDiagnostics(node, layout.diagnostics, diagnostics);
    const worldGeometry = coordinatesAreLocal
      ? {
          x: originX + layout.bounds.x,
          y: originY + layout.bounds.y,
          width: layout.bounds.width,
          height: layout.bounds.height,
        }
      : layout.bounds;

    nodesById.set(node.id, node);
    worldGeometryById.set(node.id, worldGeometry);
    ancestorIdsById.set(node.id, Object.freeze([...ancestors]));

    if (!Array.isArray(children) || children.length === 0) {
      return;
    }
    const childAncestors = Object.freeze([...ancestors, node.id]);
    const local = childrenUseLocalLayout(node, layout.bounds, children);
    children.forEach((child, index) => {
      visit(
        child,
        local ? worldGeometry.x : 0,
        local ? worldGeometry.y : 0,
        local,
        layout.childGeometries[index],
        childAncestors,
      );
    });
  };

  for (const root of scene.roots) {
    visit(root, 0, 0, false, undefined, Object.freeze([]));
  }

  return Object.freeze({
    scene,
    nodesById,
    worldGeometryById,
    ancestorIdsById,
    generatedPartsById: new Map(),
    connectorsById: new Map(),
    diagnostics: Object.freeze(diagnostics),
  });
}
