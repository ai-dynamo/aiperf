/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure semantic layout resolution shared by render-time geometry verifiers.

import type {
  SceneGeometryLike,
  SceneNodeLike,
} from "../SceneRenderer.js";
import { resolveCapabilityLayout } from "./registry.js";

function finite(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function geometryOf(node: SceneNodeLike): SceneGeometryLike {
  const geometry = node.geometry ?? node.layout;
  return {
    x: finite(geometry?.x),
    y: finite(geometry?.y),
    width: Math.max(0, finite(geometry?.width)),
    height: Math.max(0, finite(geometry?.height)),
  };
}

function capabilityOf(node: SceneNodeLike): string {
  return String(node.capabilityId ?? node.capability ?? node.kind ?? "core.group");
}

function anchorPoint(
  geometry: SceneGeometryLike,
  anchor: unknown,
): Readonly<{ x: number; y: number }> {
  const center = {
    x: geometry.x + geometry.width / 2,
    y: geometry.y + geometry.height / 2,
  };
  switch (String(anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: geometry.x, y: center.y };
    case "right":
    case "east":
    case "e":
      return { x: geometry.x + geometry.width, y: center.y };
    case "top":
    case "north":
    case "n":
      return { x: center.x, y: geometry.y };
    case "bottom":
    case "south":
    case "s":
      return { x: center.x, y: geometry.y + geometry.height };
    case "ne":
      return { x: geometry.x + geometry.width, y: geometry.y };
    case "nw":
      return { x: geometry.x, y: geometry.y };
    case "se":
      return {
        x: geometry.x + geometry.width,
        y: geometry.y + geometry.height,
      };
    case "sw":
      return { x: geometry.x, y: geometry.y + geometry.height };
    default:
      return center;
  }
}

function resolveChildren(
  children: readonly SceneNodeLike[] | undefined,
): readonly SceneNodeLike[] | undefined {
  if (!Array.isArray(children)) return undefined;
  return children.map((child) => {
    const grandchildren = resolveChildren(child.children);
    const members = grandchildren ?? [];
    const layout = resolveCapabilityLayout(child, members);
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

function childrenFitParent(
  parent: SceneGeometryLike,
  children: readonly SceneNodeLike[],
): boolean {
  if (parent.width <= 0 || parent.height <= 0) return false;
  let sawGeometry = false;
  for (const child of children) {
    if (child.geometry === undefined && child.layout === undefined) continue;
    sawGeometry = true;
    const geometry = geometryOf(child);
    if (geometry.x < -0.5 || geometry.y < -0.5) return false;
    if (geometry.x + geometry.width > parent.width + 0.5) return false;
    if (geometry.y + geometry.height > parent.height + 0.5) return false;
  }
  return sawGeometry;
}

function childrenAreLocal(
  node: SceneNodeLike,
  parent: SceneGeometryLike,
  children: readonly SceneNodeLike[],
): boolean {
  if (children.length === 0) return false;
  const coordinateSpace = node.style?.coordinateSpace;
  if (coordinateSpace === "absolute") return false;
  if (coordinateSpace === "local") return true;
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
  if (childrenFitParent(parent, children)) return true;
  if (capability !== "core.group" && node.kind !== "group") return false;
  if (parent.width > 0 && parent.height > 0) return false;
  if (parent.x === 0 && parent.y === 0) return false;
  for (const child of children) {
    if (child.geometry === undefined && child.layout === undefined) continue;
    const geometry = geometryOf(child);
    if (geometry.x >= parent.x - 0.5 && geometry.y >= parent.y - 0.5) {
      return false;
    }
  }
  return true;
}

/**
 * Resolve native semantic capability layout bottom-up and return world bounds.
 */
export function resolveSceneWorldGeometry(
  roots: readonly SceneNodeLike[],
): ReadonlyMap<string, SceneGeometryLike> {
  const worldGeometryById = new Map<string, SceneGeometryLike>();

  const visit = (
    node: SceneNodeLike,
    originX: number,
    originY: number,
    coordinatesAreLocal: boolean,
    geometryOverride?: SceneGeometryLike,
  ): void => {
    let authored = geometryOverride ?? geometryOf(node);
    if (geometryOverride === undefined && node.relativePosition !== undefined) {
      const relative = node.relativePosition;
      const target = worldGeometryById.get(relative.nodeId);
      if (target !== undefined) {
        const anchor = anchorPoint(target, relative.anchor);
        const worldX = anchor.x + finite(relative.dx);
        const worldY = anchor.y + finite(relative.dy);
        authored = {
          ...authored,
          x: coordinatesAreLocal ? worldX - originX : worldX,
          y: coordinatesAreLocal ? worldY - originY : worldY,
        };
      }
    }

    const children = node.children ?? [];
    const resolvedChildren = resolveChildren(children) ?? [];
    const layout = resolveCapabilityLayout(
      { ...node, geometry: authored },
      resolvedChildren,
    );
    const world = coordinatesAreLocal
      ? {
          x: originX + layout.bounds.x,
          y: originY + layout.bounds.y,
          width: layout.bounds.width,
          height: layout.bounds.height,
        }
      : layout.bounds;
    worldGeometryById.set(node.id, world);

    const local = childrenAreLocal(node, layout.bounds, children);
    children.forEach((child, index) => {
      visit(
        child,
        local ? world.x : 0,
        local ? world.y : 0,
        local,
        layout.childGeometries[index],
      );
    });
  };

  roots.forEach((root) => visit(root, 0, 0, false));
  return worldGeometryById;
}

/**
 * Clone roots with resolved world bounds for consumers outside the TS graph.
 */
export function materializeSceneWorldGeometry(
  roots: readonly SceneNodeLike[],
): readonly SceneNodeLike[] {
  const world = resolveSceneWorldGeometry(roots);
  const materialize = (node: SceneNodeLike): SceneNodeLike => ({
    ...node,
    geometry: world.get(node.id) ?? geometryOf(node),
    children: node.children?.map(materialize),
  });
  return roots.map(materialize);
}
