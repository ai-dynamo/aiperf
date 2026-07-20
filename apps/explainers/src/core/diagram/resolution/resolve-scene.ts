/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure deterministic traversal from authored Scene IR to canonical world layout.

import { resolveCapabilityLayout } from "../capabilities/registry.js";
import {
  hasNativeSemanticChrome,
  resolveSemanticChrome,
} from "../capabilities/chrome.js";
import type { CapabilityLayoutDiagnostic } from "../capabilities/types.js";
import type {
  SceneGeometryLike,
  SceneIrLike,
  SceneNodeLike,
  SceneSourceRangeLike,
} from "../scene-types.js";
import { resolveConnectors } from "./resolve-connectors.js";
import type {
  ResolvedConnector,
  ResolvedGeneratedPart,
  ResolvedScene,
  SceneResolutionDiagnostic,
} from "./types.js";

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

function boundsOverlap(
  left: SceneGeometryLike,
  right: SceneGeometryLike,
): boolean {
  return (
    left.x < right.x + right.width &&
    left.x + left.width > right.x &&
    left.y < right.y + right.height &&
    left.y + left.height > right.y
  );
}

function boundsEscapeViewport(
  bounds: SceneGeometryLike,
  viewport: Readonly<{ width: number; height: number }>,
): boolean {
  return (
    bounds.x < 0 ||
    bounds.y < 0 ||
    bounds.x + bounds.width > viewport.width ||
    bounds.y + bounds.height > viewport.height
  );
}

function appendFinalValidationDiagnostics(input: {
  scene: SceneIrLike;
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
  generatedPartsById: ReadonlyMap<string, ResolvedGeneratedPart>;
  connectorsById: ReadonlyMap<string, ResolvedConnector>;
  diagnostics: SceneResolutionDiagnostic[];
}): void {
  const viewport = input.scene.viewport ?? { width: 700, height: 400 };
  const siblingGroups = new Map<string, string[]>();
  const escapedNodeIds = new Set<string>();
  for (const [id, node] of input.nodesById) {
    const bounds = input.worldGeometryById.get(id);
    if (
      bounds === undefined ||
      bounds.width <= 0 ||
      bounds.height <= 0 ||
      input.connectorsById.has(id)
    ) {
      continue;
    }
    const ancestors = input.ancestorIdsById.get(id) ?? [];
    const parentId = ancestors.at(-1);
    const parent = parentId === undefined ? undefined : input.nodesById.get(parentId);
    if (capabilityOf(parent ?? { id: "" }) !== "layout.overlay") {
      const key = parentId ?? "<root>";
      const siblings = siblingGroups.get(key) ?? [];
      siblings.push(id);
      siblingGroups.set(key, siblings);
    }
    if (boundsEscapeViewport(bounds, viewport)) {
      escapedNodeIds.add(id);
      input.diagnostics.push({
        code: "SCENE_VIEWPORT_ESCAPE",
        severity: "warning",
        message: `Node "${id}" exceeds the ${viewport.width}×${viewport.height} scene viewport.`,
        range: node.sourceMap ?? UNKNOWN_SCENE_RANGE,
        nodeIds: [id],
        repair: "Move or resize the node so its resolved bounds remain inside the viewport.",
      });
    }
  }
  for (const siblingIds of siblingGroups.values()) {
    for (let leftIndex = 0; leftIndex < siblingIds.length; leftIndex += 1) {
      for (
        let rightIndex = leftIndex + 1;
        rightIndex < siblingIds.length;
        rightIndex += 1
      ) {
        const leftId = siblingIds[leftIndex];
        const rightId = siblingIds[rightIndex];
        const leftNode = input.nodesById.get(leftId);
        const rightNode = input.nodesById.get(rightId);
        if (
          capabilityOf(leftNode ?? { id: "" }) === "core.band" ||
          capabilityOf(leftNode ?? { id: "" }) === "core.bracket" ||
          capabilityOf(leftNode ?? { id: "" }) === "core.divider" ||
          capabilityOf(rightNode ?? { id: "" }) === "core.band" ||
          capabilityOf(rightNode ?? { id: "" }) === "core.bracket" ||
          capabilityOf(rightNode ?? { id: "" }) === "core.divider"
        ) {
          continue;
        }
        const left = input.worldGeometryById.get(leftId);
        const right = input.worldGeometryById.get(rightId);
        if (left === undefined || right === undefined || !boundsOverlap(left, right)) {
          continue;
        }
        input.diagnostics.push({
          code: "SCENE_ABSOLUTE_SIBLING_OVERLAP",
          severity: "warning",
          message: `Resolved siblings "${leftId}" and "${rightId}" overlap.`,
          range: rightNode?.sourceMap ?? UNKNOWN_SCENE_RANGE,
          nodeIds: [leftId, rightId],
          repair: "Move the siblings apart or place intentional overlap in layout.overlay.",
        });
      }
    }
  }
  for (const part of input.generatedPartsById.values()) {
    if (
      escapedNodeIds.has(part.ownerId) ||
      !boundsEscapeViewport(part.geometry, viewport)
    ) {
      continue;
    }
    input.diagnostics.push({
      code: "SCENE_VIEWPORT_ESCAPE",
      severity: "warning",
      message: `Generated part "${part.id}" exceeds the scene viewport.`,
      range: input.nodesById.get(part.ownerId)?.sourceMap ?? UNKNOWN_SCENE_RANGE,
      nodeIds: [part.ownerId, part.id],
    });
  }
  for (const connector of input.connectorsById.values()) {
    if (!connector.showArrowhead) continue;
    if (
      connector.target.x >= 0 &&
      connector.target.y >= 0 &&
      connector.target.x <= viewport.width &&
      connector.target.y <= viewport.height
    ) {
      continue;
    }
    input.diagnostics.push({
      code: "SCENE_VIEWPORT_ESCAPE",
      severity: "warning",
      message: `Arrow tip for "${connector.id}" exceeds the scene viewport.`,
      range: input.nodesById.get(connector.id)?.sourceMap ?? UNKNOWN_SCENE_RANGE,
      nodeIds: [connector.id],
    });
  }
  for (const diagnostic of [...input.diagnostics]) {
    if (
      diagnostic.code !== "SCENE_MANAGED_CONTENT_OVERFLOW" ||
      input.diagnostics.some(
        (candidate) =>
          candidate.code === "SCENE_FIXED_CONTENT_OVERFLOW" &&
          candidate.nodeIds[0] === diagnostic.nodeIds[0],
      )
    ) {
      continue;
    }
    input.diagnostics.push({
      ...diagnostic,
      code: "SCENE_FIXED_CONTENT_OVERFLOW",
    });
  }
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
  const visitedNodes: SceneNodeLike[] = [];

  const visit = (
    node: SceneNodeLike,
    originX: number,
    originY: number,
    coordinatesAreLocal: boolean,
    geometryOverride: SceneGeometryLike | undefined,
    ancestors: readonly string[],
  ): void => {
    let authored = geometryOverride ?? geometryOf(node);
    if (node.relativePosition !== undefined) {
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
      } else {
        diagnostics.push({
          code: "SCENE_RELATIVE_POSITION_TARGET_MISSING",
          severity: "error",
          message: `Node "${node.id}" has a relativePosition referencing unresolved node "${node.relativePosition.nodeId}".`,
          range: node.sourceMap ?? UNKNOWN_SCENE_RANGE,
          nodeIds: [node.id, node.relativePosition.nodeId],
          repair:
            "Reference a node id that resolves before this node, or author explicit geometry.",
        });
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

    if (nodesById.has(node.id)) {
      diagnostics.push({
        code: "SCENE_DUPLICATE_NODE_ID",
        severity: "error",
        message: `Node id "${node.id}" is used by more than one node in the scene.`,
        range: node.sourceMap ?? UNKNOWN_SCENE_RANGE,
        nodeIds: [node.id],
        repair: "Give each authored node a unique id.",
      });
    }
    nodesById.set(node.id, node);
    worldGeometryById.set(node.id, worldGeometry);
    ancestorIdsById.set(node.id, Object.freeze([...ancestors]));
    visitedNodes.push(node);

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
  const generatedPartsById = new Map<string, ResolvedGeneratedPart>();
  for (const node of visitedNodes) {
    if (!hasNativeSemanticChrome(node)) {
      continue;
    }
    const geometry = worldGeometryById.get(node.id);
    if (geometry === undefined) {
      continue;
    }
    const chrome = resolveSemanticChrome(node, geometry);
    const parts = [
      ...(chrome.rootBox === undefined ? [] : [chrome.rootBox]),
      ...chrome.boxes,
      ...chrome.texts,
    ];
    for (const part of parts) {
      const authoredOwner = nodesById.get(part.id);
      const generatedOwner = generatedPartsById.get(part.id);
      const isSemanticStepperPart =
        part.role === "step" &&
        authoredOwner !== undefined &&
        capabilityOf(authoredOwner) === "core.step" &&
        (ancestorIdsById.get(authoredOwner.id) ?? []).includes(node.id);
      const priorOwnerId =
        (isSemanticStepperPart ? undefined : authoredOwner?.id) ??
        generatedOwner?.ownerId;
      if (priorOwnerId !== undefined) {
        diagnostics.push({
          code: "SCENE_DUPLICATE_PAINT_OWNER",
          severity: "error",
          message: `Generated part "${part.id}" is owned by both "${priorOwnerId}" and "${node.id}".`,
          range: node.sourceMap ?? UNKNOWN_SCENE_RANGE,
          nodeIds: [priorOwnerId, node.id],
          repair:
            "Remove the compatibility child; semantic chrome owns this role.",
        });
        if (generatedOwner !== undefined) {
          diagnostics.push({
            code: "SCENE_DUPLICATE_GENERATED_ID",
            severity: "error",
            message: `Generated part ID "${part.id}" is repeated by "${generatedOwner.ownerId}" and "${node.id}".`,
            range: node.sourceMap ?? UNKNOWN_SCENE_RANGE,
            nodeIds: [generatedOwner.ownerId, node.id],
            repair: "Give each semantic chrome owner a unique authored ID.",
          });
        }
        continue;
      }
      const partGeometry =
        "geometry" in part
          ? part.geometry
          : {
              x: part.x,
              y: part.y,
              width: part.width,
              height: part.height,
            };
      const generatedPart: ResolvedGeneratedPart = {
        id: part.id,
        ownerId: node.id,
        role: part.role,
        kind: "geometry" in part ? "box" : "text",
        geometry: partGeometry,
        ...("geometry" in part
          ? { radius: part.radius }
          : {
              text: part.text,
              fontSize: part.fontSize,
              ...(part.fontWeight === undefined
                ? {}
                : { fontWeight: part.fontWeight }),
              ...(part.fontFamily === undefined
                ? {}
                : { fontFamily: part.fontFamily }),
              ...(part.fontStyle === undefined
                ? {}
                : { fontStyle: part.fontStyle }),
              ...(part.whiteSpace === undefined
                ? {}
                : { whiteSpace: part.whiteSpace }),
              anchor: part.anchor,
              ...(part.tone === undefined ? {} : { tone: part.tone }),
              ...(part.inkRole === undefined
                ? {}
                : { inkRole: part.inkRole }),
            }),
      };
      generatedPartsById.set(part.id, generatedPart);
      worldGeometryById.set(part.id, partGeometry);
    }
  }
  const connectors = resolveConnectors({
    nodesById,
    worldGeometryById,
    ancestorIdsById,
    generatedPartIds: new Set(generatedPartsById.keys()),
  });
  diagnostics.push(...connectors.diagnostics);
  appendFinalValidationDiagnostics({
    scene,
    nodesById,
    worldGeometryById,
    ancestorIdsById,
    generatedPartsById,
    connectorsById: connectors.connectorsById,
    diagnostics,
  });
  diagnostics.sort(
    (left, right) =>
      left.range.source.localeCompare(right.range.source) ||
      left.range.start.offset - right.range.start.offset ||
      left.code.localeCompare(right.code) ||
      (left.nodeIds[0] ?? "").localeCompare(right.nodeIds[0] ?? ""),
  );

  return Object.freeze({
    scene,
    nodesById,
    worldGeometryById,
    ancestorIdsById,
    generatedPartsById,
    connectorsById: connectors.connectorsById,
    diagnostics: Object.freeze(diagnostics),
  });
}
