// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure foundation Flow IR scene evaluation.

import {
  resolveCapabilityId,
  type ComponentNodeIr,
  type RenderNodeIr,
  type SceneIr,
  type SourceRange,
} from "@aiperf/flow-schema";

import {
  buildDisplayList,
  type Bounds,
  type DrawCommand,
  type HitRegion,
  type SourceReference,
} from "../display-list.js";
import {
  mergeContributions,
  type EvaluatorContribution,
} from "./merge-contributions.js";
import type {
  CapabilityContribution,
  FrozenCapabilityEvaluatorRegistry,
} from "./registry.js";
import type {
  EvaluatedScene,
  SemanticEntityProjection,
  SemanticProjection,
  SemanticRelationProjection,
} from "./types.js";

/** Optional injected capability evaluators for component-node dispatch. */
export type EvaluateSceneOptions = Readonly<{
  evaluators?: FrozenCapabilityEvaluatorRegistry;
}>;

function sourceReference(range: SourceRange): SourceReference {
  return {
    source: range.source,
    startOffset: range.start.offset,
    endOffset: range.end.offset,
  };
}

function geometry(node: RenderNodeIr): Bounds {
  const { x, y, width, height } = node.geometry;
  if (
    ![x, y, width, height].every(Number.isFinite) ||
    width < 0 ||
    height < 0
  ) {
    throw new Error(`Node "${node.id}" geometry must contain finite numbers.`);
  }
  return { x, y, width, height };
}

function unionBounds(bounds: readonly Bounds[]): Bounds {
  if (bounds.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  const left = Math.min(...bounds.map(({ x }) => x));
  const top = Math.min(...bounds.map(({ y }) => y));
  const right = Math.max(...bounds.map(({ x, width }) => x + width));
  const bottom = Math.max(...bounds.map(({ y, height }) => y + height));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

function pathStyle(
  style: RenderNodeIr["style"],
): Readonly<{
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
}> {
  return {
    ...(typeof style.fill === "string" ? { fill: style.fill } : {}),
    ...(typeof style.stroke === "string" ? { stroke: style.stroke } : {}),
    ...(typeof style.strokeWidth === "number"
      ? { strokeWidth: style.strokeWidth }
      : {}),
  };
}

function rectanglePath({ x, y, width, height }: Bounds): string {
  return `M ${x} ${y} H ${x + width} V ${y + height} H ${x} Z`;
}

type EvaluationIndex = Readonly<{
  nodes: Map<string, RenderNodeIr>;
  order: Map<string, number>;
}>;

type EvaluationPass = Readonly<{
  atMs: number;
  evaluators: FrozenCapabilityEvaluatorRegistry | undefined;
  contributions: EvaluatorContribution[];
}>;

function indexNodes(scene: SceneIr): EvaluationIndex {
  const nodes = new Map<string, RenderNodeIr>();
  const order = new Map<string, number>();

  function visit(node: RenderNodeIr): void {
    if (nodes.has(node.id)) {
      throw new Error(`Duplicate scene node id "${node.id}".`);
    }
    geometry(node);
    order.set(node.id, order.size);
    nodes.set(node.id, node);
    if (node.kind === "group" || node.kind === "component") {
      node.children.forEach(visit);
    }
  }

  scene.roots.forEach(visit);
  return { nodes, order };
}

function connectorBounds(
  node: Extract<RenderNodeIr, { kind: "connector" }>,
  index: EvaluationIndex,
): Readonly<{
  bounds: Bounds;
  from: Readonly<{ x: number; y: number }>;
  to: Readonly<{ x: number; y: number }>;
}> {
  const fromNode = index.nodes.get(node.from.nodeId);
  const toNode = index.nodes.get(node.to.nodeId);
  if (fromNode === undefined || toNode === undefined) {
    throw new Error(`Connector "${node.id}" references an unknown endpoint.`);
  }
  const fromGeometry = geometry(fromNode);
  const toGeometry = geometry(toNode);
  const from = {
    x: fromGeometry.x + fromGeometry.width / 2,
    y: fromGeometry.y + fromGeometry.height / 2,
  };
  const to = {
    x: toGeometry.x + toGeometry.width / 2,
    y: toGeometry.y + toGeometry.height / 2,
  };
  return {
    bounds: {
      x: Math.min(from.x, to.x),
      y: Math.min(from.y, to.y),
      width: Math.abs(to.x - from.x),
      height: Math.abs(to.y - from.y),
    },
    from,
    to,
  };
}

function toEvaluatorContribution(
  node: ComponentNodeIr,
  contribution: CapabilityContribution,
  sourceOrder: number,
): EvaluatorContribution {
  return {
    id: node.id,
    sourceOrder,
    commands: contribution.display.commands,
    hitRegions: contribution.display.hitRegions,
    semanticEntities: contribution.semantic.entities,
    semanticRelations: contribution.semantic.relations,
    readingOrder: contribution.semantic.readingOrder,
  };
}

function evaluateComponent(
  node: ComponentNodeIr,
  order: number,
  index: EvaluationIndex,
  pass: EvaluationPass,
): DrawCommand {
  if (pass.evaluators === undefined) {
    throw new Error(
      `Foundation evaluator cannot evaluate component "${node.id}".`,
    );
  }

  const capabilityId = resolveCapabilityId(node);
  const contribution = pass.evaluators
    .require(capabilityId)
    .evaluate(node, { atMs: pass.atMs });
  const sourceOrder = index.order.get(node.id);
  if (sourceOrder === undefined) {
    throw new Error(`Component "${node.id}" is missing from the evaluation index.`);
  }
  pass.contributions.push(
    toEvaluatorContribution(node, contribution, sourceOrder),
  );

  const bounds = geometry(node);
  return {
    id: node.id,
    order,
    paintBounds: bounds,
    damageBounds: bounds,
    source: sourceReference(node.sourceMap),
    kind: "group",
    children: contribution.display.commands.map((command, childOrder) => ({
      ...command,
      order: childOrder,
    })),
  };
}

function drawCommand(
  node: RenderNodeIr,
  order: number,
  index: EvaluationIndex,
  pass: EvaluationPass,
): DrawCommand {
  const bounds = geometry(node);
  const base = {
    id: node.id,
    order,
    paintBounds: bounds,
    damageBounds: bounds,
    source: sourceReference(node.sourceMap),
  };

  switch (node.kind) {
    case "group":
      return {
        ...base,
        kind: "group",
        children: node.children.map((child, childOrder) =>
          drawCommand(child, childOrder, index, pass),
        ),
      };
    case "rect":
      return {
        ...base,
        kind: "path",
        path: rectanglePath(bounds),
        ...pathStyle(node.style),
      };
    case "text":
      return {
        ...base,
        kind: "text",
        text: node.text,
        origin: { x: bounds.x, y: bounds.y },
        font: {
          family:
            typeof node.style.fontFamily === "string"
              ? node.style.fontFamily
              : "sans-serif",
          sizePx:
            typeof node.style.fontSize === "number" ? node.style.fontSize : 16,
          ...(typeof node.style.fontWeight === "number"
            ? { weight: node.style.fontWeight }
            : {}),
        },
        ...(typeof node.style.fill === "string"
          ? { fill: node.style.fill }
          : {}),
      };
    case "connector": {
      const connector = connectorBounds(node, index);
      return {
        ...base,
        kind: "path",
        paintBounds: connector.bounds,
        damageBounds: connector.bounds,
        path: `M ${connector.from.x} ${connector.from.y} L ${connector.to.x} ${connector.to.y}`,
        ...pathStyle(node.style),
      };
    }
    case "component":
      return evaluateComponent(node, order, index, pass);
  }
}

function semanticProjection(
  scene: SceneIr,
  index: EvaluationIndex,
): SemanticProjection {
  const entities: SemanticEntityProjection[] = [];
  const relations: SemanticRelationProjection[] = [];

  for (const id of scene.accessibility.readingOrder) {
    const node = index.nodes.get(id);
    if (node === undefined) {
      throw new Error(`Accessibility reading order references unknown node "${id}".`);
    }
    const common = {
      id,
      label: node.accessibility.label,
      ...(node.accessibility.description === undefined
        ? {}
        : { description: node.accessibility.description }),
    };
    if (node.kind === "connector") {
      relations.push({
        id,
        fromId: node.from.nodeId,
        toId: node.to.nodeId,
        label: node.accessibility.label,
      });
    } else {
      entities.push(common);
    }
  }

  return {
    sceneId: scene.id,
    entities,
    relations,
    readingOrder: [...scene.accessibility.readingOrder],
  };
}

function assertUniqueSemanticId(ids: Set<string>, id: string, kind: string): void {
  if (ids.has(id)) {
    throw new Error(`Duplicate ${kind} id "${id}".`);
  }
  ids.add(id);
}

function mergeComponentProducts(
  foundation: EvaluatedScene,
  contributions: readonly EvaluatorContribution[],
): EvaluatedScene {
  if (contributions.length === 0) {
    return foundation;
  }

  const merged = mergeContributions(foundation.sceneId, contributions);
  const entityIds = new Set(foundation.semantic.entities.map(({ id }) => id));
  const relationIds = new Set(
    foundation.semantic.relations.map(({ id }) => id),
  );

  for (const entity of merged.semantic.entities) {
    assertUniqueSemanticId(entityIds, entity.id, "semantic entity");
  }
  for (const relation of merged.semantic.relations) {
    assertUniqueSemanticId(relationIds, relation.id, "semantic relation");
    if (entityIds.has(relation.id)) {
      throw new Error(`Duplicate semantic id "${relation.id}".`);
    }
  }

  const readingOrder = [
    ...foundation.semantic.readingOrder,
    ...merged.semantic.readingOrder.filter(
      (id) => !foundation.semantic.readingOrder.includes(id),
    ),
  ];
  const hitRegions: HitRegion[] = [
    ...foundation.displayList.hitRegions,
    ...merged.displayList.hitRegions.map((region, offset) => ({
      ...region,
      order: foundation.displayList.hitRegions.length + offset,
    })),
  ];
  const paintBounds = unionBounds([
    foundation.displayList.paintBounds,
    merged.displayList.paintBounds,
  ]);
  const damageBounds = unionBounds([
    foundation.displayList.damageBounds,
    merged.displayList.damageBounds,
  ]);

  return {
    sceneId: foundation.sceneId,
    atMs: foundation.atMs,
    displayList: buildDisplayList({
      commands: foundation.displayList.commands,
      hitRegions,
      paintBounds,
      damageBounds,
    }),
    semantic: {
      sceneId: foundation.sceneId,
      entities: [...foundation.semantic.entities, ...merged.semantic.entities],
      relations: [
        ...foundation.semantic.relations,
        ...merged.semantic.relations,
      ],
      readingOrder,
      ...(foundation.semantic.transcriptCueId === undefined
        ? {}
        : { transcriptCueId: foundation.semantic.transcriptCueId }),
      ...(merged.semantic.captions === undefined &&
      foundation.semantic.captions === undefined
        ? {}
        : {
            captions: [
              ...(foundation.semantic.captions ?? []),
              ...(merged.semantic.captions ?? []),
            ],
          }),
    },
  };
}

/** Evaluates foundation Scene IR into deterministic backend-neutral products. */
export function evaluateScene(
  scene: SceneIr,
  atMs = 0,
  options: EvaluateSceneOptions = {},
): EvaluatedScene {
  if (!Number.isSafeInteger(atMs) || atMs < 0) {
    throw new RangeError("Scene evaluation time must be a non-negative integer.");
  }

  const index = indexNodes(scene);
  const pass: EvaluationPass = {
    atMs,
    evaluators: options.evaluators,
    contributions: [],
  };
  const commands = scene.roots.map((node, order) =>
    drawCommand(node, order, index, pass),
  );
  const paintBounds = unionBounds(commands.map(({ paintBounds }) => paintBounds));
  const hitRegions: HitRegion[] = scene.accessibility.readingOrder.map(
    (id, order) => {
      const node = index.nodes.get(id);
      if (node === undefined) {
        throw new Error(
          `Accessibility reading order references unknown node "${id}".`,
        );
      }
      return {
        id: `hit:${id}`,
        semanticId: id,
        order,
        bounds: geometry(node),
        source: sourceReference(node.sourceMap),
      };
    },
  );

  const foundation: EvaluatedScene = {
    sceneId: scene.id,
    atMs,
    displayList: buildDisplayList({
      commands,
      hitRegions,
      paintBounds,
      damageBounds: paintBounds,
    }),
    semantic: semanticProjection(scene, index),
  };

  return mergeComponentProducts(foundation, pass.contributions);
}
