// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic assembly of backend-neutral evaluator contributions.

import {
  buildDisplayList,
  type Bounds,
  type DisplayList,
  type DrawCommand,
  type HitRegion,
  type SourceReference,
} from "../display-list.js";
import type {
  SemanticEntityProjection,
  SemanticProjection,
  SemanticRelationProjection,
} from "./types.js";

/** A stable diagnostic emitted while evaluating one component. */
export type EvaluatorDiagnostic = Readonly<{
  id: string;
  severity: "error" | "warning" | "info";
  message: string;
  source?: SourceReference;
}>;

/** Backend-neutral products emitted by one component evaluator. */
export type EvaluatorContribution = Readonly<{
  id: string;
  sourceOrder: number;
  commands?: readonly DrawCommand[];
  hitRegions?: readonly HitRegion[];
  paintBounds?: Bounds;
  damageBounds?: Bounds;
  semanticEntities?: readonly SemanticEntityProjection[];
  semanticRelations?: readonly SemanticRelationProjection[];
  readingOrder?: readonly string[];
  captions?: readonly string[];
  diagnostics?: readonly EvaluatorDiagnostic[];
}>;

/** Canonical products assembled for one evaluated scene. */
export type MergedEvaluatorContributions = Readonly<{
  displayList: DisplayList;
  semantic: SemanticProjection;
  diagnostics: readonly EvaluatorDiagnostic[];
}>;

function compareContributions(
  left: EvaluatorContribution,
  right: EvaluatorContribution,
): number {
  if (left.sourceOrder !== right.sourceOrder) {
    return left.sourceOrder < right.sourceOrder ? -1 : 1;
  }
  return left.id.localeCompare(right.id);
}

function assertUniqueId(
  ids: Set<string>,
  id: string,
  kind: string,
): void {
  if (ids.has(id)) {
    throw new Error(`Duplicate ${kind} id "${id}".`);
  }
  ids.add(id);
}

function visitCommands(
  commands: readonly DrawCommand[],
  ids: Set<string>,
): void {
  for (const command of commands) {
    assertUniqueId(ids, command.id, "command");
    if (
      command.kind === "group" ||
      command.kind === "clip" ||
      command.kind === "layer"
    ) {
      visitCommands(command.children, ids);
    }
  }
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

function cloneEntity(
  entity: SemanticEntityProjection,
): SemanticEntityProjection {
  return {
    ...entity,
    ...(entity.evidenceIds === undefined
      ? {}
      : { evidenceIds: [...entity.evidenceIds] }),
    ...(entity.source === undefined ? {} : { source: { ...entity.source } }),
  };
}

function cloneRelation(
  relation: SemanticRelationProjection,
): SemanticRelationProjection {
  return {
    ...relation,
    ...(relation.source === undefined
      ? {}
      : { source: { ...relation.source } }),
  };
}

function cloneDiagnostic(
  diagnostic: EvaluatorDiagnostic,
): EvaluatorDiagnostic {
  return {
    ...diagnostic,
    ...(diagnostic.source === undefined
      ? {}
      : { source: { ...diagnostic.source } }),
  };
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) {
      deepFreeze(nested);
    }
    Object.freeze(value);
  }
  return value;
}

/**
 * Merges component products by source order and contribution id, rejecting
 * ambiguous identities and dangling semantic references.
 */
export function mergeContributions(
  sceneId: string,
  contributions: readonly EvaluatorContribution[],
): MergedEvaluatorContributions {
  const contributionIds = new Set<string>();
  for (const contribution of contributions) {
    assertUniqueId(contributionIds, contribution.id, "contribution");
    if (!Number.isSafeInteger(contribution.sourceOrder)) {
      throw new RangeError(
        `Contribution "${contribution.id}" sourceOrder must be a safe integer.`,
      );
    }
  }

  const ordered = [...contributions].sort(compareContributions);
  const commands: DrawCommand[] = [];
  const hitRegions: HitRegion[] = [];
  const entities: SemanticEntityProjection[] = [];
  const relations: SemanticRelationProjection[] = [];
  const readingOrder: string[] = [];
  const captions: string[] = [];
  const diagnostics: EvaluatorDiagnostic[] = [];
  const paintBounds: Bounds[] = [];
  const damageBounds: Bounds[] = [];

  const commandIds = new Set<string>();
  const hitRegionIds = new Set<string>();
  const entityIds = new Set<string>();
  const relationIds = new Set<string>();
  const readingOrderIds = new Set<string>();
  const diagnosticIds = new Set<string>();

  for (const contribution of ordered) {
    const contributedCommands = contribution.commands ?? [];
    visitCommands(contributedCommands, commandIds);
    for (const command of contributedCommands) {
      commands.push({ ...command, order: commands.length });
      paintBounds.push(command.paintBounds);
      damageBounds.push(command.damageBounds);
    }
    for (const region of contribution.hitRegions ?? []) {
      assertUniqueId(hitRegionIds, region.id, "hit region");
      hitRegions.push({ ...region, order: hitRegions.length });
    }
    for (const entity of contribution.semanticEntities ?? []) {
      assertUniqueId(entityIds, entity.id, "semantic entity");
      entities.push(cloneEntity(entity));
    }
    for (const relation of contribution.semanticRelations ?? []) {
      assertUniqueId(relationIds, relation.id, "semantic relation");
      relations.push(cloneRelation(relation));
    }
    for (const id of contribution.readingOrder ?? []) {
      assertUniqueId(readingOrderIds, id, "reading-order");
      readingOrder.push(id);
    }
    captions.push(...(contribution.captions ?? []));
    for (const diagnostic of contribution.diagnostics ?? []) {
      assertUniqueId(diagnosticIds, diagnostic.id, "diagnostic");
      diagnostics.push(cloneDiagnostic(diagnostic));
    }
    if (contribution.paintBounds !== undefined) {
      paintBounds.push(contribution.paintBounds);
    }
    if (contribution.damageBounds !== undefined) {
      damageBounds.push(contribution.damageBounds);
    }
  }

  for (const relation of relations) {
    if (entityIds.has(relation.id)) {
      throw new Error(`Duplicate semantic id "${relation.id}".`);
    }
    for (const endpoint of [relation.fromId, relation.toId]) {
      if (!entityIds.has(endpoint)) {
        throw new Error(
          `Semantic relation "${relation.id}" references unknown entity id "${endpoint}".`,
        );
      }
    }
  }
  const semanticIds = new Set([...entityIds, ...relationIds]);
  for (const region of hitRegions) {
    if (!semanticIds.has(region.semanticId)) {
      throw new Error(
        `Hit region "${region.id}" references unknown semantic id "${region.semanticId}".`,
      );
    }
  }
  for (const id of readingOrder) {
    if (!semanticIds.has(id)) {
      throw new Error(`Reading order references unknown semantic id "${id}".`);
    }
  }

  const semantic: SemanticProjection = {
    sceneId,
    entities,
    relations,
    readingOrder,
    ...(captions.length === 0 ? {} : { captions }),
  };
  return deepFreeze({
    displayList: buildDisplayList({
      commands,
      hitRegions,
      paintBounds: unionBounds(paintBounds),
      damageBounds: unionBounds(damageBounds),
    }),
    semantic,
    diagnostics,
  });
}
