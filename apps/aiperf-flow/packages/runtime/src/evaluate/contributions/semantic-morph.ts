// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Bounds,
  DrawCommand,
  HitRegion,
  LayerDrawCommand,
  PathDrawCommand,
} from "../../display-list.js";
import type {
  SemanticEntityProjection,
  SemanticRelationProjection,
} from "../types.js";

/** Authored entity geometry participating in a morph correspondence. */
export type SemanticMorphEntity = Readonly<{
  id: string;
  label: string;
  bounds: Bounds;
  kind?: string;
  role?: string;
  description?: string;
}>;

/** Deterministic source→target correspondence retained under every motion policy. */
export type SemanticMorphCorrespondence = Readonly<{
  id: string;
  sourceIds: readonly string[];
  targetIds: readonly string[];
  kind: string;
}>;

/** Reduced-motion degradation when spatial tweens are suppressed. */
export type SemanticMorphReducedMotionPolicy = "cut" | "crossfade";

/** Motion policy selected for the evaluated integer beat. */
export type SemanticMorphMotionMode = "tween" | "cut" | "crossfade";

export type SemanticMorphContributionInput = Readonly<{
  id: string;
  atMs: number;
  startMs: number;
  durationMs: number;
  sources: readonly SemanticMorphEntity[];
  targets: readonly SemanticMorphEntity[];
  correspondences: readonly SemanticMorphCorrespondence[];
  reducedMotion?: boolean;
  reducedMotionPolicy?: SemanticMorphReducedMotionPolicy;
  order?: number;
  fill?: string;
}>;

export type SemanticMorphContribution = Readonly<{
  commands: readonly DrawCommand[];
  hitRegions: readonly HitRegion[];
  semanticEntities: readonly SemanticEntityProjection[];
  semanticRelations: readonly SemanticRelationProjection[];
  correspondences: readonly SemanticMorphCorrespondence[];
  progress: number;
  motionMode: SemanticMorphMotionMode;
}>;

function integerTime(timeMs: number): number {
  if (!Number.isFinite(timeMs)) {
    return 0;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.trunc(timeMs)));
}

function rawProgress(
  atMs: number,
  startMs: number,
  durationMs: number,
): number {
  const time = integerTime(atMs);
  const start = integerTime(startMs);
  const duration = Math.max(0, integerTime(durationMs));
  if (duration === 0) {
    return time >= start ? 1 : 0;
  }
  if (time <= start) {
    return 0;
  }
  if (time >= start + duration) {
    return 1;
  }
  return (time - start) / duration;
}

function resolveProgress(
  progress: number,
  reducedMotion: boolean,
  policy: SemanticMorphReducedMotionPolicy,
): { progress: number; motionMode: SemanticMorphMotionMode } {
  if (!reducedMotion) {
    return { progress, motionMode: "tween" };
  }
  if (policy === "cut") {
    return { progress: progress < 0.5 ? 0 : 1, motionMode: "cut" };
  }
  return { progress, motionMode: "crossfade" };
}

function lerp(start: number, end: number, progress: number): number {
  return start + (end - start) * progress;
}

function lerpBounds(start: Bounds, end: Bounds, progress: number): Bounds {
  return {
    x: lerp(start.x, end.x, progress),
    y: lerp(start.y, end.y, progress),
    width: lerp(start.width, end.width, progress),
    height: lerp(start.height, end.height, progress),
  };
}

function unionBounds(boundsList: readonly Bounds[]): Bounds | undefined {
  if (boundsList.length === 0) {
    return undefined;
  }
  const left = Math.min(...boundsList.map(({ x }) => x));
  const top = Math.min(...boundsList.map(({ y }) => y));
  const right = Math.max(...boundsList.map(({ x, width }) => x + width));
  const bottom = Math.max(...boundsList.map(({ y, height }) => y + height));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

function rectanglePath({ x, y, width, height }: Bounds): string {
  return `M ${x} ${y} H ${x + width} V ${y + height} H ${x} Z`;
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

function projectEntity(entity: SemanticMorphEntity): SemanticEntityProjection {
  return {
    id: entity.id,
    label: entity.label,
    ...(entity.kind === undefined ? {} : { kind: entity.kind }),
    ...(entity.role === undefined ? {} : { role: entity.role }),
    ...(entity.description === undefined
      ? {}
      : { description: entity.description }),
  };
}

function projectRelations(
  correspondences: readonly SemanticMorphCorrespondence[],
): readonly SemanticRelationProjection[] {
  const relations: SemanticRelationProjection[] = [];
  for (const correspondence of correspondences) {
    for (const sourceId of correspondence.sourceIds) {
      for (const targetId of correspondence.targetIds) {
        relations.push({
          id: `${correspondence.id}:${sourceId}:${targetId}`,
          fromId: sourceId,
          toId: targetId,
          role: correspondence.kind,
        });
      }
    }
  }
  return relations;
}

function lookupBounds(
  ids: readonly string[],
  entities: ReadonlyMap<string, SemanticMorphEntity>,
): Bounds | undefined {
  const bounds = ids.flatMap((id) => {
    const entity = entities.get(id);
    return entity === undefined ? [] : [entity.bounds];
  });
  return unionBounds(bounds);
}

function pathCommand(
  id: string,
  order: number,
  bounds: Bounds,
  fill: string | undefined,
): PathDrawCommand {
  return {
    kind: "path",
    id,
    order,
    paintBounds: bounds,
    damageBounds: bounds,
    path: rectanglePath(bounds),
    ...(fill === undefined ? {} : { fill }),
  };
}

function layerCommand(
  id: string,
  order: number,
  bounds: Bounds,
  opacity: number,
  fill: string | undefined,
): LayerDrawCommand {
  return {
    kind: "layer",
    id,
    order,
    paintBounds: bounds,
    damageBounds: bounds,
    opacity,
    children: [pathCommand(`${id}:path`, order, bounds, fill)],
  };
}

function hitRegion(
  id: string,
  semanticId: string,
  order: number,
  bounds: Bounds,
): HitRegion {
  return {
    id: `${id}:hit`,
    semanticId,
    order,
    bounds,
  };
}

/**
 * Emits backend-neutral morph commands, hit regions, and stable semantic
 * identities for `core.semantic-morph` at one integer virtual time.
 */
export function contributeSemanticMorph(
  input: SemanticMorphContributionInput,
): SemanticMorphContribution {
  const order = input.order ?? 0;
  const policy = input.reducedMotionPolicy ?? "cut";
  const reducedMotion = input.reducedMotion === true;
  const { progress, motionMode } = resolveProgress(
    rawProgress(input.atMs, input.startMs, input.durationMs),
    reducedMotion,
    policy,
  );

  const sources = new Map(input.sources.map((entity) => [entity.id, entity]));
  const targets = new Map(input.targets.map((entity) => [entity.id, entity]));
  const commands: DrawCommand[] = [];
  const hitRegions: HitRegion[] = [];

  for (const correspondence of input.correspondences) {
    const sourceBounds = lookupBounds(correspondence.sourceIds, sources);
    const targetBounds = lookupBounds(correspondence.targetIds, targets);
    const commandBaseId = `${input.id}:${correspondence.id}`;

    if (sourceBounds === undefined && targetBounds === undefined) {
      continue;
    }

    if (sourceBounds === undefined && targetBounds !== undefined) {
      const opacity =
        motionMode === "cut" ? (progress < 0.5 ? 0 : 1) : progress;
      commands.push(
        layerCommand(commandBaseId, order, targetBounds, opacity, input.fill),
      );
      hitRegions.push(
        hitRegion(commandBaseId, correspondence.id, order, targetBounds),
      );
      continue;
    }

    if (sourceBounds !== undefined && targetBounds === undefined) {
      const opacity =
        motionMode === "cut" ? (progress < 0.5 ? 1 : 0) : 1 - progress;
      commands.push(
        layerCommand(commandBaseId, order, sourceBounds, opacity, input.fill),
      );
      hitRegions.push(
        hitRegion(commandBaseId, correspondence.id, order, sourceBounds),
      );
      continue;
    }

    if (sourceBounds === undefined || targetBounds === undefined) {
      continue;
    }

    if (motionMode === "crossfade") {
      commands.push(
        layerCommand(
          `${commandBaseId}:source`,
          order,
          sourceBounds,
          1 - progress,
          input.fill,
        ),
        layerCommand(
          `${commandBaseId}:target`,
          order,
          targetBounds,
          progress,
          input.fill,
        ),
      );
      hitRegions.push(
        hitRegion(
          commandBaseId,
          correspondence.id,
          order,
          progress < 0.5 ? sourceBounds : targetBounds,
        ),
      );
      continue;
    }

    const bounds = lerpBounds(sourceBounds, targetBounds, progress);
    commands.push(pathCommand(commandBaseId, order, bounds, input.fill));
    hitRegions.push(
      hitRegion(commandBaseId, correspondence.id, order, bounds),
    );
  }

  return deepFreeze({
    commands,
    hitRegions,
    semanticEntities: [
      ...input.sources.map(projectEntity),
      ...input.targets.map(projectEntity),
    ],
    semanticRelations: projectRelations(input.correspondences),
    correspondences: input.correspondences.map((correspondence) => ({
      id: correspondence.id,
      sourceIds: [...correspondence.sourceIds],
      targetIds: [...correspondence.targetIds],
      kind: correspondence.kind,
    })),
    progress,
    motionMode,
  });
}
