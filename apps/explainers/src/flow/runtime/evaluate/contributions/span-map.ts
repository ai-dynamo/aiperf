// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  diagnostic,
  type Diagnostic,
  type SourceRange,
} from "../../../schema/index.js";

import {
  type SpanCoverageRequirement,
  type SpanEdge,
  type SpanInterval,
  projectCoverage,
} from "../../leaves/span-interval.js";
import type {
  Bounds,
  DrawCommand,
  HitRegion,
  SourceReference,
} from "../../display-list.js";
import type {
  SemanticEntityProjection,
  SemanticRelationProjection,
} from "../types.js";

/** An authored span with backend-neutral geometry and semantics. */
export type SpanMapSpan = SpanInterval &
  Readonly<{
    label: string;
    bounds: Bounds;
    role?: string;
    description?: string;
    source?: SourceRange;
  }>;

/** An authored many-to-many mapping edge. */
export type SpanMapEdge = SpanEdge &
  Readonly<{
    label?: string;
    role?: string;
    source?: SourceRange;
  }>;

/** Input for one pure `core.span-map` evaluator contribution. */
export type SpanMapContributionInput = Readonly<{
  id: string;
  source: SourceRange;
  spans: readonly SpanMapSpan[];
  edges: readonly SpanMapEdge[];
  requireCover?: SpanCoverageRequirement;
  order?: number;
  spanFill?: string;
  spanStroke?: string;
  uncoveredStroke?: string;
  edgeStroke?: string;
  strokeWidth?: number;
}>;

/** Backend-neutral products emitted by `core.span-map`. */
export type SpanMapContribution = Readonly<{
  commands: readonly DrawCommand[];
  hitRegions: readonly HitRegion[];
  semanticEntities: readonly SemanticEntityProjection[];
  semanticRelations: readonly SemanticRelationProjection[];
  diagnostics: readonly Diagnostic[];
}>;

function sourceReference(range: SourceRange): SourceReference {
  return {
    source: range.source,
    startOffset: range.start.offset,
    endOffset: range.end.offset,
  };
}

function rectanglePath({ x, y, width, height }: Bounds): string {
  return `M ${x} ${y} H ${x + width} V ${y + height} H ${x} Z`;
}

function isFiniteBounds(bounds: Bounds): boolean {
  return (
    Number.isFinite(bounds.x) &&
    Number.isFinite(bounds.y) &&
    Number.isFinite(bounds.width) &&
    Number.isFinite(bounds.height) &&
    bounds.width >= 0 &&
    bounds.height >= 0
  );
}

function edgeGeometry(
  source: Bounds,
  target: Bounds,
): Readonly<{ path: string; bounds: Bounds; hitBounds: Bounds }> {
  const from = {
    x: source.x + source.width / 2,
    y: source.y + source.height / 2,
  };
  const to = {
    x: target.x + target.width / 2,
    y: target.y + target.height / 2,
  };
  const middleX = (from.x + to.x) / 2;
  const bounds = {
    x: Math.min(from.x, to.x),
    y: Math.min(from.y, to.y),
    width: Math.abs(to.x - from.x),
    height: Math.abs(to.y - from.y),
  };
  const hitPadding = 4;
  return {
    path: `M ${from.x} ${from.y} C ${middleX} ${from.y} ${middleX} ${to.y} ${to.x} ${to.y}`,
    bounds,
    hitBounds: {
      x: bounds.x - hitPadding,
      y: bounds.y - hitPadding,
      width: bounds.width + hitPadding * 2,
      height: bounds.height + hitPadding * 2,
    },
  };
}

function relationId(
  edge: SpanMapEdge,
  sourceId: string,
  targetId: string,
): string {
  if (edge.sourceSpanIds.length === 1 && edge.targetSpanIds.length === 1) {
    return edge.id;
  }
  return `${edge.id}:${sourceId}->${targetId}`;
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
 * Projects span interval analysis into immutable display, hit, semantic, and
 * diagnostic fragments without consulting a render backend or clock.
 */
export function contributeSpanMap(
  input: SpanMapContributionInput,
): SpanMapContribution {
  const diagnostics: Diagnostic[] = [];
  const spans: SpanMapSpan[] = [];
  const spanById = new Map<string, SpanMapSpan>();

  for (const span of input.spans) {
    const range = span.source ?? input.source;
    if (
      !Number.isFinite(span.start) ||
      !Number.isFinite(span.end) ||
      span.end < span.start ||
      !isFiniteBounds(span.bounds)
    ) {
      diagnostics.push(
        diagnostic(
          "SPAN_MAP_INVALID_SPAN",
          "error",
          `Span "${span.id}" in "${input.id}" must have finite half-open interval and bounds.`,
          range,
          "Provide finite bounds and an interval whose end is not before its start.",
        ),
      );
      continue;
    }
    if (spanById.has(span.id)) {
      diagnostics.push(
        diagnostic(
          "SPAN_MAP_DUPLICATE_SPAN",
          "error",
          `Span map "${input.id}" contains duplicate span id "${span.id}".`,
          range,
          "Use a unique authored semantic id for each span.",
        ),
      );
      continue;
    }
    spans.push(span);
    spanById.set(span.id, span);
  }

  const validEdges: SpanMapEdge[] = [];
  for (const edge of input.edges) {
    const unknownIds = [
      ...edge.sourceSpanIds,
      ...edge.targetSpanIds,
    ].filter((spanId) => !spanById.has(spanId));
    if (unknownIds.length > 0) {
      diagnostics.push(
        diagnostic(
          "SPAN_MAP_UNKNOWN_SPAN",
          "error",
          `Edge "${edge.id}" in "${input.id}" references unknown span ids: ${[
            ...new Set(unknownIds),
          ].join(", ")}.`,
          edge.source ?? input.source,
          "Reference authored span ids declared by this span map.",
        ),
      );
      continue;
    }
    validEdges.push(edge);
  }

  const requireCover = input.requireCover ?? "none";
  const coverage = projectCoverage(spans, validEdges, requireCover);
  const uncovered = new Set(coverage.uncovered);
  const baseOrder = input.order ?? 0;
  const strokeWidth = input.strokeWidth ?? 1;
  const commands: DrawCommand[] = [];
  const hitRegions: HitRegion[] = [];
  const semanticEntities: SemanticEntityProjection[] = [];
  const semanticRelations: SemanticRelationProjection[] = [];

  spans.forEach((span, index) => {
    const order = baseOrder + index;
    const source = sourceReference(span.source ?? input.source);
    const hasCoverageGap = uncovered.has(span.id);
    const coverageDescription = `Missing required "${requireCover}" mapping coverage.`;
    commands.push({
      kind: "path",
      id: `${input.id}:span:${span.id}`,
      order,
      paintBounds: span.bounds,
      damageBounds: span.bounds,
      path: rectanglePath(span.bounds),
      fill: input.spanFill ?? "transparent",
      stroke: hasCoverageGap
        ? (input.uncoveredStroke ?? "#ef4444")
        : (input.spanStroke ?? "#94a3b8"),
      strokeWidth,
      source,
    });
    hitRegions.push({
      id: `${input.id}:hit:${span.id}`,
      semanticId: span.id,
      order,
      bounds: span.bounds,
      source,
    });
    semanticEntities.push({
      id: span.id,
      label: span.label,
      role: span.role ?? "span",
      kind: hasCoverageGap ? "span-uncovered" : "span",
      ...(span.description === undefined && !hasCoverageGap
        ? {}
        : {
            description: [
              span.description,
              hasCoverageGap ? coverageDescription : undefined,
            ]
              .filter((part): part is string => part !== undefined)
              .join(" "),
          }),
      source,
    });
  });

  for (const spanId of coverage.uncovered) {
    diagnostics.push(
      diagnostic(
        "SPAN_MAP_COVERAGE_GAP",
        "warning",
        `Span "${spanId}" does not satisfy required "${requireCover}" edge coverage in "${input.id}".`,
        spanById.get(spanId)?.source ?? input.source,
        `Add an edge covering "${spanId}" or relax the requireCover policy.`,
      ),
    );
  }

  let pairIndex = 0;
  for (const edge of validEdges) {
    for (const sourceId of edge.sourceSpanIds) {
      for (const targetId of edge.targetSpanIds) {
        const sourceSpan = spanById.get(sourceId);
        const targetSpan = spanById.get(targetId);
        if (sourceSpan === undefined || targetSpan === undefined) {
          continue;
        }
        const id = relationId(edge, sourceId, targetId);
        const order = baseOrder + spans.length + pairIndex;
        const geometry = edgeGeometry(sourceSpan.bounds, targetSpan.bounds);
        const source = sourceReference(edge.source ?? input.source);
        commands.push({
          kind: "path",
          id: `${input.id}:edge:${id}`,
          order,
          paintBounds: geometry.bounds,
          damageBounds: geometry.bounds,
          path: geometry.path,
          fill: "transparent",
          stroke: input.edgeStroke ?? "#38bdf8",
          strokeWidth,
          source,
        });
        hitRegions.push({
          id: `${input.id}:hit:${id}`,
          semanticId: id,
          order,
          bounds: geometry.hitBounds,
          source,
        });
        semanticRelations.push({
          id,
          fromId: sourceId,
          toId: targetId,
          label: edge.label ?? edge.kind,
          role: edge.role ?? edge.kind,
          source,
        });
        pairIndex += 1;
      }
    }
  }

  return deepFreeze({
    commands,
    hitRegions,
    semanticEntities,
    semanticRelations,
    diagnostics,
  });
}
