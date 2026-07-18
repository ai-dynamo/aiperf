// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type SpanInterval = Readonly<{
  id: string;
  start: number; // inclusive
  end: number; // exclusive
}>;

export type SpanEdge = Readonly<{
  id: string;
  sourceSpanIds: readonly string[];
  targetSpanIds: readonly string[];
  kind: string;
}>;

export type SpanCoverageRequirement = "source" | "target" | "both" | "none";

export type SpanIntervalIndex = Readonly<{
  spans: readonly SpanInterval[];
  overlaps: readonly Readonly<{ left: string; right: string }>[];
  uncovered: readonly string[]; // span ids that fail requireCover
  covered: boolean;
}>;

function intervalsOverlap(left: SpanInterval, right: SpanInterval): boolean {
  return left.start < right.end && right.start < left.end;
}

/** Builds an overlap index for half-open span intervals. */
export function buildSpanIntervalIndex(
  spans: readonly SpanInterval[],
): SpanIntervalIndex {
  const overlaps: { left: string; right: string }[] = [];

  for (let leftIndex = 0; leftIndex < spans.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < spans.length; rightIndex += 1) {
      const first = spans[leftIndex];
      const second = spans[rightIndex];
      if (first === undefined || second === undefined) {
        continue;
      }
      if (!intervalsOverlap(first, second)) {
        continue;
      }

      const [left, right] =
        first.id.localeCompare(second.id) < 0 ? [first.id, second.id] : [second.id, first.id];
      overlaps.push({ left, right });
    }
  }

  return {
    spans,
    overlaps,
    uncovered: [],
    covered: true,
  };
}

/** Projects edge coverage requirements onto a span interval index. */
export function projectCoverage(
  spans: readonly SpanInterval[],
  edges: readonly SpanEdge[],
  requireCover: SpanCoverageRequirement,
): SpanIntervalIndex {
  const base = buildSpanIntervalIndex(spans);

  if (requireCover === "none") {
    return base;
  }

  const sourceCovered = new Set<string>();
  const targetCovered = new Set<string>();

  for (const edge of edges) {
    for (const spanId of edge.sourceSpanIds) {
      sourceCovered.add(spanId);
    }
    for (const spanId of edge.targetSpanIds) {
      targetCovered.add(spanId);
    }
  }

  const uncovered = new Set<string>();

  if (requireCover === "source" || requireCover === "both") {
    for (const span of spans) {
      if (!sourceCovered.has(span.id)) {
        uncovered.add(span.id);
      }
    }
  }

  if (requireCover === "target" || requireCover === "both") {
    for (const span of spans) {
      if (!targetCovered.has(span.id)) {
        uncovered.add(span.id);
      }
    }
  }

  const uncoveredIds = [...uncovered].sort((left, right) => left.localeCompare(right));

  return {
    spans,
    overlaps: base.overlaps,
    uncovered: uncoveredIds,
    covered: uncoveredIds.length === 0,
  };
}
