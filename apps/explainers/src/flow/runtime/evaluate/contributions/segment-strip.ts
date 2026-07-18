// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Bounds,
  DrawCommand,
  HitRegion,
  PathDrawCommand,
} from "../../display-list.js";
import {
  layoutSegmentStrip,
  type SegmentStripInput,
  type SegmentStripLayoutOptions,
} from "../../leaves/segment-strip-layout.js";
import type { SemanticEntityProjection } from "../types.js";

export type SegmentStripStyle = Readonly<{
  fill?: string;
  textFill?: string;
  continuationStroke?: string;
  font?: Readonly<{
    family: string;
    sizePx: number;
    weight?: number;
  }>;
}>;

export type SegmentStripContributionInput = Readonly<{
  id: string;
  segments: readonly SegmentStripInput[];
  layout: SegmentStripLayoutOptions;
  order?: number;
  style?: SegmentStripStyle;
}>;

export type SegmentStripContribution = Readonly<{
  commands: readonly DrawCommand[];
  semanticEntities: readonly SemanticEntityProjection[];
  hitRegions: readonly HitRegion[];
}>;

const DEFAULT_FONT = { family: "sans-serif", sizePx: 12 } as const;

function rectanglePath(bounds: Bounds): string {
  const right = bounds.x + bounds.width;
  const bottom = bounds.y + bounds.height;
  return `M ${bounds.x} ${bounds.y} H ${right} V ${bottom} H ${bounds.x} Z`;
}

function continuationPath(bounds: Bounds): string {
  const middleY = bounds.y + bounds.height / 2;
  const bottom = bounds.y + bounds.height;
  const inset = Math.min(4, bounds.width);
  return `M ${bounds.x} ${bounds.y} L ${bounds.x + inset} ${middleY} L ${bounds.x} ${bottom}`;
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

function describeSegment(
  segment: SegmentStripInput,
  clip: boolean,
  continuation: boolean,
): string {
  const parts = [
    `${segment.tokens} ${segment.tokens === 1 ? "token" : "tokens"}`,
  ];
  if (clip) {
    parts.push("truncated");
  }
  if (continuation) {
    parts.push("continuation");
  }
  return parts.join("; ");
}

/** Emits deterministic display, semantic, and interaction fragments for a segment strip. */
export function contributeSegmentStrip(
  input: SegmentStripContributionInput,
): SegmentStripContribution {
  const layout = layoutSegmentStrip(input.segments, input.layout);
  const order = input.order ?? 0;
  const fill = input.style?.fill ?? "#334155";
  const textFill = input.style?.textFill ?? "#f8fafc";
  const continuationStroke =
    input.style?.continuationStroke ?? "#38bdf8";
  const font = input.style?.font ?? DEFAULT_FONT;
  const commands: DrawCommand[] = [];
  const semanticEntities: SemanticEntityProjection[] = [];
  const hitRegions: HitRegion[] = [];

  layout.nodes.forEach((node, index) => {
    const segment = input.segments[index]!;
    const bounds = { ...node.bounds };
    const segmentOrder = order + index;
    const children: DrawCommand[] = [
      {
        kind: "path",
        id: `${input.id}:${segment.id}:rect`,
        order: 0,
        paintBounds: bounds,
        damageBounds: bounds,
        path: rectanglePath(bounds),
        fill,
      },
      {
        kind: "text",
        id: `${input.id}:${segment.id}:label`,
        order: 1,
        paintBounds: bounds,
        damageBounds: bounds,
        text: segment.role,
        origin: {
          x: bounds.x + 4,
          y: bounds.y + (bounds.height * 2) / 3,
        },
        font: { ...font },
        fill: textFill,
      },
    ];

    if (node.continuation === true) {
      const continuation: PathDrawCommand = {
        kind: "path",
        id: `${input.id}:${segment.id}:continuation`,
        order: 2,
        paintBounds: bounds,
        damageBounds: bounds,
        path: continuationPath(bounds),
        stroke: continuationStroke,
        strokeWidth: 2,
      };
      children.push(continuation);
    }

    const commandBase = {
      id: `${input.id}:${segment.id}`,
      order: segmentOrder,
      paintBounds: bounds,
      damageBounds: bounds,
      children,
    } as const;
    commands.push(
      node.clip === true
        ? { kind: "clip", ...commandBase, path: rectanglePath(bounds) }
        : { kind: "group", ...commandBase },
    );
    semanticEntities.push({
      id: segment.id,
      label: segment.role,
      role: segment.role,
      kind: "segment",
      description: describeSegment(
        segment,
        node.clip === true,
        node.continuation === true,
      ),
    });
    hitRegions.push({
      id: `${input.id}:${segment.id}:hit`,
      semanticId: segment.id,
      order: segmentOrder,
      bounds,
    });
  });

  return deepFreeze({ commands, semanticEntities, hitRegions });
}
