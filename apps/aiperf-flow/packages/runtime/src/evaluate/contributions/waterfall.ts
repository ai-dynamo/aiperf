// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Bounds,
  DrawCommand,
  HitRegion,
} from "../../display-list.js";
import {
  layoutWaterfallNest,
  type WaterfallEvent,
  type WaterfallLayoutOptions,
} from "../../leaves/waterfall-nest-layout.js";
import type {
  SemanticEntityProjection,
  SemanticRelationProjection,
} from "../types.js";

export type WaterfallContributionEvent = WaterfallEvent &
  Readonly<{
    label?: string;
    open?: boolean;
  }>;

export type WaterfallContributionStyle = Readonly<{
  pointFill?: string;
  intervalFill?: string;
  textFill?: string;
  playheadStroke?: string;
  labelOffsetX?: number;
  labelWidth?: number;
  font?: Readonly<{
    family: string;
    sizePx: number;
    weight?: number;
  }>;
}>;

export type WaterfallContributionInput = Readonly<{
  id: string;
  events: readonly WaterfallContributionEvent[];
  layout: WaterfallLayoutOptions;
  atMs: number;
  order?: number;
  reducedMotion?: boolean;
  style?: WaterfallContributionStyle;
}>;

export type WaterfallContribution = Readonly<{
  commands: readonly DrawCommand[];
  semanticEntities: readonly SemanticEntityProjection[];
  semanticRelations: readonly SemanticRelationProjection[];
  hitRegions: readonly HitRegion[];
}>;

const DEFAULT_FONT = { family: "sans-serif", sizePx: 12 } as const;

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) {
      deepFreeze(nested);
    }
    Object.freeze(value);
  }
  return value;
}

function rectanglePath(bounds: Bounds): string {
  return `M ${bounds.x} ${bounds.y} H ${bounds.x + bounds.width} V ${bounds.y + bounds.height} H ${bounds.x} Z`;
}

function assertFiniteNonNegative(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be finite and non-negative.`);
  }
}

function assertInput(input: WaterfallContributionInput): void {
  if (!Number.isSafeInteger(input.atMs) || input.atMs < 0) {
    throw new RangeError(
      "Waterfall evaluation time must be a non-negative safe integer.",
    );
  }
  if (!Number.isSafeInteger(input.order ?? 0)) {
    throw new RangeError("Waterfall order must be a safe integer.");
  }

  const { layout } = input;
  try {
    assertFiniteNonNegative(layout.originX, "originX");
    assertFiniteNonNegative(layout.originY, "originY");
    assertFiniteNonNegative(layout.laneHeight, "laneHeight");
    assertFiniteNonNegative(layout.laneGap, "laneGap");
    assertFiniteNonNegative(layout.pxPerMs, "pxPerMs");
  } catch {
    throw new RangeError(
      "Waterfall layout values must be finite and non-negative.",
    );
  }

  for (const event of input.events) {
    if (!Number.isFinite(event.start) || !Number.isFinite(event.end)) {
      throw new RangeError(
        `Waterfall event "${event.id}" times must be finite.`,
      );
    }
  }
}

function eventKind(
  event: WaterfallContributionEvent,
): "point" | "interval" | "open-interval" {
  if (event.open === true) {
    return "open-interval";
  }
  return event.start === event.end ? "point" : "interval";
}

function effectiveEvent(
  event: WaterfallContributionEvent,
  atMs: number,
): WaterfallEvent | undefined {
  if (event.start > atMs) {
    return undefined;
  }
  if (event.open === true) {
    return {
      id: event.id,
      lane: event.lane,
      start: event.start,
      end: Math.max(event.end, atMs),
    };
  }
  return {
    id: event.id,
    lane: event.lane,
    start: event.start,
    end: event.end,
  };
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

/**
 * Projects nested waterfall lane geometry into immutable backend-neutral
 * display, semantic, and hit-region products at one authored integer time.
 */
export function contributeWaterfall(
  input: WaterfallContributionInput,
): WaterfallContribution {
  assertInput(input);

  const visibleEvents = input.events.flatMap((event) => {
    const effective = effectiveEvent(event, input.atMs);
    return effective === undefined ? [] : [{ event, effective }];
  });

  const layout = layoutWaterfallNest(
    visibleEvents.map(({ effective }) => effective),
    input.layout,
  );
  const boundsById = new Map(
    layout.nodes.map((node) => [node.nodeId, { ...node.bounds }] as const),
  );

  const order = input.order ?? 0;
  const pointFill = input.style?.pointFill ?? "#7dcfff";
  const intervalFill = input.style?.intervalFill ?? "#38bdf8";
  const textFill = input.style?.textFill ?? "#f8fafc";
  const playheadStroke = input.style?.playheadStroke ?? "#fbbf24";
  const labelOffsetX = input.style?.labelOffsetX ?? 40;
  const labelWidth = input.style?.labelWidth ?? 36;
  const font = input.style?.font ?? DEFAULT_FONT;
  const reducedMotion = input.reducedMotion === true;

  const commands: DrawCommand[] = [];
  const semanticEntities: SemanticEntityProjection[] = [];
  const semanticRelations: SemanticRelationProjection[] = [];
  const hitRegions: HitRegion[] = [];

  const eventsByLane = new Map<string, typeof visibleEvents>();
  for (const entry of visibleEvents) {
    const bucket = eventsByLane.get(entry.event.lane) ?? [];
    bucket.push(entry);
    eventsByLane.set(entry.event.lane, bucket);
  }

  input.layout.laneOrder.forEach((lane, laneIndex) => {
    const laneEvents = eventsByLane.get(lane) ?? [];
    if (laneEvents.length === 0) {
      return;
    }

    const laneOrder = order + laneIndex;
    const laneSemanticId = `${input.id}:lane:${lane}`;
    const laneY =
      input.layout.originY +
      laneIndex * (input.layout.laneHeight + input.layout.laneGap);
    const labelBounds: Bounds = {
      x: input.layout.originX - labelOffsetX,
      y: laneY,
      width: labelWidth,
      height: input.layout.laneHeight,
    };
    const children: DrawCommand[] = [
      {
        kind: "text",
        id: `${laneSemanticId}:label`,
        order: 0,
        paintBounds: labelBounds,
        damageBounds: labelBounds,
        text: lane,
        origin: {
          x: labelBounds.x,
          y: labelBounds.y + (input.layout.laneHeight * 3) / 4,
        },
        font: { ...font },
        fill: textFill,
      },
    ];

    semanticEntities.push({
      id: laneSemanticId,
      label: lane,
      role: "row",
      kind: "lane",
    });

    const eventBounds: Bounds[] = [];
    for (const [eventIndex, { event }] of laneEvents.entries()) {
      const bounds = boundsById.get(event.id);
      if (bounds === undefined) {
        continue;
      }
      eventBounds.push(bounds);
      const kind = eventKind(event);
      const fill =
        kind === "interval" || kind === "open-interval"
          ? intervalFill
          : pointFill;
      children.push({
        kind: "path",
        id: `${input.id}:${event.id}`,
        order: eventIndex + 1,
        paintBounds: bounds,
        damageBounds: bounds,
        path: rectanglePath(bounds),
        fill,
      });
      semanticEntities.push({
        id: event.id,
        label: event.label ?? event.id,
        role: "listitem",
        kind,
        description:
          kind === "point"
            ? `Point event on lane ${lane}`
            : kind === "open-interval"
              ? `Open interval event on lane ${lane}`
              : `Interval event on lane ${lane}`,
      });
      semanticRelations.push({
        id: `${input.id}:rel:${event.id}`,
        fromId: laneSemanticId,
        toId: event.id,
        label: "contains",
        role: "contains",
      });
      hitRegions.push({
        id: `${input.id}:${event.id}:hit`,
        semanticId: event.id,
        order: laneOrder,
        bounds,
      });
    }

    const paintBounds = unionBounds(eventBounds);
    commands.push({
      kind: "group",
      id: laneSemanticId,
      order: laneOrder,
      paintBounds,
      damageBounds: paintBounds,
      children,
    });
  });

  if (!reducedMotion) {
    const playheadX =
      input.layout.originX + input.atMs * input.layout.pxPerMs;
    const laneCount = input.layout.laneOrder.length;
    const height =
      laneCount === 0
        ? 0
        : laneCount * input.layout.laneHeight +
          Math.max(0, laneCount - 1) * input.layout.laneGap;
    const playheadBounds: Bounds = {
      x: playheadX,
      y: input.layout.originY,
      width: 0,
      height,
    };
    commands.push({
      kind: "path",
      id: `${input.id}:playhead`,
      order: order + input.layout.laneOrder.length,
      paintBounds: playheadBounds,
      damageBounds: {
        x: playheadX - 1,
        y: input.layout.originY,
        width: 2,
        height,
      },
      path: `M ${playheadX} ${input.layout.originY} V ${input.layout.originY + height}`,
      stroke: playheadStroke,
      strokeWidth: 1,
    });
  }

  return deepFreeze({
    commands,
    semanticEntities,
    semanticRelations,
    hitRegions,
  });
}
