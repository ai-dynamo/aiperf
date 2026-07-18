// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Bounds, DisplayList } from "../../display-list.js";
import type { CanvasQualityMode } from "./quality.js";
import { CanvasTextAtlas } from "./text-atlas.js";

/** User Timing measure names collected by Playwright runtime metrics. */
export const RUNTIME_PERFORMANCE_ENTRY_NAMES = Object.freeze({
  evaluation: "aiperf-flow:evaluation",
  draw: "aiperf-flow:draw",
  total: "aiperf-flow:total",
});

let runtimeMeasureSequence = 0;

function canRecordRuntimeMeasures(): boolean {
  return (
    typeof performance !== "undefined" &&
    typeof performance.mark === "function" &&
    typeof performance.measure === "function" &&
    typeof performance.clearMarks === "function"
  );
}

/**
 * Records one real wall-clock phase as a User Timing measure when available.
 * Falls through unchanged when the Performance API is absent.
 */
export function measureRuntimePhase<T>(
  phase: keyof typeof RUNTIME_PERFORMANCE_ENTRY_NAMES,
  work: () => T,
): T {
  if (!canRecordRuntimeMeasures()) {
    return work();
  }
  const name = RUNTIME_PERFORMANCE_ENTRY_NAMES[phase];
  const token = `${name}:${runtimeMeasureSequence++}`;
  const startMark = `${token}:start`;
  const endMark = `${token}:end`;
  performance.mark(startMark);
  try {
    return work();
  } finally {
    performance.mark(endMark);
    try {
      performance.measure(name, startMark, endMark);
    } finally {
      performance.clearMarks(startMark);
      performance.clearMarks(endMark);
    }
  }
}

type CanvasMethod =
  | "beginPath"
  | "clip"
  | "closePath"
  | "fill"
  | "fillRect"
  | "fillText"
  | "lineTo"
  | "measureText"
  | "moveTo"
  | "restore"
  | "save"
  | "scale"
  | "stroke";

export type CanvasRenderContext = Pick<CanvasRenderingContext2D, CanvasMethod> &
  Pick<
    CanvasRenderingContext2D,
    | "fillStyle"
    | "font"
    | "globalAlpha"
    | "globalCompositeOperation"
    | "lineWidth"
    | "strokeStyle"
  >;

export type CanvasRenderOptions = Readonly<{
  devicePixelRatio?: number;
  textAtlas?: CanvasTextAtlas;
}>;

export type CanvasRenderMetrics = Readonly<{
  commandCount: number;
}>;

export type CanvasSemanticHitRegion = Readonly<{
  entityId: string;
  label: string;
  focusTarget: string;
  focusable: boolean;
  selected: boolean;
  bounds: Bounds;
}>;

export type CanvasDisplayListOutput = Readonly<{
  hitRegions: readonly CanvasSemanticHitRegion[];
}>;

export type CanvasDisplayListOptions = Readonly<{
  devicePixelRatio?: number | undefined;
  quality?: CanvasQualityMode | undefined;
}>;

type UnknownRecord = Readonly<Record<string, unknown>>;
type Point = Readonly<{ x: number; y: number }>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function number(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function point(value: unknown): Point {
  const properties = record(value);
  return { x: number(properties.x), y: number(properties.y) };
}

function children(command: UnknownRecord): readonly unknown[] {
  return Array.isArray(command.children) ? command.children : [];
}

function applyPaint(
  context: CanvasRenderContext,
  command: UnknownRecord,
): void {
  if (typeof command.fill === "string") {
    context.fillStyle = command.fill;
  }
  if (typeof command.stroke === "string") {
    context.strokeStyle = command.stroke;
  }
  if (typeof command.lineWidth === "number") {
    context.lineWidth = command.lineWidth;
  }
}

function drawPath(context: CanvasRenderContext, path: string): void {
  const tokens = path.match(/[MLHVZ]|-?(?:\d+\.?\d*|\.\d+)/gi) ?? [];
  let cursor = { x: 0, y: 0 };
  let index = 0;

  context.beginPath();
  while (index < tokens.length) {
    const operation = tokens[index++]?.toUpperCase();
    if (operation === "M" || operation === "L") {
      cursor = {
        x: Number(tokens[index++]),
        y: Number(tokens[index++]),
      };
      if (operation === "M") {
        context.moveTo(cursor.x, cursor.y);
      } else {
        context.lineTo(cursor.x, cursor.y);
      }
    } else if (operation === "H") {
      cursor = { x: Number(tokens[index++]), y: cursor.y };
      context.lineTo(cursor.x, cursor.y);
    } else if (operation === "V") {
      cursor = { x: cursor.x, y: Number(tokens[index++]) };
      context.lineTo(cursor.x, cursor.y);
    } else if (operation === "Z") {
      context.closePath();
    } else {
      throw new Error(`Unsupported display-list path operation "${operation}".`);
    }
  }
}

function drawCommand(
  context: CanvasRenderContext,
  textAtlas: CanvasTextAtlas,
  value: unknown,
): number {
  const command = record(value);
  applyPaint(context, command);

  switch (command.kind) {
    case "rect": {
      const bounds = record(command.bounds);
      context.fillRect(
        number(bounds.x),
        number(bounds.y),
        number(bounds.width),
        number(bounds.height),
      );
      return 1;
    }
    case "text": {
      const origin = point(command.origin);
      const font = record(command.font);
      const family = asString(font.family);
      const sizePx = number(font.sizePx);
      if (family !== "" && sizePx > 0) {
        textAtlas.measure(asString(command.text), {
          family,
          sizePx,
          ...(typeof font.weight === "number" ? { weight: font.weight } : {}),
        });
      }
      context.fillText(asString(command.text), origin.x, origin.y);
      return 1;
    }
    case "line": {
      const from = point(command.from);
      const to = point(command.to);
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      context.stroke();
      return 1;
    }
    case "path":
      drawPath(context, asString(command.path));
      if (typeof command.fill === "string") {
        context.fill();
      }
      if (typeof command.stroke === "string" || command.fill === undefined) {
        context.stroke();
      }
      return 1;
    case "group":
      return children(command).reduce<number>(
        (total, child) => total + drawCommand(context, textAtlas, child),
        0,
      );
    case "clip": {
      context.save();
      drawPath(context, asString(command.path));
      context.clip();
      const count = children(command).reduce<number>(
        (total, child) => total + drawCommand(context, textAtlas, child),
        0,
      );
      context.restore();
      return count;
    }
    case "layer": {
      context.save();
      context.globalAlpha = number(command.opacity);
      if (typeof command.blendMode === "string") {
        context.globalCompositeOperation =
          command.blendMode as GlobalCompositeOperation;
      }
      const count = children(command).reduce<number>(
        (total, child) => total + drawCommand(context, textAtlas, child),
        0,
      );
      context.restore();
      return count;
    }
    default:
      throw new Error(`Unsupported display-list command "${asString(command.kind)}".`);
  }
}

function normalizeHitRegions(
  displayList: DisplayList,
): readonly CanvasSemanticHitRegion[] {
  return displayList.hitRegions.map((region) => {
    const loose = region as UnknownRecord & {
      bounds: Bounds;
      semanticId: string;
    };
    const entityId =
      typeof loose.entityId === "string" && loose.entityId.length > 0
        ? loose.entityId
        : region.semanticId;
    return {
      entityId,
      label: asString(loose.label, entityId),
      focusTarget: asString(loose.focusTarget, entityId),
      focusable: loose.focusable !== false,
      selected: loose.selected === true,
      bounds: region.bounds,
    };
  });
}

/** Renders an already evaluated display list in logical scene coordinates. */
export function renderDisplayList(
  context: CanvasRenderContext,
  displayList: DisplayList,
  options: CanvasRenderOptions = {},
): CanvasRenderMetrics {
  return measureRuntimePhase("total", () => {
    const ratio = options.devicePixelRatio ?? 1;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      throw new RangeError("devicePixelRatio must be a positive finite number.");
    }
    const textAtlas = options.textAtlas ?? new CanvasTextAtlas(context);

    return measureRuntimePhase("draw", () => {
      if (ratio !== 1) {
        context.save();
        context.scale(ratio, ratio);
      }

      const commandCount = displayList.commands.reduce(
        (count, command) =>
          count + drawCommand(context, textAtlas, command),
        0,
      );

      if (ratio !== 1) {
        context.restore();
      }
      return { commandCount };
    });
  });
}

/** Renders a display list and returns semantic hit metadata for conformance checks. */
export function renderCanvasDisplayList(
  displayList: DisplayList,
  context: CanvasRenderContext,
  options: CanvasDisplayListOptions = {},
): CanvasDisplayListOutput {
  renderDisplayList(
    context,
    displayList,
    options.devicePixelRatio === undefined
      ? {}
      : { devicePixelRatio: options.devicePixelRatio },
  );
  return { hitRegions: normalizeHitRegions(displayList) };
}
