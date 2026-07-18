// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Axis-aligned bounds in logical scene coordinates. */
export type Bounds = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/** A point in logical scene coordinates. */
export type Point = Readonly<{
  x: number;
  y: number;
}>;

/** Serializable provenance for evaluated scene output. */
export type SourceReference = Readonly<{
  source: string;
  startOffset: number;
  endOffset: number;
}>;

type DrawCommandBase = Readonly<{
  id: string;
  order: number;
  paintBounds: Bounds;
  damageBounds: Bounds;
  source?: SourceReference;
}>;

export type GroupDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "group";
    children: readonly DrawCommand[];
  }>;

export type PathDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "path";
    path: string;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
  }>;

export type TextDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "text";
    text: string;
    origin: Point;
    font: Readonly<{
      family: string;
      sizePx: number;
      weight?: number;
    }>;
    fill?: string;
  }>;

export type ImageDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "image";
    assetId: string;
    destination: Bounds;
    opacity?: number;
  }>;

export type ClipDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "clip";
    path: string;
    children: readonly DrawCommand[];
  }>;

export type LayerDrawCommand = DrawCommandBase &
  Readonly<{
    kind: "layer";
    children: readonly DrawCommand[];
    opacity?: number;
    blendMode?:
      | "normal"
      | "multiply"
      | "screen"
      | "overlay"
      | "darken"
      | "lighten";
  }>;

/** A backend-neutral, serializable visual operation. */
export type DrawCommand =
  | GroupDrawCommand
  | PathDrawCommand
  | TextDrawCommand
  | ImageDrawCommand
  | ClipDrawCommand
  | LayerDrawCommand;

/** A semantic interaction target independent of rendered pixels. */
export type HitRegion = Readonly<{
  id: string;
  semanticId: string;
  order: number;
  bounds: Bounds;
  source?: SourceReference;
}>;

/** An immutable, deterministically ordered visual frame. */
export type DisplayList = Readonly<{
  commands: readonly DrawCommand[];
  hitRegions: readonly HitRegion[];
  paintBounds: Bounds;
  damageBounds: Bounds;
}>;

/** Input accepted by {@link buildDisplayList} before ordering and freezing. */
export type DisplayListInput = Readonly<{
  commands: readonly DrawCommand[];
  hitRegions: readonly HitRegion[];
  paintBounds: Bounds;
  damageBounds: Bounds;
}>;

function assertFiniteBounds(bounds: Bounds, location: string): void {
  if (
    !Number.isFinite(bounds.x) ||
    !Number.isFinite(bounds.y) ||
    !Number.isFinite(bounds.width) ||
    !Number.isFinite(bounds.height) ||
    bounds.width < 0 ||
    bounds.height < 0
  ) {
    throw new RangeError(`${location} must have finite bounds`);
  }
}

function compareOrdered(
  left: Readonly<{ id: string; order: number }>,
  right: Readonly<{ id: string; order: number }>,
): number {
  return left.order - right.order || left.id.localeCompare(right.id);
}

function normalizeCommand(command: DrawCommand): DrawCommand {
  assertFiniteBounds(command.paintBounds, `command ${command.id} paintBounds`);
  assertFiniteBounds(command.damageBounds, `command ${command.id} damageBounds`);
  if (!Number.isSafeInteger(command.order)) {
    throw new RangeError(`command ${command.id} order must be a safe integer`);
  }

  if (
    command.kind === "group" ||
    command.kind === "clip" ||
    command.kind === "layer"
  ) {
    return {
      ...command,
      children: command.children.map(normalizeCommand).sort(compareOrdered),
    };
  }
  return { ...command };
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
 * Validates frame bounds, applies canonical draw ordering, and freezes the
 * resulting serializable display list.
 */
export function buildDisplayList(input: DisplayListInput): DisplayList {
  assertFiniteBounds(input.paintBounds, "display-list paintBounds");
  assertFiniteBounds(input.damageBounds, "display-list damageBounds");

  const hitRegions = input.hitRegions.map((region) => {
    assertFiniteBounds(region.bounds, `hit region ${region.id}`);
    if (!Number.isSafeInteger(region.order)) {
      throw new RangeError(
        `hit region ${region.id} order must be a safe integer`,
      );
    }
    return { ...region };
  });

  return deepFreeze({
    commands: input.commands.map(normalizeCommand).sort(compareOrdered),
    hitRegions: hitRegions.sort(compareOrdered),
    paintBounds: { ...input.paintBounds },
    damageBounds: { ...input.damageBounds },
  });
}
