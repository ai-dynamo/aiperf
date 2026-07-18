// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Bounds,
  DisplayList,
  DrawCommand,
} from "../display-list.js";

function compareBounds(left: Bounds, right: Bounds): number {
  return (
    left.x - right.x ||
    left.y - right.y ||
    left.width - right.width ||
    left.height - right.height
  );
}

function intersectsOrTouches(left: Bounds, right: Bounds): boolean {
  return (
    left.x <= right.x + right.width &&
    right.x <= left.x + left.width &&
    left.y <= right.y + right.height &&
    right.y <= left.y + left.height
  );
}

function unionBounds(left: Bounds, right: Bounds): Bounds {
  const x = Math.min(left.x, right.x);
  const y = Math.min(left.y, right.y);
  return {
    x,
    y,
    width: Math.max(left.x + left.width, right.x + right.width) - x,
    height: Math.max(left.y + left.height, right.y + right.height) - y,
  };
}

/**
 * Coalesces intersecting damage bounds into deterministic axis-aligned
 * supersets while preserving gaps between independent regions.
 */
export function mergeDamageRegions(
  regions: readonly Bounds[],
): readonly Bounds[] {
  const merged = regions
    .filter(({ width, height }) => width > 0 && height > 0)
    .map((bounds) => ({ ...bounds }))
    .sort(compareBounds);

  for (let index = 0; index < merged.length; index += 1) {
    let candidate = merged[index]!;
    let changed = true;

    while (changed) {
      changed = false;
      for (let otherIndex = index + 1; otherIndex < merged.length; ) {
        const other = merged[otherIndex]!;
        if (!intersectsOrTouches(candidate, other)) {
          otherIndex += 1;
          continue;
        }

        candidate = unionBounds(candidate, other);
        merged[index] = candidate;
        merged.splice(otherIndex, 1);
        changed = true;
      }
    }
  }

  merged.sort(compareBounds);
  for (const bounds of merged) {
    Object.freeze(bounds);
  }
  return Object.freeze(merged);
}

function stableSerialize(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableSerialize).join(",")}]`;
  }

  const record = value as Readonly<Record<string, unknown>>;
  return `{${Object.keys(record)
    .filter((key) => record[key] !== undefined)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableSerialize(record[key])}`)
    .join(",")}}`;
}

type CommandSnapshot = Readonly<{
  damageBounds: Bounds;
  visualState: string;
}>;

function snapshotCommands(
  commands: readonly DrawCommand[],
): ReadonlyMap<string, CommandSnapshot> {
  const snapshots = new Map<string, CommandSnapshot>();
  const occurrences = new Map<string, number>();

  const visit = (command: DrawCommand): void => {
    const occurrence = occurrences.get(command.id) ?? 0;
    occurrences.set(command.id, occurrence + 1);

    const visualState = Object.fromEntries(
      Object.entries(command).filter(
        ([key]) => key !== "children" && key !== "source",
      ),
    );
    snapshots.set(`${command.id}\u0000${occurrence}`, {
      damageBounds: command.damageBounds,
      visualState: stableSerialize(visualState),
    });

    if (
      command.kind === "group" ||
      command.kind === "clip" ||
      command.kind === "layer"
    ) {
      command.children.forEach(visit);
    }
  };

  commands.forEach(visit);
  return snapshots;
}

/**
 * Computes the smallest command-derived damage supersets between two frames.
 *
 * Nested commands are compared independently so removing a decorative child
 * does not invalidate an unchanged semantic container.
 */
export function computeDamageBetween(
  previous: DisplayList,
  current: DisplayList,
): readonly Bounds[] {
  const previousCommands = snapshotCommands(previous.commands);
  const currentCommands = snapshotCommands(current.commands);
  const keys = [...new Set([...previousCommands.keys(), ...currentCommands.keys()])]
    .sort();
  const damage: Bounds[] = [];

  for (const key of keys) {
    const before = previousCommands.get(key);
    const after = currentCommands.get(key);
    if (before?.visualState === after?.visualState) {
      continue;
    }
    if (before !== undefined) {
      damage.push(before.damageBounds);
    }
    if (after !== undefined) {
      damage.push(after.damageBounds);
    }
  }

  return mergeDamageRegions(damage);
}
