// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure quality-policy applier for evaluated display lists.
//!
//! Degraded tiers may suppress decorative draw work only. Required-semantic
//! commands and select/inspect/focus hit regions are never removed. Reduced
//! motion zeroes motion metadata without dropping semantic content.

import {
  buildDisplayList,
  type DisplayList,
  type DrawCommand,
  type HitRegion,
} from "../display-list.js";
import {
  qualityProfile,
  type DecorativeQuality,
  type QualityProfile,
  type QualityTier,
} from "../quality-profiles.js";

/** Whether a draw command is required for semantics or optional decoration. */
export type CommandQualityClass = "required-semantic" | "decorative";

/** Decorative effect families that quality tiers may suppress. */
export type DecorativeFamily = "particles" | "blur" | "shadow" | "glow";

/** Interaction roles used for hit-region preservation policy. */
export type HitRegionRole =
  | "select"
  | "inspect"
  | "scrub"
  | "focus"
  | "compare"
  | "navigate";

/** Motion metadata that reduced-motion profiles zero without dropping commands. */
export type CommandMotionMetadata = Readonly<{
  progress: number;
  pathId?: string | undefined;
}>;

/** Orthogonal accessibility / fidelity axes applied with a quality tier. */
export type QualityPolicyAxes = Readonly<{
  motion: "full" | "reduced";
  contrast: "standard" | "high";
  depth: "full" | "none";
}>;

/**
 * Runtime quality profile.
 *
 * Reuses the Canvas decorative contract and adds motion/contrast/depth axes
 * that the planned schema `QualityProfileIr` will mirror.
 */
export type QualityPolicyProfile = QualityProfile & QualityPolicyAxes;

/**
 * Optional capability display-contract budget hints.
 *
 * Schema promotion of `displayContract` is out of scope for this module; the
 * applier accepts a minimal local shape so callers can pass descriptor budgets
 * without editing shared schema files.
 */
export type DisplayContract = Readonly<{
  maxDecorativeCommands?: number;
  supportedDecorativeFamilies?: readonly DecorativeFamily[];
}>;

/** Draw command carrying quality annotations used by the applier. */
export type QualityAnnotatedCommand = DrawCommand &
  Readonly<{
    qualityClass?: CommandQualityClass;
    decorativeFamily?: DecorativeFamily;
    semanticEntityId?: string;
    narrationCueMarker?: boolean;
    motion?: CommandMotionMetadata;
  }>;

/** Hit region carrying role and quality annotations. */
export type QualityAnnotatedHitRegion = HitRegion &
  Readonly<{
    role?: HitRegionRole;
    qualityClass?: CommandQualityClass;
    decorativeFamily?: DecorativeFamily;
  }>;

/** Display list whose commands and hit regions may carry quality annotations. */
export type QualityDisplayList = Readonly<{
  commands: readonly QualityAnnotatedCommand[];
  hitRegions: readonly QualityAnnotatedHitRegion[];
  paintBounds: DisplayList["paintBounds"];
  damageBounds: DisplayList["damageBounds"];
}>;

/** Diagnostic report of decorative work suppressed by a quality profile. */
export type DegradationReport = Readonly<{
  tier: QualityTier;
  motionReduced: boolean;
  /**
   * Pre-order indices into the *original* (pre-policy) command tree.
   * Filter and decorative-budget suppressions share this single index space.
   */
  suppressedCommandIndices: readonly number[];
  suppressedFamilies: readonly DecorativeFamily[];
  suppressedHitRegionIds: readonly string[];
}>;

export type QualityPolicyResult = Readonly<{
  list: QualityDisplayList;
  report: DegradationReport;
}>;

const PROTECTED_HIT_ROLES: ReadonlySet<HitRegionRole> = new Set([
  "select",
  "inspect",
  "focus",
]);

const FAMILY_ORDER: readonly DecorativeFamily[] = [
  "blur",
  "glow",
  "particles",
  "shadow",
];

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) {
      deepFreeze(nested);
    }
    Object.freeze(value);
  }
  return value;
}

function isDecorativeFamily(value: unknown): value is DecorativeFamily {
  return (
    value === "particles" ||
    value === "blur" ||
    value === "shadow" ||
    value === "glow"
  );
}

/**
 * Builds a quality-policy profile from the shared Canvas tier lookup plus
 * optional motion/contrast/depth axes.
 */
export function qualityPolicyProfile(
  tier: QualityTier,
  axes: Partial<QualityPolicyAxes> = {},
): QualityPolicyProfile {
  const base = qualityProfile(tier);
  return deepFreeze({
    tier: base.tier,
    decorative: { ...base.decorative },
    motion: axes.motion ?? "full",
    contrast: axes.contrast ?? "standard",
    depth: axes.depth ?? "full",
  });
}

function familyAllowed(
  family: DecorativeFamily,
  decorative: DecorativeQuality,
  depth: QualityPolicyAxes["depth"],
): boolean {
  if (depth === "none" && (family === "shadow" || family === "glow")) {
    return false;
  }
  switch (family) {
    case "particles":
      return decorative.particles;
    case "blur":
      return decorative.blur;
    case "glow":
      return decorative.glow;
    case "shadow":
      // Canvas profiles gate shadow with glow; degraded turns both off.
      return decorative.glow;
  }
}

function commandQualityClass(
  command: QualityAnnotatedCommand,
): CommandQualityClass {
  if (command.narrationCueMarker === true) {
    return "required-semantic";
  }
  return command.qualityClass ?? "required-semantic";
}

function shouldKeepCommand(
  command: QualityAnnotatedCommand,
  profile: QualityPolicyProfile,
  displayContract: DisplayContract | undefined,
): boolean {
  if (commandQualityClass(command) === "required-semantic") {
    return true;
  }

  const family = command.decorativeFamily;
  if (family !== undefined) {
    if (
      displayContract?.supportedDecorativeFamilies !== undefined &&
      !displayContract.supportedDecorativeFamilies.includes(family)
    ) {
      return false;
    }
    return familyAllowed(family, profile.decorative, profile.depth);
  }

  // Untagged decorative commands are suppressed only on degraded tiers.
  return profile.tier === "reference";
}

function zeroMotion(
  command: QualityAnnotatedCommand,
): QualityAnnotatedCommand {
  if (command.motion === undefined) {
    return command;
  }
  return {
    ...command,
    motion: { progress: 0, pathId: undefined },
  };
}

type FilterCommandsResult = Readonly<{
  commands: readonly QualityAnnotatedCommand[];
  suppressedIndices: readonly number[];
  suppressedFamilies: ReadonlySet<DecorativeFamily>;
  nextIndex: number;
  /**
   * Original-tree pre-order indices for every node that remains in the
   * filtered command tree (same pre-order walk `enforceDecorativeBudget` uses).
   */
  keptOriginalIndices: readonly number[];
}>;

function filterCommands(
  commands: readonly QualityAnnotatedCommand[],
  profile: QualityPolicyProfile,
  displayContract: DisplayContract | undefined,
  startIndex: number,
): FilterCommandsResult {
  const kept: QualityAnnotatedCommand[] = [];
  const suppressedIndices: number[] = [];
  const suppressedFamilies = new Set<DecorativeFamily>();
  const keptOriginalIndices: number[] = [];
  let index = startIndex;

  for (const command of commands) {
    const commandIndex = index;
    index += 1;

    let nextCommand = command;
    let nestedKeptOriginalIndices: readonly number[] = [];
    if (
      command.kind === "group" ||
      command.kind === "clip" ||
      command.kind === "layer"
    ) {
      const nested = filterCommands(
        command.children as readonly QualityAnnotatedCommand[],
        profile,
        displayContract,
        index,
      );
      index = nested.nextIndex;
      nestedKeptOriginalIndices = nested.keptOriginalIndices;
      for (const suppressed of nested.suppressedIndices) {
        suppressedIndices.push(suppressed);
      }
      for (const family of nested.suppressedFamilies) {
        suppressedFamilies.add(family);
      }
      nextCommand = {
        ...command,
        children: nested.commands,
      } as QualityAnnotatedCommand;
    }

    if (!shouldKeepCommand(nextCommand, profile, displayContract)) {
      suppressedIndices.push(commandIndex);
      if (isDecorativeFamily(nextCommand.decorativeFamily)) {
        suppressedFamilies.add(nextCommand.decorativeFamily);
      }

      if (
        (nextCommand.kind === "group" ||
          nextCommand.kind === "clip" ||
          nextCommand.kind === "layer") &&
        nextCommand.children.length > 0
      ) {
        // Hoisted children keep their original indices; the container does not.
        for (const originalIndex of nestedKeptOriginalIndices) {
          keptOriginalIndices.push(originalIndex);
        }
        for (const child of nextCommand.children as readonly QualityAnnotatedCommand[]) {
          kept.push(
            profile.motion === "reduced" ? zeroMotion(child) : child,
          );
        }
      }
      continue;
    }

    if (profile.motion === "reduced") {
      nextCommand = zeroMotion(nextCommand);
    }

    keptOriginalIndices.push(commandIndex);
    for (const originalIndex of nestedKeptOriginalIndices) {
      keptOriginalIndices.push(originalIndex);
    }
    kept.push(nextCommand);
  }

  return {
    commands: kept,
    suppressedIndices,
    suppressedFamilies,
    nextIndex: index,
    keptOriginalIndices,
  };
}

function shouldKeepHitRegion(
  region: QualityAnnotatedHitRegion,
  profile: QualityPolicyProfile,
): boolean {
  if (region.role !== undefined && PROTECTED_HIT_ROLES.has(region.role)) {
    return true;
  }
  if (region.qualityClass === "required-semantic") {
    return true;
  }
  if (region.qualityClass !== "decorative") {
    return true;
  }
  if (region.decorativeFamily !== undefined) {
    return familyAllowed(
      region.decorativeFamily,
      profile.decorative,
      profile.depth,
    );
  }
  return profile.tier === "reference";
}

function enforceDecorativeBudget(
  commands: readonly QualityAnnotatedCommand[],
  maxDecorativeCommands: number,
  originalIndices: readonly number[],
  walkStart: number,
): FilterCommandsResult {
  const kept: QualityAnnotatedCommand[] = [];
  const suppressedIndices: number[] = [];
  const suppressedFamilies = new Set<DecorativeFamily>();
  const keptOriginalIndices: number[] = [];
  let decorativeCount = 0;
  let walk = walkStart;

  for (const command of commands) {
    const commandIndex = originalIndices[walk];
    if (commandIndex === undefined) {
      throw new Error(
        `Decorative budget walk is missing original index at offset ${walk}.`,
      );
    }
    walk += 1;

    let nextCommand = command;
    let nestedKeptOriginalIndices: readonly number[] = [];
    if (
      command.kind === "group" ||
      command.kind === "clip" ||
      command.kind === "layer"
    ) {
      const nested = enforceDecorativeBudget(
        command.children as readonly QualityAnnotatedCommand[],
        Math.max(0, maxDecorativeCommands - decorativeCount),
        originalIndices,
        walk,
      );
      walk = nested.nextIndex;
      nestedKeptOriginalIndices = nested.keptOriginalIndices;
      for (const suppressed of nested.suppressedIndices) {
        suppressedIndices.push(suppressed);
      }
      for (const family of nested.suppressedFamilies) {
        suppressedFamilies.add(family);
      }
      decorativeCount += nested.commands.filter(
        (child) => commandQualityClass(child) === "decorative",
      ).length;
      nextCommand = {
        ...command,
        children: nested.commands,
      } as QualityAnnotatedCommand;
    }

    if (commandQualityClass(nextCommand) === "decorative") {
      if (decorativeCount >= maxDecorativeCommands) {
        suppressedIndices.push(commandIndex);
        if (isDecorativeFamily(nextCommand.decorativeFamily)) {
          suppressedFamilies.add(nextCommand.decorativeFamily);
        }
        continue;
      }
      decorativeCount += 1;
    }

    // Pre-order: this node, then remaining descendants.
    keptOriginalIndices.push(commandIndex);
    for (const originalIndex of nestedKeptOriginalIndices) {
      keptOriginalIndices.push(originalIndex);
    }
    kept.push(nextCommand);
  }

  return {
    commands: kept,
    suppressedIndices,
    suppressedFamilies,
    nextIndex: walk,
    keptOriginalIndices,
  };
}

/**
 * Applies a quality profile to a display list.
 *
 * Preserves every `required-semantic` command and every hit region whose role
 * is `select`, `inspect`, or `focus`. Decorative particle/blur/shadow/glow
 * work is suppressed according to the shared Canvas decorative flags.
 * Reduced-motion zeroes motion metadata on remaining commands.
 */
export function applyQualityPolicy(
  list: QualityDisplayList | DisplayList,
  profile: QualityPolicyProfile,
  displayContract?: DisplayContract,
): QualityPolicyResult {
  const annotated = list as QualityDisplayList;
  let filtered = filterCommands(annotated.commands, profile, displayContract, 0);

  if (
    displayContract?.maxDecorativeCommands !== undefined &&
    Number.isSafeInteger(displayContract.maxDecorativeCommands) &&
    displayContract.maxDecorativeCommands >= 0
  ) {
    const budgeted = enforceDecorativeBudget(
      filtered.commands,
      displayContract.maxDecorativeCommands,
      filtered.keptOriginalIndices,
      0,
    );
    const suppressedFamilies = new Set(filtered.suppressedFamilies);
    for (const family of budgeted.suppressedFamilies) {
      suppressedFamilies.add(family);
    }
    filtered = {
      commands: budgeted.commands,
      suppressedIndices: [
        ...filtered.suppressedIndices,
        ...budgeted.suppressedIndices,
      ].sort((left, right) => left - right),
      suppressedFamilies,
      nextIndex: budgeted.nextIndex,
      keptOriginalIndices: budgeted.keptOriginalIndices,
    };
  }

  const keptHitRegions: QualityAnnotatedHitRegion[] = [];
  const suppressedHitRegionIds: string[] = [];
  for (const region of annotated.hitRegions) {
    if (shouldKeepHitRegion(region, profile)) {
      keptHitRegions.push(region);
    } else {
      suppressedHitRegionIds.push(region.id);
    }
  }

  const nextList = buildDisplayList({
    commands: filtered.commands,
    hitRegions: keptHitRegions,
    paintBounds: annotated.paintBounds,
    damageBounds: annotated.damageBounds,
  }) as QualityDisplayList;

  const suppressedFamilies = FAMILY_ORDER.filter((family) =>
    filtered.suppressedFamilies.has(family),
  );

  const report: DegradationReport = deepFreeze({
    tier: profile.tier,
    motionReduced: profile.motion === "reduced",
    suppressedCommandIndices: [...filtered.suppressedIndices],
    suppressedFamilies,
    suppressedHitRegionIds,
  });

  return deepFreeze({
    list: nextList,
    report,
  });
}
