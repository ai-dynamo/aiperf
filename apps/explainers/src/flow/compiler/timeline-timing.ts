// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolves timeline cue timing (`at <ms>` or `after <ref> [+<gapMs>]`) to
//! absolute millisecond offsets. Shared by native scene lowering
//! (`lower.ts`) and SDK scene expansion (`expand-sdk.ts`) so both authoring
//! paths compute identical absolute times from the same relative-timing
//! semantics.

import type { TimelineCueAst } from "../language/index.js";

/**
 * Resolves each cue's absolute start time in source order. `after <ref>`
 * resolves to the end time (start + duration, or start + step*index +
 * duration for the referenced stagger member) of the most recent prior cue
 * in the same timeline whose target — or, for `stagger`/`enter-children`,
 * whose `targets` list — contains `ref`, plus the cue's `gap`. An `after`
 * cue with no prior match for `ref` resolves to `gap` (fail-soft: link-time
 * validation is responsible for rejecting unresolvable references before
 * lowering ever runs).
 */
export function resolveTimelineCueTiming(
  cues: readonly TimelineCueAst[],
): readonly number[] {
  const endByTarget = new Map<string, number>();
  const resolved: number[] = [];
  for (const cue of cues) {
    const at =
      cue.timing.mode === "at"
        ? cue.timing.ms
        : (endByTarget.get(cue.timing.ref) ?? 0) + cue.timing.gap;
    resolved.push(at);
    if (cue.targets !== undefined && cue.targets.length > 0) {
      cue.targets.forEach((id, index) => {
        endByTarget.set(id, at + (cue.step ?? 0) * index + cue.duration);
      });
    } else if (cue.target.length > 0) {
      endByTarget.set(cue.target, at + cue.duration);
    }
  }
  return resolved;
}

/**
 * Cue indices whose `after <ref>` names an id that no earlier cue in the
 * same timeline has targeted — an unresolvable relative reference (typo,
 * forward reference, or reference to a node with no prior cue). Callers use
 * this to fail closed at link/validation time; `resolveTimelineCueTiming`
 * itself resolves these fail-soft to `gap` so lowering never throws.
 */
export function findUnresolvedAfterRefs(
  cues: readonly TimelineCueAst[],
): readonly number[] {
  const seenTargets = new Set<string>();
  const unresolved: number[] = [];
  cues.forEach((cue, index) => {
    if (cue.timing.mode === "after" && !seenTargets.has(cue.timing.ref)) {
      unresolved.push(index);
    }
    if (cue.targets !== undefined) {
      cue.targets.forEach((id) => seenTargets.add(id));
    } else if (cue.target.length > 0) {
      seenTargets.add(cue.target);
    }
  });
  return unresolved;
}
