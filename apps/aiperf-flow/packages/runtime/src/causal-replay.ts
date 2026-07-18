// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic projection and traversal of authored causal beats.

import type { SceneIr } from "@aiperf/flow-schema";

/** One authored timeline or narrative beat in the causal replay path. */
export type CausalBeat = Readonly<{
  id: string;
  label: string;
  description?: string;
  timeMs: number;
  endMs: number;
  targetEntityIds: readonly string[];
  source: "timeline" | "narrative";
}>;

/** A causal beat's state at one virtual timestamp. */
export type CausalBeatState = "complete" | "active" | "future";

type OrderedCausalBeat = Readonly<{
  beat: CausalBeat;
  authoredOrder: number;
}>;

function assertTime(timeMs: number, field: string): void {
  if (!Number.isSafeInteger(timeMs) || timeMs < 0) {
    throw new RangeError(`${field} must be a non-negative safe integer.`);
  }
}

function checkedEndMs(timeMs: number, durationMs: number, field: string): number {
  assertTime(timeMs, `${field} start`);
  assertTime(durationMs, `${field} duration`);
  const endMs = timeMs + durationMs;
  assertTime(endMs, `${field} end`);
  return endMs;
}

function normalizedLabel(label: string): string {
  return label.replace(/[-_]+/g, " ").replace(/\s+/g, " ").trim();
}

/**
 * Collects an id → accessibility-label map from the scene render tree so a
 * generic `reveal` timeline cue can be labelled by the entity it reveals
 * rather than by the internal rendering verb.
 */
function collectNodeLabels(
  roots: SceneIr["roots"],
  labels: Map<string, string>,
): void {
  for (const node of roots) {
    const label = node.accessibility?.label;
    if (typeof label === "string" && label.length > 0) {
      labels.set(node.id, label);
    }
    if (node.kind === "group" || node.kind === "component") {
      collectNodeLabels(node.children, labels);
    }
  }
}

/**
 * Resolves a human-facing beat label for a timeline cue. `reveal` is a
 * rendering primitive, not a caption, so reveal cues borrow the label of the
 * entity they target; every other authored action reads verbatim.
 */
function timelineBeatLabel(
  cue: SceneIr["timeline"][number],
  nodeLabels: Map<string, string>,
): string {
  if (cue.action.trim().toLowerCase() === "reveal") {
    const targetLabel = nodeLabels.get(cue.target);
    if (targetLabel !== undefined) {
      return normalizedLabel(targetLabel);
    }
  }
  return normalizedLabel(cue.action);
}

function freezeBeat(beat: CausalBeat): CausalBeat {
  return Object.freeze({
    ...beat,
    targetEntityIds: Object.freeze([...beat.targetEntityIds]),
  });
}

/**
 * Projects authored timeline and narrative cues into one deterministic path.
 *
 * Timeline cues take precedence when a narrative cue reuses the same ID.
 */
export function projectCausalBeats(scene: SceneIr): readonly CausalBeat[] {
  const projected: OrderedCausalBeat[] = [];
  const projectedIds = new Set<string>();
  const timelineIds = new Set<string>();
  const narrativeIds = new Set<string>();
  const nodeLabels = new Map<string, string>();
  collectNodeLabels(scene.roots, nodeLabels);

  scene.timeline.forEach((cue, authoredOrder) => {
    if (timelineIds.has(cue.id)) {
      throw new Error(`Duplicate timeline causal beat id "${cue.id}".`);
    }
    timelineIds.add(cue.id);
    const endMs = checkedEndMs(cue.at, cue.duration, `Timeline cue "${cue.id}"`);
    projectedIds.add(cue.id);
    projected.push({
      authoredOrder,
      beat: freezeBeat({
        id: cue.id,
        label: timelineBeatLabel(cue, nodeLabels),
        timeMs: cue.at,
        endMs,
        targetEntityIds: [cue.target],
        source: "timeline",
      }),
    });
  });

  scene.narrativeTrack?.cues.forEach((cue, narrativeOrder) => {
    if (narrativeIds.has(cue.id)) {
      throw new Error(`Duplicate narrative causal beat id "${cue.id}".`);
    }
    narrativeIds.add(cue.id);
    if (projectedIds.has(cue.id)) {
      return;
    }
    assertTime(cue.startMs, `Narrative cue "${cue.id}" start`);
    assertTime(cue.endMs, `Narrative cue "${cue.id}" end`);
    if (cue.endMs < cue.startMs) {
      throw new RangeError(
        `Narrative cue "${cue.id}" end must not precede its start.`,
      );
    }
    projectedIds.add(cue.id);
    projected.push({
      authoredOrder: scene.timeline.length + narrativeOrder,
      beat: freezeBeat({
        id: cue.id,
        label: normalizedLabel(cue.subtitleText),
        description: cue.spokenText,
        timeMs: cue.startMs,
        endMs: cue.endMs,
        targetEntityIds: [],
        source: "narrative",
      }),
    });
  });

  projected.sort(
    (left, right) =>
      left.beat.timeMs - right.beat.timeMs ||
      left.authoredOrder - right.authoredOrder ||
      left.beat.id.localeCompare(right.beat.id, "en", {
        sensitivity: "variant",
      }),
  );
  return Object.freeze(projected.map(({ beat }) => beat));
}

/** Returns the first authored beat active at one virtual timestamp. */
export function activeCausalBeat(
  beats: readonly CausalBeat[],
  timeMs: number,
): CausalBeat | null {
  assertTime(timeMs, "Causal replay time");
  let active: CausalBeat | null = null;
  for (const beat of beats) {
    if (
      causalBeatState(beat, timeMs) === "active" &&
      (active === null || beat.timeMs > active.timeMs)
    ) {
      active = beat;
    }
  }
  return active;
}

/** Derives one causal beat's state at a virtual timestamp. */
export function causalBeatState(
  beat: CausalBeat,
  timeMs: number,
): CausalBeatState {
  assertTime(timeMs, "Causal replay time");
  if (timeMs < beat.timeMs) {
    return "future";
  }
  return timeMs < beat.endMs ? "active" : "complete";
}

/** Resolves keyboard traversal without wrapping at path boundaries. */
export function adjacentCausalBeat(
  beats: readonly CausalBeat[],
  activeId: string | null,
  direction: "first" | "previous" | "next" | "last",
): CausalBeat | null {
  if (beats.length === 0) {
    return null;
  }
  if (direction === "first") {
    return beats[0] ?? null;
  }
  if (direction === "last") {
    return beats.at(-1) ?? null;
  }
  if (activeId === null) {
    return null;
  }

  const activeIndex = beats.findIndex((beat) => beat.id === activeId);
  if (activeIndex < 0) {
    return null;
  }
  const adjacentIndex =
    direction === "previous" ? activeIndex - 1 : activeIndex + 1;
  return beats[adjacentIndex] ?? null;
}
