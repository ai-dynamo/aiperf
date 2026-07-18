// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure timeline evaluation and backend-neutral visual contributions.

import type { SceneIr } from "../../schema/index.js";

import type {
  Bounds,
  DrawCommand,
  SourceReference,
} from "../display-list.js";
import type {
  TimelineSnapshot,
  TimelineTargetState,
} from "./timeline-types.js";

type Timeline = SceneIr["timeline"];
type UnknownRecord = Readonly<Record<string, unknown>>;

export type TimelineEvaluationOptions = Readonly<{
  reducedMotion?: boolean;
}>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function finiteNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function timelineDurationMs(timeline: Timeline): number {
  return Math.ceil(
    timeline.reduce((maximum, value) => {
      const cue = record(value);
      return Math.max(
        maximum,
        finiteNumber(cue.at) + finiteNumber(cue.duration),
      );
    }, 0),
  );
}

/**
 * Derives the complete timeline target state from one integer virtual time.
 * Reduced motion selects the authored final state without changing semantics.
 */
export function evaluateTimelineState(
  timeline: Timeline,
  atMs: number,
  options: TimelineEvaluationOptions = {},
): TimelineSnapshot {
  if (!Number.isSafeInteger(atMs) || atMs < 0) {
    throw new RangeError(
      "Timeline evaluation time must be a non-negative safe integer.",
    );
  }

  const durationMs = timelineDurationMs(timeline);
  const timeMs =
    options.reducedMotion === true ? durationMs : Math.min(atMs, durationMs);
  const targets: Record<string, TimelineTargetState> = {};

  for (const value of timeline) {
    const cue = record(value);
    const target = typeof cue.target === "string" ? cue.target : "";
    if (target === "") {
      continue;
    }
    const startMs = finiteNumber(cue.at);
    const cueDurationMs = finiteNumber(cue.duration);
    const progress =
      cueDurationMs === 0
        ? Number(timeMs >= startMs)
        : Math.min(1, Math.max(0, (timeMs - startMs) / cueDurationMs));
    targets[target] = Object.freeze({
      action: typeof cue.action === "string" ? cue.action : "",
      progress,
    });
  }

  return Object.freeze({
    timeMs,
    complete: timeMs >= durationMs,
    targets: Object.freeze(targets),
  });
}

function commandBase(command: DrawCommand): Readonly<{
  order: number;
  paintBounds: Bounds;
  damageBounds: Bounds;
  source?: SourceReference;
}> {
  return {
    order: command.order,
    paintBounds: command.paintBounds,
    damageBounds: command.damageBounds,
    ...(command.source === undefined ? {} : { source: command.source }),
  };
}

function traceClipPath(bounds: Bounds, progress: number): string {
  const right = bounds.x + bounds.width;
  const bottom = bounds.y + bounds.height;
  if (bounds.width >= bounds.height) {
    const progressRight = bounds.x + bounds.width * progress;
    return `M ${bounds.x} ${bounds.y} H ${progressRight} V ${bottom} H ${bounds.x} Z`;
  }
  const progressBottom = bounds.y + bounds.height * progress;
  return `M ${bounds.x} ${bounds.y} H ${right} V ${progressBottom} H ${bounds.x} Z`;
}

function applyTargetEffect(
  command: DrawCommand,
  target: TimelineTargetState | undefined,
): DrawCommand {
  if (target === undefined || target.progress >= 1) {
    return command;
  }
  if (target.action === "reveal") {
    return {
      kind: "layer",
      id: `${command.id}:timeline-reveal`,
      ...commandBase(command),
      opacity: target.progress,
      children: [command],
    };
  }
  if (target.action === "trace") {
    return {
      kind: "clip",
      id: `${command.id}:timeline-trace`,
      ...commandBase(command),
      path: traceClipPath(command.paintBounds, target.progress),
      children: [command],
    };
  }
  return command;
}

function applyCommand(
  command: DrawCommand,
  targets: TimelineSnapshot["targets"],
): DrawCommand {
  const nested =
    command.kind === "group" ||
    command.kind === "clip" ||
    command.kind === "layer"
      ? {
          ...command,
          children: command.children.map((child) =>
            applyCommand(child, targets),
          ),
        }
      : command;
  return applyTargetEffect(nested, targets[command.id]);
}

/** Applies timeline reveal and trace effects without backend or wall-clock access. */
export function applyTimelineState(
  commands: readonly DrawCommand[],
  state: TimelineSnapshot,
): readonly DrawCommand[] {
  return commands.map((command) => applyCommand(command, state.targets));
}
