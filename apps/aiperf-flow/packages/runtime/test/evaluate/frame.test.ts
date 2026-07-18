// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for the pure `evaluateFrame` composition seam.
//
// Focuses on:
// - evaluation at an exact non-negative integer virtual time;
// - reference-by-default vs. degraded quality tiers preserving semantic hit
//   regions and semantic IDs;
// - the hit index deriving keyboard traversal from the emitted display list;
// - damage regions computed against an optional previous frame vs. the
//   first-frame full-damage fallback;
// - deep-frozen, deterministic, JSON-serializable output.
//
// Out of scope: quality-policy suppression internals (see
// evaluate/quality-policy.test.ts), damage geometry (see
// evaluate/damage-tracker.test.ts), and picking (see
// evaluate/hit-region-index.test.ts).

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  buildDisplayList,
  type DisplayList,
} from "../../src/display-list.js";
import { computeDamageBetween } from "../../src/evaluate/damage-tracker.js";
import {
  evaluateFrame,
  type EvaluatedFrame,
} from "../../src/evaluate/frame.js";
import { qualityPolicyProfile } from "../../src/evaluate/quality-policy.js";

const sourceMap = {
  source: "request-lifecycle.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function scene(overrides: Partial<SceneIr> = {}): SceneIr {
  return {
    id: "request-lifecycle",
    title: "Request lifecycle",
    summary: "Arrival through first token",
    roots: [
      {
        kind: "rect",
        id: "request-a",
        geometry: { x: 16, y: 16, width: 96, height: 32 },
        style: { fill: "#76b900" },
        accessibility: { label: "Request A" },
        fallback: "Request A unavailable",
        sourceMap,
      },
      {
        kind: "rect",
        id: "first-token",
        geometry: { x: 160, y: 16, width: 96, height: 32 },
        style: { fill: "#123456" },
        accessibility: {
          label: "First token",
          description: "The first streamed output token",
        },
        fallback: "First token unavailable",
        sourceMap,
      },
      {
        kind: "connector",
        id: "route",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#ffffff" },
        accessibility: { label: "Request route" },
        fallback: "Route unavailable",
        sourceMap,
        from: { nodeId: "request-a" },
        to: { nodeId: "first-token" },
      },
    ],
    camera: [],
    timeline: [
      {
        id: "arrival",
        at: 0,
        duration: 1000,
        target: "request-a",
        action: "Reveal",
        sourceMap,
      },
      {
        id: "admission",
        at: 1000,
        duration: 1000,
        target: "first-token",
        action: "Trace",
        sourceMap,
      },
    ],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: {
      label: "Request lifecycle scene",
      readingOrder: ["request-a", "first-token", "route"],
    },
    fallback: "Scene unavailable",
    sourceMap,
    ...overrides,
  };
}

function previousList(): DisplayList {
  return buildDisplayList({
    commands: [
      {
        kind: "path",
        id: "request-a",
        order: 0,
        path: "M 0 16 H 80 V 48 H 0 Z",
        fill: "#76b900",
        paintBounds: { x: 0, y: 16, width: 80, height: 32 },
        damageBounds: { x: 0, y: 16, width: 80, height: 32 },
      },
    ],
    hitRegions: [],
    paintBounds: { x: 0, y: 16, width: 80, height: 32 },
    damageBounds: { x: 0, y: 16, width: 80, height: 32 },
  });
}

describe("evaluateFrame", () => {
  test("composes a degraded frame at the exact authored integer time", () => {
    const previousDisplayList = previousList();

    const frame = evaluateFrame(scene(), 1500, {
      quality: qualityPolicyProfile("degraded", { motion: "reduced" }),
      previousDisplayList,
    });

    expect(frame.scene.atMs).toBe(1500);
    expect(
      frame.displayList.hitRegions.map(({ semanticId }) => semanticId),
    ).toEqual(
      frame.scene.displayList.hitRegions.map(({ semanticId }) => semanticId),
    );
    expect(frame.report.tier).toBe("degraded");
    expect(frame.report.motionReduced).toBe(true);
    expect(frame.hitIndex.keyboardTraversal).toEqual(
      frame.displayList.hitRegions,
    );
    expect(frame.damageRegions).toEqual(
      computeDamageBetween(previousDisplayList, frame.displayList),
    );
    expect(Object.isFrozen(frame)).toBe(true);
  });

  test("carries the quality-filtered display list on the evaluated scene", () => {
    const frame = evaluateFrame(scene(), 0);

    expect(frame.scene.displayList).toBe(frame.displayList);
  });

  test("defaults to the reference tier with full motion", () => {
    const frame = evaluateFrame(scene(), 0);

    expect(frame.report.tier).toBe("reference");
    expect(frame.report.motionReduced).toBe(false);
    expect(frame.report.suppressedCommandIndices).toEqual([]);
    expect(frame.report.suppressedHitRegionIds).toEqual([]);
  });

  test("preserves every semantic hit region when degrading quality", () => {
    const reference = evaluateFrame(scene(), 500);
    const degraded = evaluateFrame(scene(), 500, {
      quality: qualityPolicyProfile("degraded"),
    });

    expect(degraded.displayList.hitRegions.map(({ semanticId }) => semanticId))
      .toEqual(
        reference.displayList.hitRegions.map(({ semanticId }) => semanticId),
      );
    expect(degraded.report.suppressedHitRegionIds).toEqual([]);
    expect(degraded.scene.semantic.entities.map(({ id }) => id)).toEqual([
      "request-a",
      "first-token",
    ]);
  });

  test("derives keyboard traversal from the emitted display-list order", () => {
    const frame = evaluateFrame(scene(), 0);

    expect(frame.hitIndex.keyboardTraversal.map(({ semanticId }) => semanticId))
      .toEqual(["request-a", "first-token", "route"]);
  });

  test("reports whole-frame damage when no previous frame is supplied", () => {
    const frame = evaluateFrame(scene(), 0);

    expect(frame.damageRegions).toEqual([frame.displayList.damageBounds]);
  });

  test("computes incremental damage against the supplied previous frame", () => {
    const previousDisplayList = previousList();
    const frame = evaluateFrame(scene(), 0, { previousDisplayList });

    expect(frame.damageRegions).toEqual(
      computeDamageBetween(previousDisplayList, frame.displayList),
    );
    expect(frame.damageRegions).not.toEqual([frame.displayList.damageBounds]);
  });

  test("evaluates identical inputs to deep-equal serializable frames", () => {
    const first = evaluateFrame(scene(), 750, {
      quality: qualityPolicyProfile("degraded", { motion: "reduced" }),
    });
    const second = evaluateFrame(scene(), 750, {
      quality: qualityPolicyProfile("degraded", { motion: "reduced" }),
    });

    expect(first).toEqual(second);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });

  test("deeply freezes the scene, display list, and hit index", () => {
    const frame = evaluateFrame(scene(), 0);

    expect(Object.isFrozen(frame.scene)).toBe(true);
    expect(Object.isFrozen(frame.displayList)).toBe(true);
    expect(Object.isFrozen(frame.displayList.hitRegions)).toBe(true);
    expect(Object.isFrozen(frame.hitIndex)).toBe(true);
    expect(Object.isFrozen(frame.report)).toBe(true);
    expect(Object.isFrozen(frame.damageRegions)).toBe(true);
  });

  test.each<{ label: string; timeMs: number }>([
    { label: "fractional", timeMs: 1500.5 },
    { label: "negative", timeMs: -1 },
    { label: "not-a-number", timeMs: Number.NaN },
    { label: "beyond-safe-integer", timeMs: Number.MAX_SAFE_INTEGER + 1 },
  ])(
    "rejects $label evaluation time with a non-negative-safe-integer error",
    ({ timeMs }) => {
      expect(() => evaluateFrame(scene(), timeMs)).toThrow(
        /non-negative safe integer/iu,
      );
    },
  );

  test("keeps every required-semantic command under a decorative budget", () => {
    const unbudgeted = evaluateFrame(scene(), 0);
    const budgeted = evaluateFrame(scene(), 0, {
      displayContract: {
        maxDecorativeCommands: 0,
        supportedDecorativeFamilies: [],
      },
    });

    const ids = (frame: EvaluatedFrame): readonly string[] =>
      frame.displayList.commands.map(({ id }) => id);

    expect(ids(budgeted)).toEqual(ids(unbudgeted));
    expect(budgeted.report.suppressedCommandIndices).toEqual([]);
  });
});
