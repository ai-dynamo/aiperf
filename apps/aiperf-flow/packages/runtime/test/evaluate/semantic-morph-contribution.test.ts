// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import {
  contributeSemanticMorph,
  type SemanticMorphContributionInput,
} from "../../src/evaluate/contributions/semantic-morph.js";

const cafeRocketInput = {
  id: "tok-morph",
  startMs: 800,
  durationMs: 1200,
  sources: [
    {
      id: "g0",
      label: "c",
      kind: "grapheme",
      bounds: { x: 8, y: 8, width: 12, height: 24 },
    },
    {
      id: "g1",
      label: "a",
      kind: "grapheme",
      bounds: { x: 20, y: 8, width: 12, height: 24 },
    },
    {
      id: "g2",
      label: "f",
      kind: "grapheme",
      bounds: { x: 32, y: 8, width: 12, height: 24 },
    },
    {
      id: "g3",
      label: "é",
      kind: "grapheme",
      bounds: { x: 44, y: 8, width: 12, height: 24 },
    },
    {
      id: "g5",
      label: "🚀",
      kind: "grapheme",
      bounds: { x: 68, y: 8, width: 24, height: 24 },
    },
  ],
  targets: [
    {
      id: "t0",
      label: "151643",
      kind: "token",
      bounds: { x: 8, y: 40, width: 72, height: 24 },
    },
    {
      id: "t1",
      label: "103097",
      kind: "token",
      bounds: { x: 88, y: 40, width: 72, height: 24 },
    },
    {
      id: "t2",
      label: "139",
      kind: "token",
      bounds: { x: 168, y: 40, width: 48, height: 24 },
    },
    {
      id: "t3",
      label: "279",
      kind: "token",
      bounds: { x: 224, y: 40, width: 48, height: 24 },
    },
    {
      id: "t4",
      label: "<|special|>",
      kind: "special-token",
      bounds: { x: 280, y: 40, width: 96, height: 24 },
    },
  ],
  correspondences: [
    {
      id: "e0",
      sourceIds: ["g0", "g1", "g2"],
      targetIds: ["t0"],
      kind: "many-to-one",
    },
    {
      id: "e1",
      sourceIds: ["g3"],
      targetIds: ["t1"],
      kind: "one-to-one",
    },
    {
      id: "e2",
      sourceIds: ["g5"],
      targetIds: ["t2", "t3"],
      kind: "one-to-many",
    },
    {
      id: "e3",
      sourceIds: [],
      targetIds: ["t4"],
      kind: "special-insert",
    },
  ],
  order: 3,
  fill: "#7aa2f7",
} as const satisfies Omit<SemanticMorphContributionInput, "atMs">;

function input(
  atMs: number,
  overrides: Partial<SemanticMorphContributionInput> = {},
): SemanticMorphContributionInput {
  return { ...cafeRocketInput, atMs, ...overrides };
}

function layerOpacity(command: { kind: string; opacity?: number }): number {
  expect(command.kind).toBe("layer");
  return command.opacity ?? 1;
}

describe("contributeSemanticMorph", () => {
  test("uses integer virtual time and never reads wall clocks", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });
    const performanceNow = vi
      .spyOn(performance, "now")
      .mockImplementation(() => {
        throw new Error("wall time must not be read");
      });

    const contribution = contributeSemanticMorph(input(800.9));

    expect(contribution.progress).toBe(0);
    expect(contribution.motionMode).toBe("tween");
    expect(dateNow).not.toHaveBeenCalled();
    expect(performanceNow).not.toHaveBeenCalled();
  });

  test("interpolates correspondence bounds deterministically at mid-beat", () => {
    const mid = contributeSemanticMorph(input(1400));
    const again = contributeSemanticMorph(input(1400));

    expect(mid.progress).toBe(0.5);
    expect(mid).toEqual(again);

    const oneToOne = mid.commands.find((command) => command.id === "tok-morph:e1");
    expect(oneToOne).toMatchObject({
      kind: "path",
      id: "tok-morph:e1",
      order: 3,
      paintBounds: { x: 66, y: 24, width: 42, height: 24 },
      damageBounds: { x: 66, y: 24, width: 42, height: 24 },
      fill: "#7aa2f7",
    });
  });

  test("preserves stable semantic identities and correspondence table", () => {
    const start = contributeSemanticMorph(input(800));
    const end = contributeSemanticMorph(input(2000));

    expect(start.semanticEntities.map(({ id }) => id)).toEqual([
      "g0",
      "g1",
      "g2",
      "g3",
      "g5",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
    ]);
    expect(end.semanticEntities).toEqual(start.semanticEntities);
    expect(start.correspondences).toEqual(cafeRocketInput.correspondences);
    expect(end.correspondences).toEqual(cafeRocketInput.correspondences);
    expect(start.semanticRelations).toEqual([
      {
        id: "e0:g0:t0",
        fromId: "g0",
        toId: "t0",
        role: "many-to-one",
      },
      {
        id: "e0:g1:t0",
        fromId: "g1",
        toId: "t0",
        role: "many-to-one",
      },
      {
        id: "e0:g2:t0",
        fromId: "g2",
        toId: "t0",
        role: "many-to-one",
      },
      {
        id: "e1:g3:t1",
        fromId: "g3",
        toId: "t1",
        role: "one-to-one",
      },
      {
        id: "e2:g5:t2",
        fromId: "g5",
        toId: "t2",
        role: "one-to-many",
      },
      {
        id: "e2:g5:t3",
        fromId: "g5",
        toId: "t3",
        role: "one-to-many",
      },
    ]);
    expect(end.semanticRelations).toEqual(start.semanticRelations);
  });

  test("freezes emitted commands so callers cannot mutate them", () => {
    const contribution = contributeSemanticMorph(input(1400));
    const command = contribution.commands[0];

    expect(Object.isFrozen(contribution)).toBe(true);
    expect(Object.isFrozen(contribution.commands)).toBe(true);
    expect(command).toBeDefined();
    expect(Object.isFrozen(command)).toBe(true);
    expect(() => {
      (command as { order: number }).order = 99;
    }).toThrow();
  });

  test("reduced-motion cut switches at the midpoint without spatial tween", () => {
    const before = contributeSemanticMorph(
      input(1399, { reducedMotion: true, reducedMotionPolicy: "cut" }),
    );
    const after = contributeSemanticMorph(
      input(1400, { reducedMotion: true, reducedMotionPolicy: "cut" }),
    );

    expect(before.motionMode).toBe("cut");
    expect(after.motionMode).toBe("cut");
    expect(before.progress).toBe(0);
    expect(after.progress).toBe(1);

    const beforeOneToOne = before.commands.find(
      (command) => command.id === "tok-morph:e1",
    );
    const afterOneToOne = after.commands.find(
      (command) => command.id === "tok-morph:e1",
    );

    expect(beforeOneToOne).toMatchObject({
      paintBounds: { x: 44, y: 8, width: 12, height: 24 },
    });
    expect(afterOneToOne).toMatchObject({
      paintBounds: { x: 88, y: 40, width: 72, height: 24 },
    });
    expect(before.correspondences).toEqual(cafeRocketInput.correspondences);
    expect(after.correspondences).toEqual(cafeRocketInput.correspondences);
  });

  test("reduced-motion crossfade keeps fixed bounds and blends opacity", () => {
    const mid = contributeSemanticMorph(
      input(1400, { reducedMotion: true, reducedMotionPolicy: "crossfade" }),
    );

    expect(mid.motionMode).toBe("crossfade");
    expect(mid.progress).toBe(0.5);

    const sourceLayer = mid.commands.find(
      (command) => command.id === "tok-morph:e1:source",
    );
    const targetLayer = mid.commands.find(
      (command) => command.id === "tok-morph:e1:target",
    );

    expect(sourceLayer).toMatchObject({
      kind: "layer",
      paintBounds: { x: 44, y: 8, width: 12, height: 24 },
      opacity: 0.5,
    });
    expect(targetLayer).toMatchObject({
      kind: "layer",
      paintBounds: { x: 88, y: 40, width: 72, height: 24 },
      opacity: 0.5,
    });
    expect(layerOpacity(sourceLayer!)).toBe(0.5);
    expect(layerOpacity(targetLayer!)).toBe(0.5);
    expect(mid.correspondences).toEqual(cafeRocketInput.correspondences);
  });

  test("special-insert fades the target in without inventing source geometry", () => {
    const start = contributeSemanticMorph(input(800));
    const mid = contributeSemanticMorph(input(1400));
    const end = contributeSemanticMorph(input(2000));

    const startInsert = start.commands.find(
      (command) => command.id === "tok-morph:e3",
    );
    const midInsert = mid.commands.find(
      (command) => command.id === "tok-morph:e3",
    );
    const endInsert = end.commands.find(
      (command) => command.id === "tok-morph:e3",
    );

    expect(startInsert).toMatchObject({
      kind: "layer",
      opacity: 0,
      paintBounds: { x: 280, y: 40, width: 96, height: 24 },
    });
    expect(midInsert).toMatchObject({
      kind: "layer",
      opacity: 0.5,
      paintBounds: { x: 280, y: 40, width: 96, height: 24 },
    });
    expect(endInsert).toMatchObject({
      kind: "layer",
      opacity: 1,
      paintBounds: { x: 280, y: 40, width: 96, height: 24 },
    });
  });

  test("hit regions track interpolated correspondence bounds", () => {
    const mid = contributeSemanticMorph(input(1400));

    expect(mid.hitRegions).toEqual([
      {
        id: "tok-morph:e0:hit",
        semanticId: "e0",
        order: 3,
        bounds: { x: 8, y: 24, width: 54, height: 24 },
      },
      {
        id: "tok-morph:e1:hit",
        semanticId: "e1",
        order: 3,
        bounds: { x: 66, y: 24, width: 42, height: 24 },
      },
      {
        id: "tok-morph:e2:hit",
        semanticId: "e2",
        order: 3,
        bounds: { x: 118, y: 24, width: 64, height: 24 },
      },
      {
        id: "tok-morph:e3:hit",
        semanticId: "e3",
        order: 3,
        bounds: { x: 280, y: 40, width: 96, height: 24 },
      },
    ]);
  });
});
