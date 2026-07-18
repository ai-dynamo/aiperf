// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test, vi } from "vitest";

import type { DrawCommand } from "../../src/display-list.js";
import {
  applyTimelineState,
  evaluateTimelineState,
} from "../../src/evaluate/timeline-state.js";
import { TimelinePlayer } from "../../src/player.js";

const sourceMap = {
  source: "timeline.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

const timeline = [
  {
    id: "reveal-panel",
    at: 100,
    duration: 400,
    action: "reveal",
    target: "panel",
    sourceMap,
  },
  {
    id: "trace-route",
    at: 500,
    duration: 800,
    action: "trace",
    target: "route",
    sourceMap,
  },
] satisfies SceneIr["timeline"];

const bounds = { x: 10, y: 20, width: 200, height: 80 };

const commands: readonly DrawCommand[] = [
  {
    kind: "path",
    id: "panel",
    order: 0,
    paintBounds: bounds,
    damageBounds: bounds,
    path: "M 10 20 H 210 V 100 H 10 Z",
    fill: "#123456",
  },
  {
    kind: "path",
    id: "route",
    order: 1,
    paintBounds: bounds,
    damageBounds: bounds,
    path: "M 10 20 L 210 100",
    stroke: "#76b900",
  },
];

describe("timeline evaluation contribution", () => {
  test.each([0, 100, 300, 500, 900, 1_300, 5_000])(
    "matches direct TimelinePlayer seek at %i ms",
    (atMs) => {
      expect(evaluateTimelineState(timeline, atMs)).toEqual(
        new TimelinePlayer(timeline).seek(atMs),
      );
    },
  );

  test("derives state without reading wall time", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });
    const performanceNow = vi.spyOn(performance, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });

    expect(evaluateTimelineState(timeline, 300).targets.panel?.progress).toBe(
      0.5,
    );
    expect(dateNow).not.toHaveBeenCalled();
    expect(performanceNow).not.toHaveBeenCalled();
  });

  test("applies reveal as opacity and trace as a deterministic clip", () => {
    const state = evaluateTimelineState(timeline, 900);

    expect(applyTimelineState(commands, state)).toEqual([
      commands[0],
      {
        kind: "clip",
        id: "route:timeline-trace",
        order: 1,
        paintBounds: bounds,
        damageBounds: bounds,
        path: "M 10 20 H 110 V 100 H 10 Z",
        children: [commands[1]],
      },
    ]);
  });

  test("applies effects recursively while retaining target command identity", () => {
    const group: DrawCommand = {
      kind: "group",
      id: "root",
      order: 0,
      paintBounds: bounds,
      damageBounds: bounds,
      children: commands,
    };
    const result = applyTimelineState(
      [group],
      evaluateTimelineState(timeline, 300),
    );

    expect(result[0]).toMatchObject({
      kind: "group",
      id: "root",
      children: [
        {
          kind: "layer",
          id: "panel:timeline-reveal",
          opacity: 0.5,
          children: [{ id: "panel" }],
        },
        {
          kind: "clip",
          id: "route:timeline-trace",
          children: [{ id: "route" }],
        },
      ],
    });
  });

  test("reduced motion applies the authored final state at every beat", () => {
    const state = evaluateTimelineState(timeline, 0, { reducedMotion: true });

    expect(state).toEqual(new TimelinePlayer(timeline).finalState());
    expect(applyTimelineState(commands, state)).toEqual(commands);
  });

  test("rejects non-integer evaluation time", () => {
    expect(() => evaluateTimelineState(timeline, 1.5)).toThrow(
      "Timeline evaluation time must be a non-negative safe integer.",
    );
  });
});
