/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { RectNodeIr, SceneIr, TimelineCueIr } from "../../schema/ir.js";
import { evaluateFrame } from "./frame.js";
import { qualityPolicyProfile } from "./quality-policy.js";

const SOURCE_MAP = {
  source: "frame.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function rect(
  id: string,
  geometry: Readonly<{ x: number; y: number; width: number; height: number }>,
): RectNodeIr {
  return {
    id,
    kind: "rect",
    geometry,
    style: {},
    accessibility: { label: id },
    fallback: id,
    sourceMap: SOURCE_MAP,
  };
}

function cue(
  partial: Omit<TimelineCueIr, "sourceMap"> &
    Partial<Pick<TimelineCueIr, "sourceMap">>,
): TimelineCueIr {
  return {
    sourceMap: SOURCE_MAP,
    ...partial,
  };
}

function scene(
  roots: SceneIr["roots"],
  readingOrder: readonly string[],
  timeline: readonly TimelineCueIr[] = [],
): SceneIr {
  return {
    id: "scene",
    title: "Scene",
    summary: "test",
    roots,
    camera: [],
    timeline: [...timeline],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: { label: "Scene", readingOrder: [...readingOrder] },
    fallback: "Scene",
    sourceMap: SOURCE_MAP,
  };
}

describe("evaluateFrame timeline effects", () => {
  it("applies reveal progress into the frame display list", () => {
    const node = rect("box", { x: 0, y: 0, width: 40, height: 20 });
    const ir = scene([node], ["box"], [
      cue({
        id: "reveal-box",
        at: 0,
        duration: 100,
        target: "box",
        action: "reveal",
      }),
    ]);

    const frame = evaluateFrame(ir, 50);

    const command = frame.displayList.commands[0];
    expect(command?.kind).toBe("layer");
    expect(command?.id).toBe("box:timeline-reveal");
    if (command?.kind === "layer") {
      expect(command.opacity).toBe(0.5);
      expect(command.children).toHaveLength(1);
      expect(command.children[0]?.id).toBe("box");
    }
  });

  it("applies trace progress as a clip wrapper", () => {
    const node = rect("path-node", { x: 10, y: 20, width: 100, height: 40 });
    const ir = scene([node], ["path-node"], [
      cue({
        id: "trace-path",
        at: 0,
        duration: 100,
        target: "path-node",
        action: "trace",
      }),
    ]);

    const frame = evaluateFrame(ir, 25);

    const command = frame.displayList.commands[0];
    expect(command?.kind).toBe("clip");
    expect(command?.id).toBe("path-node:timeline-trace");
    if (command?.kind === "clip") {
      expect(command.path).toBe("M 10 20 H 35 V 60 H 10 Z");
      expect(command.children[0]?.id).toBe("path-node");
    }
  });

  it("leaves completed reveal targets unwrapped", () => {
    const node = rect("box", { x: 0, y: 0, width: 40, height: 20 });
    const ir = scene([node], ["box"], [
      cue({
        id: "reveal-box",
        at: 0,
        duration: 100,
        target: "box",
        action: "reveal",
      }),
    ]);

    const frame = evaluateFrame(ir, 100);

    const command = frame.displayList.commands[0];
    expect(command?.kind).not.toBe("layer");
    expect(command?.id).toBe("box");
  });

  it("jumps to final timeline state under reduced motion", () => {
    const node = rect("box", { x: 0, y: 0, width: 40, height: 20 });
    const ir = scene([node], ["box"], [
      cue({
        id: "reveal-box",
        at: 0,
        duration: 100,
        target: "box",
        action: "reveal",
      }),
    ]);

    const frame = evaluateFrame(ir, 10, {
      quality: qualityPolicyProfile("reference", { motion: "reduced" }),
    });

    const command = frame.displayList.commands[0];
    expect(command?.id).toBe("box");
    expect(command?.kind).not.toBe("layer");
  });
});
