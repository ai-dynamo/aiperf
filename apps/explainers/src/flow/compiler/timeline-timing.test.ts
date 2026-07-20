/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { TimelineCueAst } from "../language/ast.js";
import {
  findUnresolvedAfterRefs,
  resolveTimelineCueTiming,
} from "./timeline-timing.js";

const SOURCE_MAP = {
  source: "timeline-timing.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function cue(
  partial: Pick<TimelineCueAst, "target" | "timing" | "duration"> &
    Partial<Pick<TimelineCueAst, "action" | "targets" | "step">>,
): TimelineCueAst {
  return {
    kind: "timeline-cue",
    action: partial.action ?? "enter",
    target: partial.target,
    duration: partial.duration,
    timing: partial.timing,
    sourceMap: SOURCE_MAP,
    ...(partial.targets !== undefined ? { targets: partial.targets } : {}),
    ...(partial.step !== undefined ? { step: partial.step } : {}),
  };
}

describe("timeline empty targets[] agreement", () => {
  it("falls through to cue.target when targets is an empty array", () => {
    const cues: TimelineCueAst[] = [
      cue({
        target: "node-a",
        targets: [],
        duration: 100,
        timing: { mode: "at", ms: 0 },
      }),
      cue({
        target: "node-b",
        duration: 50,
        timing: { mode: "after", ref: "node-a", gap: 0 },
      }),
    ];

    expect(findUnresolvedAfterRefs(cues)).toEqual([]);
    expect(resolveTimelineCueTiming(cues)).toEqual([0, 100]);
  });
});
