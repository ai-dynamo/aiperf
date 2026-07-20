/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { createSdkRegistry } from "../registry.js";
import type { SdkExpansionContext } from "../types.js";

const SOURCE_MAP = {
  source: "motion.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function context(instanceId: string): SdkExpansionContext {
  return {
    instanceId,
    sourceMap: SOURCE_MAP,
    themeTokens: new Map(),
  };
}

describe("sdk.flow action bindings", () => {
  it("binds draw and trace to the backing edge when edge:true", () => {
    const definition = createSdkRegistry().lookup("sdk.flow")!;
    const result = definition.factory(
      {
        id: "flow",
        edge: true,
        from: { nodeId: "a", anchor: "e" },
        to: { nodeId: "b", anchor: "w" },
      },
      {},
      context("flow"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.actions.draw).toEqual(["flow__edge", "flow__signal"]);
    // Authors choreograph motion with `trace <flow>`; the static backing edge
    // must receive the same stroke cue as `draw` or IR verifies missing-draw-cue.
    expect(result.value.actions.trace).toEqual(["flow__edge", "flow__signal"]);
  });
});
