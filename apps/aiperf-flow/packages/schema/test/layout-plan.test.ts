// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  layoutBoundsForNode,
  parseLayoutPlan,
  safeParseLayoutPlan,
} from "../src/layout-plan.js";

describe("layout plan", () => {
  test("parses node bounds and routes", () => {
    const plan = parseLayoutPlan({
      version: 1,
      nodes: [
        {
          nodeId: "seg-system",
          bounds: { x: 0, y: 0, width: 120, height: 24 },
          clip: true,
        },
      ],
      routes: [
        {
          edgeId: "e0",
          points: [
            { x: 0, y: 12 },
            { x: 40, y: 12 },
          ],
        },
      ],
    });

    expect(layoutBoundsForNode(plan, "seg-system")).toEqual({
      x: 0,
      y: 0,
      width: 120,
      height: 24,
    });
    expect(plan.routes[0]?.points).toHaveLength(2);
  });

  test("rejects unknown fields", () => {
    const result = safeParseLayoutPlan({
      version: 1,
      nodes: [],
      routes: [],
      seed: 42,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics[0]?.code).toBe("LAYOUT_PLAN_INVALID");
    }
  });
});
