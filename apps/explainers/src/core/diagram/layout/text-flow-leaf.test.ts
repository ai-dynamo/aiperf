// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { textFlowLeaf } from "./text-flow-leaf.js";
import { wrapTextToWidth } from "../text-metrics.js";

describe("textFlowLeaf", () => {
  it("reports height proportional to wrapped line count", () => {
    const text = "one two three four five six seven eight nine ten";
    const measure = textFlowLeaf(text, 14, "normal");
    const size = measure({ maxWidth: 80 });
    const expectedLines = wrapTextToWidth(text, 80, 14, "normal").length;
    expect(size.height).toBeCloseTo(expectedLines * 14 * 1.3, 5);
    expect(size.width).toBe(80);
  });

  it("reports a single line's height for short text", () => {
    const measure = textFlowLeaf("short", 14, "normal");
    const size = measure({ maxWidth: 400 });
    expect(size.height).toBeCloseTo(14 * 1.3, 5);
  });

  it("respects a custom lineHeightRatio", () => {
    const measure = textFlowLeaf("short", 14, "normal", 1.5);
    const size = measure({ maxWidth: 400 });
    expect(size.height).toBeCloseTo(14 * 1.5, 5);
  });
});
