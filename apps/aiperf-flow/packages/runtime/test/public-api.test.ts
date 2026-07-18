// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  applyQualityPolicy,
  CanvasTextAtlas,
  computeDamageBetween,
  createHitRegionIndex,
  SemanticFallbackTable,
} from "../src/index.js";

describe("runtime public API", () => {
  test("exports the landed frame, Canvas, and semantic helpers", () => {
    expect(CanvasTextAtlas).toBeTypeOf("function");
    expect(createHitRegionIndex).toBeTypeOf("function");
    expect(computeDamageBetween).toBeTypeOf("function");
    expect(applyQualityPolicy).toBeTypeOf("function");
    expect(SemanticFallbackTable).toBeTypeOf("function");
  });
});
