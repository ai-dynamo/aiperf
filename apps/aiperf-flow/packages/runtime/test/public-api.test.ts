// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  applyQualityPolicy,
  CanvasTextAtlas,
  CausalPath,
  CommandConstellation,
  computeDamageBetween,
  ContextLens,
  createHitRegionIndex,
  evaluateFrame,
  hudVisibilityFor,
  immersiveReducer,
  ImmersiveControls,
  parseImmersiveUrl,
  projectCausalBeats,
  searchCommands,
  SemanticFallbackTable,
  ThemeRegistry,
} from "../src/index.js";

describe("runtime public API", () => {
  test("exports the landed frame, Canvas, and semantic helpers", () => {
    expect(CanvasTextAtlas).toBeTypeOf("function");
    expect(createHitRegionIndex).toBeTypeOf("function");
    expect(computeDamageBetween).toBeTypeOf("function");
    expect(applyQualityPolicy).toBeTypeOf("function");
    expect(SemanticFallbackTable).toBeTypeOf("function");
    expect(evaluateFrame).toBeTypeOf("function");
    expect(projectCausalBeats).toBeTypeOf("function");
    expect(immersiveReducer).toBeTypeOf("function");
    expect(searchCommands).toBeTypeOf("function");
    expect(parseImmersiveUrl).toBeTypeOf("function");
    expect(hudVisibilityFor).toBeTypeOf("function");
    expect(CausalPath).toBeTypeOf("function");
    expect(CommandConstellation).toBeTypeOf("function");
    expect(ContextLens).toBeTypeOf("function");
    expect(ImmersiveControls).toBeTypeOf("function");
    expect(ThemeRegistry).toBeTypeOf("function");
  });
});
