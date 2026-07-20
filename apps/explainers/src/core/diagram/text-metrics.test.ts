/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  SCENE_TEXT_SCALE,
  estimateTextWidth,
  scaledSceneFontSize,
  stepperChipWidth,
} from "./text-metrics.js";

describe("scene text metrics", () => {
  it("exports the shared scene text scale", () => {
    expect(SCENE_TEXT_SCALE).toBe(0.9);
  });

  it("scales authored and default font sizes", () => {
    expect(scaledSceneFontSize(20)).toBe(18);
    expect(scaledSceneFontSize(undefined)).toBe(12.6);
  });

  it("estimates width with the scene text scale", () => {
    expect(estimateTextWidth("authoritative", 11, "bold")).toBe(
      Math.ceil(13 * 6.2 * 0.9),
    );
  });

  it("sizes stepper chips from numbered labels under the text scale", () => {
    expect(stepperChipWidth("layout", 0)).toBe(
      Math.max(72, Math.ceil("1. layout".length * 6.2 * 0.9) + 24),
    );
  });
});
