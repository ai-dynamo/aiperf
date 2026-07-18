// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  canvasQualityProfile,
  qualityProfile,
  type QualityProfile,
} from "../../src/backends/canvas/quality.js";

describe("Canvas quality profiles", () => {
  test("reference enables every decorative effect", () => {
    expect(qualityProfile("reference")).toEqual({
      tier: "reference",
      decorative: {
        blur: true,
        glow: true,
        particles: true,
      },
    } satisfies QualityProfile);
  });

  test("degraded disables decorative effects without expressing semantic filtering", () => {
    expect(qualityProfile("degraded")).toEqual({
      tier: "degraded",
      decorative: {
        blur: false,
        glow: false,
        particles: false,
      },
    } satisfies QualityProfile);
  });

  test("maps interactive rendering to the degraded profile", () => {
    expect(canvasQualityProfile("reference")).toBe(qualityProfile("reference"));
    expect(canvasQualityProfile("interactive")).toBe(qualityProfile("degraded"));
  });

  test("returns immutable shared profiles from a pure lookup", () => {
    const first = qualityProfile("degraded");
    const second = qualityProfile("degraded");

    expect(first).toBe(second);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.decorative)).toBe(true);
  });
});
