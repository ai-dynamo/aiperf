/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { surfaceClassName, inkClassName, strokeClassName } from "./tokens.js";

describe("theme role class helpers", () => {
  it("maps a surface role to its Tailwind class", () => {
    expect(surfaceClassName("elevated")).toBe("bg-surface-elevated");
  });

  it("maps an ink role to its Tailwind class", () => {
    expect(inkClassName("secondary")).toBe("text-ink-secondary");
  });

  it("maps a stroke role to its Tailwind class", () => {
    expect(strokeClassName("primary")).toBe("border-stroke-primary");
  });
});
