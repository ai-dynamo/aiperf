/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  surfaceClassName,
  inkClassName,
  strokeClassName,
  categoryBgClassName,
  categoryBgTintClassName,
} from "./tokens.js";

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

  it("maps every category role to a literal background class, including the ones that were previously JIT-invisible", () => {
    expect(categoryBgClassName("purple")).toBe("bg-category-purple");
    expect(categoryBgClassName("orange")).toBe("bg-category-orange");
    expect(categoryBgClassName("cyan")).toBe("bg-category-cyan");
    expect(categoryBgClassName("gray")).toBe("bg-category-gray");
  });

  it("maps every category role to a literal 10%-opacity tint class", () => {
    expect(categoryBgTintClassName("purple")).toBe("bg-category-purple/10");
    expect(categoryBgTintClassName("orange")).toBe("bg-category-orange/10");
    expect(categoryBgTintClassName("cyan")).toBe("bg-category-cyan/10");
    expect(categoryBgTintClassName("gray")).toBe("bg-category-gray/10");
  });
});
