/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  surfaceClassName,
  inkClassName,
  strokeClassName,
  categoryClassName,
  categoryBgClassName,
  categoryBgTintClassName,
  categoryFillClassName,
  categoryStrokeClassName,
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

  it("maps every category role to a literal text color class, including the ones that were previously JIT-invisible", () => {
    expect(categoryClassName("purple")).toBe("text-category-purple");
    expect(categoryClassName("orange")).toBe("text-category-orange");
    expect(categoryClassName("cyan")).toBe("text-category-cyan");
    expect(categoryClassName("gray")).toBe("text-category-gray");
  });

  it("maps every category role to a literal SVG fill class", () => {
    expect(categoryFillClassName("green")).toBe("fill-category-green");
    expect(categoryFillClassName("purple")).toBe("fill-category-purple");
    expect(categoryFillClassName("orange")).toBe("fill-category-orange");
    expect(categoryFillClassName("cyan")).toBe("fill-category-cyan");
    expect(categoryFillClassName("gray")).toBe("fill-category-gray");
  });

  it("maps every category role to a literal SVG stroke class", () => {
    expect(categoryStrokeClassName("green")).toBe("stroke-category-green");
    expect(categoryStrokeClassName("purple")).toBe("stroke-category-purple");
    expect(categoryStrokeClassName("orange")).toBe("stroke-category-orange");
    expect(categoryStrokeClassName("cyan")).toBe("stroke-category-cyan");
    expect(categoryStrokeClassName("gray")).toBe("stroke-category-gray");
  });
});
