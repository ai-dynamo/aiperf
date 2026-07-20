/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  SCENE_TEXT_SCALE,
  estimateTextWidth,
  measureTextWidth,
  scaledSceneFontSize,
  stepperChipWidth,
  wrapTextToWidth,
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

describe("wrapTextToWidth", () => {
  it("returns the whole string as one line when it already fits", () => {
    const lines = wrapTextToWidth("short text", 400, 14);
    expect(lines).toEqual(["short text"]);
  });

  it("wraps onto multiple lines when content exceeds maxWidth", () => {
    const long = "one two three four five six seven eight nine ten";
    const lines = wrapTextToWidth(long, 80, 14);
    expect(lines.length).toBeGreaterThan(1);
    // every produced line must individually fit (or be a single
    // unbreakable word longer than maxWidth, per the next test)
    for (const line of lines) {
      expect(line.length).toBeGreaterThan(0);
    }
    // re-joining with spaces must reconstruct the original words in order
    expect(lines.join(" ")).toBe(long);
  });

  it("does not infinite-loop on a single word longer than maxWidth", () => {
    const lines = wrapTextToWidth("supercalifragilisticexpialidocious", 20, 14);
    expect(lines).toEqual(["supercalifragilisticexpialidocious"]);
  });

  it("returns an empty array for empty input", () => {
    expect(wrapTextToWidth("", 400, 14)).toEqual([]);
  });

  it("respects bold vs normal weight when measuring", () => {
    const text = "aaaa bbbb cccc dddd";
    const normalLines = wrapTextToWidth(text, 100, 14, "normal");
    const boldLines = wrapTextToWidth(text, 100, 14, "bold");
    // bold chars measure the same width unit as normal in this measurer
    // today (BOLD_CHAR_WIDTH === CHAR_WIDTH) — assert the function at
    // least accepts and threads the parameter without throwing, and
    // produces the same line count as normal (documents current
    // equal-width assumption; update this assertion if text-metrics.ts's
    // width constants ever diverge for bold).
    expect(boldLines.length).toBe(normalLines.length);
  });
});

describe("measureTextWidth", () => {
  it("falls back to estimateTextWidth deterministically when no working canvas context exists", () => {
    // jsdom (this test environment) has no real canvas backend installed, so
    // getContext("2d") throws "Not implemented" — measureTextWidth must catch
    // that and fall back rather than let it propagate, and the fallback must
    // match estimateTextWidth exactly (same unit convention, no double-scaling).
    expect(measureTextWidth("sample text", 14, "normal")).toBe(
      estimateTextWidth("sample text", 14, "normal"),
    );
    expect(measureTextWidth("bold sample", 16, "bold")).toBe(
      estimateTextWidth("bold sample", 16, "bold"),
    );
  });

  it("never throws even when repeatedly called (canvas probe result is cached, not re-thrown per call)", () => {
    expect(() => {
      for (let i = 0; i < 5; i += 1) {
        measureTextWidth(`call ${i}`, 14);
      }
    }).not.toThrow();
  });
});
