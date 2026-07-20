/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic, scale-aware text metrics for scene layout and rendering.

export const SCENE_TEXT_SCALE = 0.9;
export const DEFAULT_SCENE_FONT_SIZE = 14;
export const CHAR_WIDTH = 6.2;
export const BOLD_CHAR_WIDTH = 6.2;
export const INSET = 8;
export const TITLE_HEIGHT = 22;
export const DETAIL_HEIGHT = 20;
export const SUBTITLE_HEIGHT = 16;
export const CHIP_PAD_X = 24;
export const STEPPER_MIN_CHIP_WIDTH = 72;
export const STEPPER_CHIP_HEIGHT = 26;
export const STEPPER_CHIP_PAD = 24;

export function scaledSceneFontSize(
  value: unknown,
  fallback = DEFAULT_SCENE_FONT_SIZE,
): number {
  const fontSize =
    typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return fontSize * SCENE_TEXT_SCALE;
}

export function estimateTextWidth(
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): number {
  const unit = weight === "bold" ? BOLD_CHAR_WIDTH : CHAR_WIDTH;
  const ratio = fontSize / 11;
  return Math.ceil(text.length * unit * ratio * SCENE_TEXT_SCALE);
}

export function stepperChipWidth(label: string, index: number): number {
  const text = `${index + 1}. ${label}`;
  return Math.max(
    STEPPER_MIN_CHIP_WIDTH,
    estimateTextWidth(text, 11, "bold") + STEPPER_CHIP_PAD,
  );
}

/**
 * Greedy word-wrap: packs whitespace-separated words onto lines that fit
 * `maxWidth` per `estimateTextWidth`, breaking only between words (never
 * mid-word). A single word wider than `maxWidth` on its own still occupies
 * its own line rather than being split or dropped.
 */
export function wrapTextToWidth(
  text: string,
  maxWidth: number,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): string[] {
  const words = text.split(/\s+/).filter((word) => word.length > 0);
  if (words.length === 0) {
    return [];
  }

  const lines: string[] = [];
  let current = words[0];

  for (let i = 1; i < words.length; i += 1) {
    const word = words[i];
    const candidate = `${current} ${word}`;
    if (estimateTextWidth(candidate, fontSize, weight) <= maxWidth) {
      current = candidate;
    } else {
      lines.push(current);
      current = word;
    }
  }
  lines.push(current);
  return lines;
}

/**
 * Line-height multiple applied to the (scaled) font size when stacking wrapped
 * lines. Mirrors `SceneRenderer`'s `fontSize * 1.3` default so expand-time box
 * auto-grow and render-time line stacking stay in agreement.
 */
export const SCENE_LINE_HEIGHT_RATIO = 1.3;

/**
 * Expand-time height a wrapped text block occupies inside `maxWidth`, in the
 * same pixel space as scene geometry. `fontSize` is the authored (unscaled)
 * value; scaling replicates `SceneRenderer` exactly (`scaledSceneFontSize`
 * feeds both the wrap measurement and the line height) so the computed line
 * count matches what the renderer draws. Returns `0` for empty text.
 */
export function measuredWrappedHeight(
  text: string,
  maxWidth: number,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): number {
  const scaled = scaledSceneFontSize(fontSize);
  const lineCount = wrapTextToWidth(text, maxWidth, scaled, weight).length;
  return lineCount * scaled * SCENE_LINE_HEIGHT_RATIO;
}
