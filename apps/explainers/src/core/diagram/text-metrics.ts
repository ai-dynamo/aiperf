/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic, scale-aware text metrics for scene layout and rendering.

export const SCENE_TEXT_SCALE = 0.9;

/**
 * Factor the scene world was scaled by when the canvas moved from the pre-4K
 * 700x400 baseline to 1920x1080. Deck-authored `fontSize=` values were migrated
 * by `scripts/rescale-decks-to-16x9.mjs` (same constant), and the spacing
 * constants below were migrated by hand — but the renderer-owned chrome font
 * sizes in `capabilities/chrome.ts` were missed, so titles rendered at the old
 * 14px baseline inside boxes padded for the new world. `SCENE_FONT` exists so
 * renderer-owned sizes live beside the geometry they have to stay in scale with.
 */
export const SCENE_WORLD_SCALE = 2.7;

/** 2.7x the pre-4K baseline (700x400 -> 1920x1080); keep in sync with SceneRenderer's VIEWPORT_WIDTH/HEIGHT. */
export const DEFAULT_SCENE_FONT_SIZE = 37.8;

/**
 * Renderer-owned chrome font sizes, in scene-world units (pre-`SCENE_TEXT_SCALE`).
 * Each is its pre-4K baseline times `SCENE_WORLD_SCALE`, so the ladder stays
 * proportional to `INSET` / `TITLE_HEIGHT` / `DETAIL_HEIGHT` below.
 */
export const SCENE_FONT = {
  /** 14 — panel/card titles, and the fallback for un-sized text parts. */
  title: 37.8,
  /** 13 — `core.header` and `diagram.*` titles. */
  titleCompact: 35.1,
  /** 12 — `diagram.boundary` titles, quote/code blocks, icon labels. */
  body: 32.4,
  /** 11.5 — panel/card detail lines. */
  detail: 31.05,
  /** 11 — `core.chip` labels and stepper step labels. */
  chip: 29.7,
  /** 10 — subtitles and `diagram.*` detail lines. */
  caption: 27,
} as const;
export const CHAR_WIDTH = 6.2;
export const BOLD_CHAR_WIDTH = 6.2;
export const INSET = 21.6;
export const TITLE_HEIGHT = 59.4;
export const DETAIL_HEIGHT = 54;
export const SUBTITLE_HEIGHT = 43.2;
export const CHIP_PAD_X = 64.8;
export const STEPPER_MIN_CHIP_WIDTH = 194.4;
export const STEPPER_CHIP_HEIGHT = 70.2;
export const STEPPER_CHIP_PAD = 64.8;

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

/** Font family scene text renders with, kept in sync with `index.css`'s `font-family`. */
const SCENE_FONT_FAMILY =
  'Manrope, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

let canvasMeasureContext: CanvasRenderingContext2D | null | undefined;

/**
 * Lazily creates (and caches) a canvas 2D context for real glyph-metric text
 * measurement, feature-testing it first: jsdom's canvas shim (when the
 * optional native `canvas` package isn't installed) accepts `measureText`
 * calls but always reports a width of `0`, which would silently make every
 * wrap decision collapse to one word per line. If the probe measurement
 * isn't a plausible positive width, this returns `null` and callers fall
 * back to the deterministic character-width estimate — this is what keeps
 * test/SSR/CI environments deterministic while a real browser gets accurate
 * glyph metrics instead of a per-character guess.
 */
function getCanvasMeasureContext(): CanvasRenderingContext2D | null {
  if (canvasMeasureContext !== undefined) {
    return canvasMeasureContext;
  }
  canvasMeasureContext = null;
  try {
    if (typeof document === "undefined") {
      return canvasMeasureContext;
    }
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (ctx === null || typeof ctx.measureText !== "function") {
      return canvasMeasureContext;
    }
    const probeWidth = ctx.measureText("metrics probe").width;
    if (!(probeWidth > 0)) {
      return canvasMeasureContext;
    }
    canvasMeasureContext = ctx;
  } catch {
    canvasMeasureContext = null;
  }
  return canvasMeasureContext;
}

/**
 * Measures `text`'s real rendered width via canvas glyph metrics when a
 * working canvas context is available (a real browser), falling back to
 * `estimateTextWidth`'s deterministic character-count model otherwise. The
 * canvas path applies the same `SCENE_TEXT_SCALE` factor `estimateTextWidth`
 * already bakes in, so callers see one consistent unit convention regardless
 * of which measurement backend actually served the call.
 */
export function measureTextWidth(
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): number {
  const ctx = getCanvasMeasureContext();
  if (ctx === null) {
    return estimateTextWidth(text, fontSize, weight);
  }
  ctx.font = `${weight === "bold" ? "700" : "400"} ${fontSize}px ${SCENE_FONT_FAMILY}`;
  return Math.ceil(ctx.measureText(text).width * SCENE_TEXT_SCALE);
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
 * `maxWidth`, breaking only between words (never mid-word). A single word
 * wider than `maxWidth` on its own still occupies its own line rather than
 * being split or dropped. Uses `measureTextWidth` (real canvas glyph metrics
 * when available, the deterministic character-count estimate otherwise), so
 * wrap decisions are as accurate as the runtime environment allows while
 * staying deterministic in tests/SSR.
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
    if (measureTextWidth(candidate, fontSize, weight) <= maxWidth) {
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
