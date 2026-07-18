// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TextDrawCommand } from "../../display-list.js";

/** Font properties accepted by Canvas text measurement. */
export type TextAtlasFont = TextDrawCommand["font"];

/** Minimal Canvas context required to measure text. */
export type TextMeasurementContext = Pick<
  CanvasRenderingContext2D,
  "font" | "measureText"
>;

/** Stable subset of browser text metrics used by the Canvas backend. */
export type CanvasTextMeasurement = Readonly<{
  width: number;
  actualBoundingBoxAscent: number;
  actualBoundingBoxDescent: number;
}>;

const UNQUOTED_FAMILY = /^(?:[\w-]+|serif|sans-serif|monospace|cursive|fantasy|system-ui)$/u;

function quotedFamily(family: string): string {
  return UNQUOTED_FAMILY.test(family)
    ? family
    : `"${family.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

/** Converts display-list font properties into one canonical Canvas font string. */
export function canvasFont(font: TextAtlasFont): string {
  const family = font.family.trim();
  if (family.length === 0) {
    throw new RangeError("font family must not be empty");
  }
  if (!Number.isFinite(font.sizePx) || font.sizePx <= 0) {
    throw new RangeError("font size must be a positive finite number");
  }
  if (
    font.weight !== undefined &&
    (!Number.isFinite(font.weight) || font.weight <= 0)
  ) {
    throw new RangeError("font weight must be a positive finite number");
  }

  const weight = font.weight === undefined ? "" : `${font.weight} `;
  return `${weight}${font.sizePx}px ${quotedFamily(family)}`;
}

function finiteMeasurement(metrics: TextMetrics): CanvasTextMeasurement {
  const measurement = {
    width: metrics.width,
    actualBoundingBoxAscent: metrics.actualBoundingBoxAscent,
    actualBoundingBoxDescent: metrics.actualBoundingBoxDescent,
  };
  if (Object.values(measurement).some((value) => !Number.isFinite(value))) {
    throw new RangeError("Canvas text measurement must be finite");
  }
  return Object.freeze(measurement);
}

/**
 * Memoizes normalized Canvas text metrics by canonical font and exact text.
 *
 * The atlas deliberately excludes browser timing and insertion-order-dependent
 * eviction so identical inputs retain identical measurements for its lifetime.
 */
export class CanvasTextAtlas {
  readonly #context: TextMeasurementContext;
  readonly #measurements = new Map<string, CanvasTextMeasurement>();

  public constructor(context: TextMeasurementContext) {
    this.#context = context;
  }

  /** Number of unique text and font tuples currently cached. */
  public get size(): number {
    return this.#measurements.size;
  }

  /** Returns the cached deterministic measurement for a text and font tuple. */
  public measure(
    text: string,
    font: TextAtlasFont,
  ): CanvasTextMeasurement {
    const fontString = canvasFont(font);
    this.#context.font = fontString;
    const key = JSON.stringify([fontString, text]);
    const cached = this.#measurements.get(key);
    if (cached !== undefined) {
      return cached;
    }

    const measurement = finiteMeasurement(this.#context.measureText(text));
    this.#measurements.set(key, measurement);
    return measurement;
  }

  /** Removes every cached measurement. */
  public clear(): void {
    this.#measurements.clear();
  }
}
