// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type GlyphUnitIr = Readonly<{
  id: string;
  text: string;
  byteStart: number;
  byteEnd: number;
  charStart: number;
  charEnd: number;
}>;

export type GlyphMeasureResult = Readonly<{
  runId: string;
  text: string;
  graphemes: readonly GlyphUnitIr[];
}>;

/** Measures grapheme boundaries for a glyph run using Intl.Segmenter. */
export function measureGlyphRun(
  runId: string,
  text: string,
  locale = "en",
): GlyphMeasureResult {
  const segmenter = new Intl.Segmenter(locale, { granularity: "grapheme" });
  const graphemes: GlyphUnitIr[] = [];
  let index = 0;

  for (const segment of segmenter.segment(text)) {
    const segmentText = segment.segment;
    const byteStart = byteOffset(text, segment.index);
    const charStart = segment.index;
    const charEnd = charStart + segmentText.length;
    const byteEnd = byteOffset(text, charEnd);

    graphemes.push({
      id: `g${index}`,
      text: segmentText,
      byteStart,
      byteEnd,
      charStart,
      charEnd,
    });
    index += 1;
  }

  return { runId, text, graphemes };
}

function byteOffset(text: string, charIndex: number): number {
  return new TextEncoder().encode(text.slice(0, charIndex)).length;
}
