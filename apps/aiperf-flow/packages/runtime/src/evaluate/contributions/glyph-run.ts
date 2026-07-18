// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Bounds, DrawCommand, Point } from "../../display-list.js";
import type { SemanticEntityProjection } from "../types.js";
import { measureGlyphRun } from "../../leaves/glyph-measure.js";

export type GlyphRunFont = Readonly<{
  family: string;
  sizePx: number;
  weight?: number;
}>;

export type GlyphRunSemantics = Readonly<{
  role: string;
  description?: string;
}>;

export type GlyphRunContributionInput = Readonly<{
  id: string;
  text: string;
  bounds: Bounds;
  origin: Point;
  font: GlyphRunFont;
  fill?: string;
  order?: number;
  locale?: string;
  semantics?: GlyphRunSemantics;
}>;

export type GlyphRunContribution = Readonly<{
  commands: readonly DrawCommand[];
  semanticEntities?: readonly SemanticEntityProjection[];
}>;

/** Emits backend-neutral display-list commands for a measured glyph run. */
export function contributeGlyphRun(
  input: GlyphRunContributionInput,
): GlyphRunContribution {
  const order = input.order ?? 0;
  const command: DrawCommand = {
    kind: "text",
    id: `${input.id}:text`,
    order,
    paintBounds: input.bounds,
    damageBounds: input.bounds,
    text: input.text,
    origin: input.origin,
    font: input.font,
    ...(input.fill === undefined ? {} : { fill: input.fill }),
  };

  if (input.semantics === undefined) {
    return { commands: [command] };
  }

  const measured = measureGlyphRun(input.id, input.text, input.locale ?? "en");
  const semanticEntities: SemanticEntityProjection[] = measured.graphemes.map(
    (grapheme) => ({
      id: `${input.id}:${grapheme.id}`,
      label: grapheme.text,
      role: input.semantics!.role,
      ...(input.semantics!.description === undefined
        ? {}
        : { description: input.semantics!.description }),
    }),
  );

  return { commands: [command], semanticEntities };
}
