/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fail-closed validation that every diagram slide carries a non-empty timeline.
//!
//! Explainer decks may include text-only slides without `render`. Any slide
//! that mounts a `render.kind === "scene"` diagram must drive motion through
//! Flow timeline cues; empty timelines are rejected so legacy CSS/SVG motion
//! cannot sneak back in through silent no-ops.

import {
  diagnostic,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
  type Result,
} from "@aiperf/flow-schema";

/** Rejects scene-rendered explainer slides that lack timeline cues. */
export function validateExplainerTimelines(
  deck: DeckPackage,
): Result<DeckPackage> {
  const diagnostics: Diagnostic[] = [];

  for (const slide of deck.slides) {
    if (slide.render?.kind !== "scene") {
      continue;
    }
    if (slide.render.scene.timeline.length > 0) {
      continue;
    }
    diagnostics.push(
      diagnostic(
        "EXPLAINER_TIMELINE_REQUIRED",
        "error",
        `Slide "${slide.id}" has render.kind "scene" but scene.timeline is empty.`,
        slide.render.scene.sourceMap,
        "Add at least one timeline cue that drives enter, draw, or emphasis motion.",
      ),
    );
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }

  return { ok: true, value: deck, diagnostics };
}
