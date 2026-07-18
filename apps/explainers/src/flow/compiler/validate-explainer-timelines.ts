/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fail-closed validation that every diagram slide carries non-empty roots
//! and timeline cues.
//!
//! Explainer decks may include text-only slides without `render`. Any slide
//! that mounts a `render.kind === "scene"` diagram must emit at least one
//! `scene.roots` node and drive motion through Flow timeline cues; empty
//! roots or timelines are rejected so silent no-ops cannot sneak back in.

import {
  diagnostic,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
  type Result,
} from "../schema/index.js";

/** Rejects scene-rendered explainer slides that lack timeline cues. */
export function validateExplainerTimelines(
  deck: DeckPackage,
): Result<DeckPackage> {
  const diagnostics: Diagnostic[] = [];

  for (const slide of deck.slides) {
    if (slide.render?.kind !== "scene") {
      continue;
    }
    if (slide.render.scene.roots.length === 0) {
      diagnostics.push(
        diagnostic(
          "EXPLAINER_SCENE_ROOTS_REQUIRED",
          "error",
          `Slide "${slide.id}" has render.kind "scene" but scene.roots is empty (fail-closed).`,
          slide.render.scene.sourceMap,
          "Lower embedded @scene roots into at least one diagram node.",
        ),
      );
    }
    if (slide.render.scene.timeline.length === 0) {
      diagnostics.push(
        diagnostic(
          "EXPLAINER_TIMELINE_REQUIRED",
          "error",
          `Slide "${slide.id}" has render.kind "scene" but scene.timeline is empty (fail-closed).`,
          slide.render.scene.sourceMap,
          "Add at least one timeline cue that drives enter, draw, or emphasis motion.",
        ),
      );
    }
  }

  if (deck.finalCard?.kind === "scene") {
    if (deck.finalCard.scene.roots.length === 0) {
      diagnostics.push(
        diagnostic(
          "EXPLAINER_SCENE_ROOTS_REQUIRED",
          "error",
          `Slide "finalCard" has render.kind "scene" but scene.roots is empty (fail-closed).`,
          deck.finalCard.scene.sourceMap,
          "Lower embedded @scene roots into at least one diagram node.",
        ),
      );
    }
    if (deck.finalCard.scene.timeline.length === 0) {
      diagnostics.push(
        diagnostic(
          "EXPLAINER_TIMELINE_REQUIRED",
          "error",
          `Slide "finalCard" has render.kind "scene" but scene.timeline is empty (fail-closed).`,
          deck.finalCard.scene.sourceMap,
          "Add at least one timeline cue that drives enter, draw, or emphasis motion.",
        ),
      );
    }
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }

  return { ok: true, value: deck, diagnostics };
}
