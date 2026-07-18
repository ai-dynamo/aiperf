/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower explainer slides into `SlidePackage[]`, including embedded `@scene`
//! bodies via `lowerExplainerScene`.

import type { SlideAst } from "@aiperf/flow-language";
import {
  diagnostic,
  hasErrors,
  type Diagnostic,
  type Result,
  type SlidePackage,
  type SourceRange,
} from "@aiperf/flow-schema";

import { lowerExplainerScene } from "./lower-explainer-scene.js";

/** Slide AST that may already carry an authored `id` (grammar extension). */
export type SlideTextAst = SlideAst & Readonly<{ id?: string }>;

function unknownRange(): SourceRange {
  return {
    source: "<unknown>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function rangeOf(ast: { sourceMap?: SourceRange }): SourceRange {
  return ast.sourceMap ?? unknownRange();
}

/** Builds a stable slide id from the slide title (or index fallback). */
export function slideIdFromTitle(title: string, index: number): string {
  const slug = title
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.length > 0 ? slug : `slide-${index}`;
}

function resolveSlideId(
  slide: SlideTextAst,
  title: string,
  index: number,
): string {
  const authored = slide.id?.trim() ?? "";
  if (authored.length > 0) {
    return authored;
  }
  return slideIdFromTitle(title || slide.title || `slide-${index}`, index);
}

function requireNonEmpty(
  value: string | undefined,
  field: string,
  range: SourceRange,
  diagnostics: Diagnostic[],
): string {
  const trimmed = value?.trim() ?? "";
  if (trimmed.length === 0) {
    diagnostics.push(
      diagnostic(
        "EXPLAINER_SLIDE_FIELD_REQUIRED",
        "error",
        `${field} is required and cannot be empty`,
        range,
      ),
    );
    return "";
  }
  return trimmed;
}

function lowerSlideText(
  slide: SlideTextAst,
  index: number,
  diagnostics: Diagnostic[],
): SlidePackage {
  const range = rangeOf(slide);
  const title = requireNonEmpty(
    slide.title,
    `slides[${index}].title`,
    range,
    diagnostics,
  );
  const narration = requireNonEmpty(
    slide.narration,
    `slides[${index}].narration`,
    range,
    diagnostics,
  );

  const id = resolveSlideId(slide, title, index);
  let pkg: SlidePackage = {
    id,
    eyebrow: slide.eyebrow ?? "",
    title,
    lede: slide.lede ?? "",
    narration,
    points: [...(slide.points ?? [])],
    caption: slide.caption ?? "",
  };

  if (slide.term !== undefined) {
    pkg = {
      ...pkg,
      term: { word: slide.term.word, meaning: slide.term.meaning },
    };
  }

  if (slide.sceneIr !== undefined) {
    const sceneResult = lowerExplainerScene(slide.sceneIr, {
      slideId: id,
      defaults: {
        id: `scene-${id}`,
        title: title || id,
        summary: slide.lede || title || id,
        narration,
        fallback: slide.caption || title || id,
      },
    });
    if (!sceneResult.ok) {
      diagnostics.push(...sceneResult.diagnostics);
    } else {
      pkg = { ...pkg, render: sceneResult.value };
    }
  }

  return pkg;
}

/**
 * Lowers explainer slides into `SlidePackage[]`, including optional scene
 * `render` payloads from embedded `@scene` AST.
 *
 * Rejects empty (or whitespace-only) `title` and `narration`. Prefer an
 * authored slide `id` when present; otherwise derive one from the title.
 */
export function lowerExplainerSlides(
  slides: readonly SlideTextAst[],
): Result<readonly SlidePackage[]> {
  const diagnostics: Diagnostic[] = [];
  const packages = slides.map((slide, index) =>
    lowerSlideText(slide, index, diagnostics),
  );

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }

  return { ok: true, value: packages, diagnostics };
}
