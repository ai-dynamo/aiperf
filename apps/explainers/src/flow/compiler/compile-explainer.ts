/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explainer `.flow` compile pipeline: parse → lower (incl. `@scene`) →
//! validate timelines → schema.
//!
//! `compileExplainerSource` is the DeckPackage entry point parallel to
//! `compileSource` for Flow IR. Stages short-circuit on the first failure so
//! callers see the earliest actionable diagnostics. Empty `scene.roots` /
//! `scene.timeline` fail closed with slide-id diagnostics from lowering or
//! `validateExplainerTimelines`.

import { parseDocument, type DocumentAst } from "../language/index.js";
import {
  diagnostic,
  safeParseDeckPackage,
  type DeckPackage,
  type Result,
  type SourceRange,
  type JsonValue,
} from "../schema/index.js";

import {
  lowerExplainerToDeckPackage,
  type ExplainerLowerInput,
  type ExplainerLowerOptions,
} from "./lower-explainer.js";
import {
  validateSdkAuthoring,
  type SdkAuthoringPolicy,
  type SdkAuthoringSceneInput,
} from "./validate-sdk-authoring.js";
import { validateExplainerTimelines } from "./validate-explainer-timelines.js";

/** Canonical validation policy for explainer compile / lower (re-export). */
export type ExplainerCompileOptions = ExplainerLowerOptions;

/** A single request to compile explainer `.flow` source into a DeckPackage. */
export type CompileExplainerRequest = ExplainerCompileOptions &
  Readonly<{
    source: string;
    sourceName: string;
    /**
     * Strict SDK-authoring gate control.
     *
     * Enforcement is phased so package-form decks keep compiling until they
     * migrate to native `sdk.*` authoring:
     *
     * - `true`: fail compilation on any prohibited bespoke / package-form
     *   signature (the final migrated `strict: true` build).
     * - `false`: skip the gate entirely.
     * - `undefined` (default): report prohibited signatures as warnings without
     *   failing, keeping the pre-migration corpus compilable.
     */
    strictSdkAuthoring?: boolean;
  }>;

/** Resolves the phased SDK-authoring policy from the compile request. */
function resolveSdkAuthoringPolicy(
  request: CompileExplainerRequest,
): SdkAuthoringPolicy {
  if (request.strictSdkAuthoring === true) {
    return "strict";
  }
  if (request.strictSdkAuthoring === false) {
    return "off";
  }
  return "report";
}

/**
 * Pairs each authored scene (raw AST) with its lowered `SceneIr` so the strict
 * authoring gate can detect package-form scenes and bespoke signatures using
 * both source structure and compiler-only SDK provenance.
 */
function collectSdkAuthoringScenes(
  explainer: ExplainerLowerInput,
  pkg: DeckPackage,
): SdkAuthoringSceneInput[] {
  const scenes: SdkAuthoringSceneInput[] = [];
  explainer.slides.forEach((slide, index) => {
    if (slide.sceneIr === undefined) {
      return;
    }
    const lowered = pkg.slides[index];
    scenes.push({
      slideId: lowered?.id ?? `slide-${index}`,
      rawScene: slide.sceneIr,
      scene: lowered?.render?.scene,
    });
  });
  if (explainer.finalCard !== undefined) {
    scenes.push({
      slideId: "finalCard",
      rawScene: explainer.finalCard,
      scene: pkg.finalCard?.scene,
    });
  }
  return scenes;
}

function unknownRange(sourceName: string): SourceRange {
  return {
    source: sourceName,
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function extractExplainer(
  document: DocumentAst,
  sourceName: string,
): Result<ExplainerLowerInput> {
  const explainers = document.explainers ?? [];
  if (explainers.length === 0) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "EXPLAINER_REQUIRED",
          "error",
          "Expected a top-level explainer document.",
          document.sourceMap ?? unknownRange(sourceName),
        ),
      ],
    };
  }
  if (explainers.length > 1) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "EXPLAINER_SINGLE_REQUIRED",
          "error",
          `Expected exactly one explainer block, found ${explainers.length}.`,
          document.sourceMap ?? unknownRange(sourceName),
        ),
      ],
    };
  }

  // ExplainerLowerInput accepts DeckPackage metadata fields that the language
  // AST is absorbing; parsed explainers already carry them when grammar is wired.
  return {
    ok: true,
    value: explainers[0] as ExplainerLowerInput,
    diagnostics: [],
  };
}

/**
 * Compiles explainer `.flow` source into a validated `DeckPackage`.
 *
 * Pipeline: parseDocument → lowerExplainerToDeckPackage (slide text +
 * embedded `@scene` → SceneIr) → validateExplainerTimelines →
 * safeParseDeckPackage.
 *
 * `capabilities` and `strict` are applied to every embedded scene.
 */
export function compileExplainerSource(
  request: CompileExplainerRequest,
): Result<DeckPackage> {
  const parsed = parseDocument(request.source, request.sourceName);
  if (!parsed.ok) {
    return parsed;
  }

  const explainer = extractExplainer(parsed.value, request.sourceName);
  if (!explainer.ok) {
    return explainer;
  }

  const tokens = new Map<string, JsonValue>(
    parsed.value.tokens.map((token) => [token.id, token.value.value]),
  );

  const lowered = lowerExplainerToDeckPackage(explainer.value, {
    capabilities: request.capabilities,
    strict: request.strict,
    sourceName: request.sourceName,
    tokens,
  });
  if (!lowered.ok) {
    return lowered;
  }

  const timelines = validateExplainerTimelines(lowered.value);
  if (!timelines.ok) {
    return timelines;
  }

  const packaged = safeParseDeckPackage(timelines.value);
  if (!packaged.ok) {
    return packaged;
  }

  const authoringPolicy = resolveSdkAuthoringPolicy(request);
  const authoring = validateSdkAuthoring(
    collectSdkAuthoringScenes(explainer.value, packaged.value),
    { policy: authoringPolicy },
  );

  const diagnostics = [
    ...parsed.diagnostics,
    ...explainer.diagnostics,
    ...lowered.diagnostics,
    ...timelines.diagnostics,
    ...packaged.diagnostics,
    ...authoring.diagnostics,
  ];

  if (!authoring.ok) {
    return { ok: false, diagnostics };
  }

  return {
    ok: true,
    value: packaged.value,
    diagnostics,
  };
}
