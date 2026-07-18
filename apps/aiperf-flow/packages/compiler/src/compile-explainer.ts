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

import { parseDocument, type DocumentAst } from "@aiperf/flow-language";
import {
  diagnostic,
  safeParseDeckPackage,
  type CapabilityRegistryManifest,
  type DeckPackage,
  type Result,
  type SourceRange,
} from "@aiperf/flow-schema";

import {
  lowerExplainerToDeckPackage,
  type ExplainerLowerInput,
} from "./lower-explainer.js";
import { validateExplainerTimelines } from "./validate-explainer-timelines.js";

/** A single request to compile explainer `.flow` source into a DeckPackage. */
export type CompileExplainerRequest = Readonly<{
  source: string;
  sourceName: string;
  capabilities: CapabilityRegistryManifest;
  strict: boolean;
}>;

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
 * `capabilities` and `strict` mirror `compileSource` for future capability
 * gating on diagram nodes.
 */
export function compileExplainerSource(
  request: CompileExplainerRequest,
): Result<DeckPackage> {
  void request.capabilities;
  void request.strict;

  const parsed = parseDocument(request.source, request.sourceName);
  if (!parsed.ok) {
    return parsed;
  }

  const explainer = extractExplainer(parsed.value, request.sourceName);
  if (!explainer.ok) {
    return explainer;
  }

  const lowered = lowerExplainerToDeckPackage(explainer.value);
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

  return {
    ok: true,
    value: packaged.value,
    diagnostics: [
      ...parsed.diagnostics,
      ...explainer.diagnostics,
      ...lowered.diagnostics,
      ...timelines.diagnostics,
      ...packaged.diagnostics,
    ],
  };
}
