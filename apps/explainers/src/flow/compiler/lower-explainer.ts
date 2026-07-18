/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower explainer AST into a DeckPackage, including slide `@scene` → SceneIr.

import type { ExplainerAst, SlideAst } from "../language/index.js";
import {
  diagnostic,
  hasErrors,
  safeParseDeckPackage,
  type DeckGlossaryEntry,
  type DeckHub,
  type DeckPackage,
  type Diagnostic,
  type CapabilityRegistryManifest,
  type JsonValue,
  type Result,
  type SceneRender,
  type SlidePackage,
  type SourceRange,
} from "../schema/index.js";
import { createSdkRegistry, type SdkRegistry } from "../sdk/index.js";

import { expandSdkInvocations } from "./expand-sdk.js";
import {
  lowerExplainerScene,
  type LowerExplainerSceneOptions,
} from "./lower-explainer-scene.js";

/** Deck packaging fields required by DeckPackage beyond today's ExplainerAst. */
export type ExplainerDeckMetadata = Readonly<{
  route: string;
  topic: string;
  storagePrefix: string;
  classPrefix: string;
  eyebrowLabel: string;
  startGateTitle: string;
  hub: DeckHub;
  css?: string | undefined;
}>;

/**
 * Explainer input for DeckPackage lowering.
 *
 * Extends `ExplainerAst` with `storagePrefix`, `classPrefix`, and `hub`
 * (and optional `css` / `glossary`) so packages validate before language AST
 * absorbs those fields.
 */
export type ExplainerLowerInput = Omit<ExplainerAst, "metadata"> &
  Readonly<{
    metadata: ExplainerDeckMetadata;
    glossary?: readonly DeckGlossaryEntry[] | undefined;
  }>;

/** Validation policy forwarded to every embedded explainer scene. */
export type ExplainerLowerOptions = Readonly<{
  capabilities: CapabilityRegistryManifest;
  strict: boolean;
  /** Owning `.flow` source name used for embedded `@scene` diagnostics. */
  sourceName?: string;
  /** Document-level token bindings for SDK prop resolution. */
  tokens?: ReadonlyMap<string, JsonValue>;
}>;

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

function sourceRangeForScene(
  ast: { sourceMap?: SourceRange },
  options: ExplainerLowerOptions,
): SourceRange {
  const local = ast.sourceMap;
  if (
    local !== undefined &&
    local.source !== "<unknown>" &&
    local.source !== "<embedded-scene>"
  ) {
    return local;
  }
  if (options.sourceName !== undefined && options.sourceName.length > 0) {
    return {
      source: options.sourceName,
      start: { offset: 0, line: 1, column: 1 },
      end: { offset: 0, line: 1, column: 1 },
    };
  }
  return local ?? unknownRange();
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
        "EXPLAINER_FIELD_REQUIRED",
        "error",
        `${field} is required and cannot be empty`,
        range,
      ),
    );
    return "";
  }
  return trimmed;
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

/**
 * Lowers an embedded `@scene`, expanding SDK component invocations first.
 *
 * When the scene invokes registered `sdk.*` / `aiperf.*` components, it is
 * expanded to ordinary Scene IR here (parse → symbols → SDK → SceneRender).
 * Scenes that invoke no SDK component fall back to the existing package /
 * native lowering path, so package-form decks compile unchanged until they
 * migrate to SDK authoring.
 */
function lowerSceneMaybeSdk(
  rawScene: unknown,
  registry: SdkRegistry,
  sceneOptions: LowerExplainerSceneOptions,
  tokens?: ReadonlyMap<string, JsonValue>,
): Result<SceneRender> {
  const outcome = expandSdkInvocations(rawScene, {
    registry,
    ...(tokens !== undefined ? { tokens } : {}),
    ...(sceneOptions.defaults !== undefined
      ? { defaults: sceneOptions.defaults }
      : {}),
    ...(sceneOptions.slideId !== undefined
      ? { slideId: sceneOptions.slideId }
      : {}),
    ...(sceneOptions.sourceRange !== undefined
      ? { sourceRange: sceneOptions.sourceRange }
      : {}),
  });
  if (outcome.status === "ok") {
    return {
      ok: true,
      value: outcome.value.render,
      diagnostics: outcome.diagnostics,
    };
  }
  if (outcome.status === "error") {
    return { ok: false, diagnostics: outcome.diagnostics };
  }
  return lowerExplainerScene(rawScene, sceneOptions);
}

function lowerSlide(
  slide: SlideAst,
  index: number,
  options: ExplainerLowerOptions,
  registry: SdkRegistry,
  diagnostics: Diagnostic[],
): SlidePackage {
  const range = rangeOf(slide);
  const title = requireNonEmpty(slide.title, `slides[${index}].title`, range, diagnostics);
  const narration = requireNonEmpty(
    slide.narration,
    `slides[${index}].narration`,
    range,
    diagnostics,
  );

  const id = slideIdFromTitle(title || slide.title || `slide-${index}`, index);
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
    const sceneResult = lowerSceneMaybeSdk(slide.sceneIr, registry, {
      slideId: id,
      defaults: {
        id: `scene-${id}`,
        title: title || id,
        summary: slide.lede || title || id,
        narration,
        fallback: slide.caption || title || id,
      },
      capabilities: options.capabilities,
      strict: options.strict,
      sourceRange: sourceRangeForScene(slide, options),
    }, options.tokens);
    if (!sceneResult.ok) {
      diagnostics.push(...sceneResult.diagnostics);
    } else {
      pkg = { ...pkg, render: sceneResult.value };
    }
  }

  return pkg;
}

function collectGlossary(
  slides: readonly SlideAst[],
  explicit: readonly DeckGlossaryEntry[] | undefined,
): DeckGlossaryEntry[] {
  if (explicit !== undefined) {
    return explicit.map((entry) => ({
      word: entry.word,
      meaning: entry.meaning,
    }));
  }

  const seen = new Set<string>();
  const glossary: DeckGlossaryEntry[] = [];
  for (const slide of slides) {
    if (slide.term === undefined) {
      continue;
    }
    const key = slide.term.word.trim().toLowerCase();
    if (key.length === 0 || seen.has(key)) {
      continue;
    }
    seen.add(key);
    glossary.push({ word: slide.term.word, meaning: slide.term.meaning });
  }
  return glossary;
}

/**
 * Maps explainer AST metadata, slide text, and embedded `@scene` bodies into
 * a DeckPackage (`schemaVersion: 1`).
 */
export function lowerExplainerToDeckPackage(
  ast: ExplainerLowerInput,
  options: ExplainerLowerOptions,
): Result<DeckPackage> {
  const diagnostics: Diagnostic[] = [];
  const range = rangeOf(ast);
  const meta = ast.metadata;
  const registry = createSdkRegistry();

  const id = requireNonEmpty(ast.id, "id", range, diagnostics);
  const route = requireNonEmpty(meta?.route, "metadata.route", range, diagnostics);
  const topic = requireNonEmpty(meta?.topic, "metadata.topic", range, diagnostics);
  const storagePrefix = requireNonEmpty(
    meta?.storagePrefix,
    "metadata.storagePrefix",
    range,
    diagnostics,
  );
  const classPrefix = requireNonEmpty(
    meta?.classPrefix,
    "metadata.classPrefix",
    range,
    diagnostics,
  );
  const eyebrowLabel = requireNonEmpty(
    meta?.eyebrowLabel,
    "metadata.eyebrowLabel",
    range,
    diagnostics,
  );
  const startGateTitle = requireNonEmpty(
    meta?.startGateTitle,
    "metadata.startGateTitle",
    range,
    diagnostics,
  );

  const hubTitle = requireNonEmpty(
    meta?.hub?.title,
    "metadata.hub.title",
    range,
    diagnostics,
  );
  const hubHighlight = requireNonEmpty(
    meta?.hub?.highlight,
    "metadata.hub.highlight",
    range,
    diagnostics,
  );
  const hubDescription = requireNonEmpty(
    meta?.hub?.description,
    "metadata.hub.description",
    range,
    diagnostics,
  );

  const slides = (ast.slides ?? []).map((slide, index) =>
    lowerSlide(slide, index, options, registry, diagnostics),
  );

  let finalCard: DeckPackage["finalCard"];
  if (ast.finalCard !== undefined) {
    const finalResult = lowerSceneMaybeSdk(ast.finalCard, registry, {
      slideId: "finalCard",
      defaults: {
        id: `scene-${id || "deck"}-final`,
        title: startGateTitle || hubTitle || id || "Final card",
        summary: hubDescription || startGateTitle || id || "Final card",
        narration: hubDescription || "",
        fallback: hubTitle || id || "Final card",
      },
      capabilities: options.capabilities,
      strict: options.strict,
      sourceRange: sourceRangeForScene(ast, options),
    }, options.tokens);
    if (!finalResult.ok) {
      diagnostics.push(...finalResult.diagnostics);
    } else {
      finalCard = finalResult.value;
    }
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }

  const candidate: DeckPackage = {
    schemaVersion: 1,
    id,
    route,
    topic,
    storagePrefix,
    classPrefix,
    eyebrowLabel,
    startGateTitle,
    hub: {
      title: hubTitle,
      highlight: hubHighlight,
      description: hubDescription,
    },
    slides,
    glossary: collectGlossary(ast.slides ?? [], ast.glossary),
    ...(finalCard === undefined ? {} : { finalCard }),
  };

  if (meta.css !== undefined) {
    return safeParseDeckPackage({ ...candidate, css: meta.css });
  }

  return safeParseDeckPackage(candidate);
}
