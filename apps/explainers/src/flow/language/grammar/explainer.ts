// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExplainerHubAst, SlideAst } from "../ast.js";
import {
  captureEmbeddedScene,
  parseEmbeddedSceneSource,
  type EmbeddedSceneSource,
  type PackageSceneIrAst,
  type PeekableTokenStream,
} from "../embedded-scene.js";

export interface ExplainerAstCompat {
  type: "explainer";
  id: string;
  metadata: {
    route: string;
    topic: string;
    storagePrefix: string;
    classPrefix: string;
    eyebrowLabel: string;
    startGateTitle: string;
    hub: ExplainerHubAst;
    css?: string;
  };
  slides: SlideAst[];
  /** Optional `finalCard: @scene { ... }` (package or native dialect). */
  finalCard?: EmbeddedSceneSource | PackageSceneIrAst;
}

export interface SlideAstCompat {
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: string[];
  caption: string;
  sceneIr?: EmbeddedSceneSource | PackageSceneIrAst;
}

/** Simplified TokenStream-like interface for validation/parsing. */
export interface TokenStream {
  expect(keyword: string): void;
  expectString(): string;
  expectIdentifier(): string;
  match(keyword: string): boolean;
  advance(): void;
  /** Required to capture `render: @scene { ... }` bodies. */
  peekImage(): string | undefined;
}

type ExplainerMetadata = ExplainerAstCompat["metadata"] & { id?: string };

/**
 * Parses an explainer block from token stream.
 *
 * Expected format:
 * explainer "Display Name" {
 *   id: "deck-id"
 *   route: "/path"
 *   topic: "topic"
 *   storagePrefix: "prefix"
 *   classPrefix: "class"
 *   eyebrowLabel: "Label"
 *   startGateTitle: "Title"
 *   hub: { title: "...", highlight: "...", description: "..." }
 *
 *   slide "title" {
 *     eyebrow: "eyebrow"
 *     title: "title"
 *     lede: "lede"
 *     narration: "narration text"
 *     points: ["p1", "p2"]
 *     caption: "caption"
 *     render: @scene { ... }
 *   }
 *
 *   finalCard: @scene { ... }
 * }
 */
export function parseExplainerBlock(tokens: TokenStream): ExplainerAstCompat {
  tokens.expect("explainer");
  const headerName = tokens.expectString();
  tokens.expect("{");

  const raw = parseExplainerMetadata(tokens);
  const slides = parseSlides(tokens);
  let finalCard: EmbeddedSceneSource | PackageSceneIrAst | undefined;
  if (tokens.match("finalCard")) {
    tokens.advance();
    tokens.expect(":");
    tokens.expect("@");
    tokens.expect("scene");
    finalCard = parseSceneBlock(tokens as PeekableTokenStream);
  }

  tokens.expect("}");

  const id = raw.id ?? headerName;
  const { id: _idField, ...metadataFields } = raw;
  void _idField;
  const metadata = requireExplainerMetadata(metadataFields);

  return {
    type: "explainer",
    id,
    metadata,
    slides,
    ...(finalCard === undefined ? {} : { finalCard }),
  };
}

function requireExplainerMetadata(
  meta: Partial<ExplainerAstCompat["metadata"]>,
): ExplainerAstCompat["metadata"] {
  const required = [
    "route",
    "topic",
    "storagePrefix",
    "classPrefix",
    "eyebrowLabel",
    "startGateTitle",
  ] as const;

  for (const key of required) {
    if (typeof meta[key] !== "string" || meta[key] === "") {
      throw new Error(`explainer metadata field "${key}" is required`);
    }
  }

  if (!meta.hub) {
    throw new Error('explainer metadata field "hub" is required');
  }

  return {
    route: meta.route!,
    topic: meta.topic!,
    storagePrefix: meta.storagePrefix!,
    classPrefix: meta.classPrefix!,
    eyebrowLabel: meta.eyebrowLabel!,
    startGateTitle: meta.startGateTitle!,
    hub: meta.hub,
    ...(typeof meta.css === "string" && meta.css.length > 0
      ? { css: meta.css }
      : {}),
  };
}

function parseExplainerMetadata(tokens: TokenStream): ExplainerMetadata {
  const meta: Partial<ExplainerMetadata> = {};

  while (!tokens.match("}") && !tokens.match("slide")) {
    const key = tokens.expectIdentifier();
    tokens.expect(":");

    if (key === "hub") {
      meta.hub = parseHubBlock(tokens);
    } else {
      (meta as Record<string, string>)[key] = tokens.expectString();
    }

    if (tokens.match(",")) tokens.advance();
  }

  return meta as ExplainerMetadata;
}

function parseHubBlock(tokens: TokenStream): ExplainerHubAst {
  tokens.expect("{");
  const hub: Partial<ExplainerHubAst> = {};

  while (!tokens.match("}")) {
    const key = tokens.expectIdentifier();
    tokens.expect(":");
    (hub as Record<string, string>)[key] = tokens.expectString();
    if (tokens.match(",")) tokens.advance();
  }

  tokens.expect("}");

  for (const key of ["title", "highlight", "description"] as const) {
    if (typeof hub[key] !== "string" || hub[key] === "") {
      throw new Error(`hub field "${key}" is required`);
    }
  }

  return {
    title: hub.title!,
    highlight: hub.highlight!,
    description: hub.description!,
  };
}

function parseSlides(tokens: TokenStream): SlideAst[] {
  const slides: SlideAst[] = [];
  while (tokens.match("slide")) {
    slides.push(parseSlideBlock(tokens));
  }
  return slides;
}

function parseSlideBlock(tokens: TokenStream): SlideAst {
  tokens.expect("slide");
  const titleString = tokens.expectString();
  tokens.expect("{");

  const slide: Record<string, unknown> = { title: titleString };
  while (!tokens.match("}")) {
    const key = tokens.expectIdentifier();
    tokens.expect(":");

    if (key === "term") {
      tokens.expect("{");
      const term = { word: "", meaning: "" };
      while (!tokens.match("}")) {
        const termKey = tokens.expectIdentifier();
        tokens.expect(":");
        term[termKey as keyof typeof term] = tokens.expectString();
        if (tokens.match(",")) tokens.advance();
      }
      tokens.expect("}");
      slide.term = term;
    } else if (key === "points") {
      tokens.expect("[");
      const points: string[] = [];
      while (!tokens.match("]")) {
        points.push(tokens.expectString());
        if (tokens.match(",")) tokens.advance();
      }
      tokens.expect("]");
      slide.points = points;
    } else if (key === "render") {
      tokens.expect("@");
      tokens.expect("scene");
      slide.sceneIr = parseSceneBlock(tokens as PeekableTokenStream);
    } else {
      slide[key] = tokens.expectString();
    }

    if (tokens.match(",")) tokens.advance();
  }
  tokens.expect("}");

  if (!slide.narration || String(slide.narration).trim() === "") {
    throw new Error("narration field is required and cannot be empty");
  }

  return slide as SlideAst;
}

/**
 * Parses `render: @scene { ... }` / `finalCard: @scene { ... }` so package
 * `roots`/`timeline` survive on the AST (`PackageSceneIrAst`), and native
 * cinematic bodies remain as capture source for `parseNativeEmbeddedScene`.
 */
export function parseSceneBlock(
  tokens: PeekableTokenStream,
): EmbeddedSceneSource | PackageSceneIrAst {
  const captured = captureEmbeddedScene(tokens);
  return parseEmbeddedSceneSource(captured);
}
