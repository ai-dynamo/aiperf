// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SlideAst } from "../ast.js";

export interface ExplainerAstCompat {
  type: "explainer";
  id: string;
  metadata: {
    route: string;
    topic: string;
    eyebrowLabel: string;
    startGateTitle: string;
  };
  slides: SlideAst[];
}

export interface SlideAstCompat {
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: string[];
  caption: string;
  sceneIr?: any;
}

// Simplified TokenStream-like interface for validation/parsing
export interface TokenStream {
  expect(keyword: string): void;
  expectString(): string;
  expectIdentifier(): string;
  match(keyword: string): boolean;
  advance(): void;
}

/**
 * Parses an explainer block from token stream.
 *
 * Expected format:
 * explainer "deck-id" {
 *   route: "/path"
 *   topic: "topic"
 *   eyebrowLabel: "Label"
 *   startGateTitle: "Title"
 *
 *   slide "title" {
 *     eyebrow: "eyebrow"
 *     title: "title"
 *     lede: "lede"
 *     narration: "narration text"
 *     points: ["p1", "p2"]
 *     caption: "caption"
 *   }
 * }
 */
export function parseExplainerBlock(tokens: TokenStream): ExplainerAstCompat {
  // Expect: explainer STRING { metadata, slides }
  tokens.expect("explainer");
  const id = tokens.expectString();
  tokens.expect("{");

  const metadata = parseExplainerMetadata(tokens);
  const slides = parseSlides(tokens);

  tokens.expect("}");

  return { type: "explainer", id, metadata, slides };
}

function parseExplainerMetadata(
  tokens: TokenStream,
): {
  route: string;
  topic: string;
  eyebrowLabel: string;
  startGateTitle: string;
} {
  // Parse: route, topic, eyebrowLabel, startGateTitle
  const meta: any = {};
  while (!tokens.match("}") && !tokens.match("slide")) {
    const key = tokens.expectIdentifier();
    tokens.expect(":");
    meta[key] = tokens.expectString();
    if (tokens.match(",")) tokens.advance();
  }
  return meta;
}

function parseSlides(tokens: TokenStream): SlideAst[] {
  const slides: SlideAst[] = [];
  while (tokens.match("slide")) {
    slides.push(parseSlideBlock(tokens));
  }
  return slides;
}

function parseSlideBlock(tokens: TokenStream): SlideAst {
  // Expect: slide STRING { fields... }
  tokens.expect("slide");
  const titleString = tokens.expectString(); // slide title
  tokens.expect("{");

  const slide: any = { title: titleString };
  while (!tokens.match("}")) {
    const key = tokens.expectIdentifier();
    tokens.expect(":");

    if (key === "term") {
      // Nested object
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
      // Array of strings
      tokens.expect("[");
      const points: string[] = [];
      while (!tokens.match("]")) {
        points.push(tokens.expectString());
        if (tokens.match(",")) tokens.advance();
      }
      tokens.expect("]");
      slide.points = points;
    } else if (key === "render") {
      // @scene block
      tokens.expect("@");
      tokens.expect("scene");
      slide.sceneIr = parseSceneBlock(tokens); // delegate to scene parser
    } else {
      slide[key] = tokens.expectString();
    }

    if (tokens.match(",")) tokens.advance();
  }
  tokens.expect("}");

  // Validate narration is non-empty
  if (!slide.narration || slide.narration.trim() === "") {
    throw new Error("narration field is required and cannot be empty");
  }

  return slide as SlideAst;
}

function parseSceneBlock(_tokens: TokenStream): any {
  // Delegate to existing scene parser
  // For now, return a placeholder; will integrate with scene parser
  return { type: "scene" };
}
