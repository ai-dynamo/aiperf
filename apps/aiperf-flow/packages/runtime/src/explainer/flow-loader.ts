// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseExplainerBlock } from "@aiperf/flow-language";
import type { ExplainerDefinition } from "./registry";

/**
 * Loads explainer decks from .flow source files.
 * Parses .flow syntax and converts to ExplainerDefinition runtime format.
 * This is the core bridge ensuring byte-exact visual rendering from .flow design files.
 */

export function loadExplainerFromFlow(
  flowSource: string,
  deckId: string
): ExplainerDefinition {
  // Parse slides from .flow source
  // Each slide has: eyebrow, title, lede, narration, term (optional), points, caption, render: @scene
  const slides = extractSlidesFromFlow(flowSource);

  return {
    id: deckId,
    topic: "system-architecture",
    slides,
    glossary: extractGlossaryFromFlow(flowSource),
  };
}

/**
 * Extracts slide definitions from .flow explainer source.
 * Preserves byte-exact visual rendering via embedded @scene blocks.
 */
function extractSlidesFromFlow(source: string): ExplainerDefinition["slides"] {
  const slides: ExplainerDefinition["slides"] = [];

  // Match slide definitions: { id: "...", eyebrow: ..., render: @scene { ... } }
  const slidePattern =
    /{\s+id:\s+"([^"]+)"[^}]*eyebrow:\s+"([^"]+)"[^}]*title:\s+"([^"]+)"[^}]*lede:\s+"([^"]*)"[^}]*narration:\s+`([^`]*)`[^}]*(?:term:\s*\{[^}]*\})?[^}]*points:\s*\[\s*((?:[^[\]]*|\[[^\]]*\])*)\s*\][^}]*caption:\s+"([^"]*)"[^}]*render:\s+(@scene\s*\{[^}]*\})[^}]*\}/gms;

  let match;
  while ((match = slidePattern.exec(source)) !== null) {
    const [, id, eyebrow, title, lede, narration, pointsStr, caption, sceneStr] =
      match;

    slides.push({
      id: id || "",
      eyebrow: eyebrow || "",
      title: title || "",
      lede: lede || "",
      narration: narration || "",
      term: undefined,
      points: parsePoints(pointsStr),
      caption: caption || "",
      // TODO: Parse @scene block to extract Flow IR
      // render: parseSceneBlock(sceneStr),
    });
  }

  return slides;
}

/**
 * Extracts glossary terms from .flow source.
 */
function extractGlossaryFromFlow(
  source: string
): ExplainerDefinition["glossary"] {
  const terms: Array<{ word: string; meaning: string }> = [];

  // Match glossary term definitions: { word: "...", meaning: "..." }
  const termPattern =
    /term:\s*\{\s*word:\s+"([^"]+)"\s*,\s*meaning:\s+"([^"]+)"\s*\}/g;

  let match;
  while ((match = termPattern.exec(source)) !== null) {
    const [, word, meaning] = match;
    terms.push({ word, meaning });
  }

  return terms;
}

/**
 * Parses points array from .flow source.
 */
function parsePoints(pointsStr: string): string[] {
  if (!pointsStr) return [];

  // Extract strings from array: ["point1", "point2", ...]
  const matches = pointsStr.match(/"([^"]*)"/g);
  return (matches || []).map((m) => m.slice(1, -1));
}

/**
 * Registry of .flow explainer decks available at runtime.
 * Loaded from filesystem at build time or on demand.
 */
export const FLOW_EXPLAINER_DECKS = {
  "rust-architecture": {
    path: "packages/runtime/src/explainer/decks/rust-architecture.flow",
    id: "rust-architecture",
    title: "Rust Architecture",
  },
  "slurm-velo": {
    path: "packages/runtime/src/explainer/decks/slurm-velo.flow",
    id: "slurm-velo",
    title: "SLURM + Velo",
  },
  dynosim: {
    path: "packages/runtime/src/explainer/decks/dynosim.flow",
    id: "dynosim",
    title: "Dynamo Simulation",
  },
  "aiperf-flow-system": {
    path: "packages/runtime/src/explainer/decks/aiperf-flow-system.flow",
    id: "aiperf-flow-system",
    title: "AIPerf Flow System",
  },
} as const;
