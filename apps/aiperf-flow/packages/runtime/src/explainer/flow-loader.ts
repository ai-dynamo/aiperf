// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseExplainerBlock } from "@aiperf/flow-language";
import type { ExplainerDefinition } from "./registry";

/**
 * Loads explainer decks from .flow source files.
 * Parses .flow syntax and converts to ExplainerDefinition runtime format.
 */

export async function loadExplainerFromFlow(
  flowSource: string,
  deckId: string
): Promise<ExplainerDefinition> {
  // For now, return a stub that bridges between .flow and ExplainerDefinition
  // In production, this would:
  // 1. Parse the .flow source with parseExplainerBlock
  // 2. Extract slides, metadata, scene definitions
  // 3. Convert to ExplainerDefinition format
  // 4. Validate against schema

  return {
    id: deckId,
    topic: "system-architecture",
    slides: [],
    glossary: [],
  };
}

/**
 * Converts parsed .flow explainer AST to ExplainerDefinition.
 * This is the byte-exact bridge from .flow design files to runtime rendering.
 */
export function flowToExplainerDefinition(
  flowAst: unknown,
  deckId: string
): ExplainerDefinition {
  // TODO: Implement full AST-to-Definition conversion
  // This requires:
  // 1. Extracting slide definitions from flowAst
  // 2. Mapping @scene blocks to scene IR
  // 3. Preserving all theme colors and typography
  // 4. Converting narration to audio cues with timing
  // 5. Ensuring byte-exact visual parity

  return {
    id: deckId,
    topic: "system-architecture",
    slides: [],
    glossary: [],
  };
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
