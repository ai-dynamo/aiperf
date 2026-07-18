// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Preview deck list backed by compiler-emitted DeckPackage JSON under
 * `apps/explainers/src/decks-generated/`. Adapts `render.scene` → `sceneBlock`
 * for ExplainerSlideViewer.
 */

import type { DeckPackage, SlidePackage } from "@aiperf/flow-schema";

import cellularAlgorithms from "../../explainers/src/decks-generated/cellular-algorithms.package.json";
import cellularInternals from "../../explainers/src/decks-generated/cellular-internals.package.json";
import dynosim from "../../explainers/src/decks-generated/dynosim.package.json";
import rustArchitectureAtlas from "../../explainers/src/decks-generated/rust-architecture-atlas.package.json";
import rustArchitecture from "../../explainers/src/decks-generated/rust-architecture.package.json";
import segmentPools from "../../explainers/src/decks-generated/segment-pools.package.json";
import slurmVelo from "../../explainers/src/decks-generated/slurm-velo.package.json";
import veloDeepDive from "../../explainers/src/decks-generated/velo-deep-dive.package.json";

/** Preview slide shape expected by ExplainerSlideViewer (`sceneBlock`). */
export type PreviewSlide = Readonly<{
  id: string;
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: readonly string[];
  caption: string;
  sceneBlock?: unknown;
}>;

/** Preview deck shape used by the aiperf-flow preview host. */
export type PreviewDeck = Readonly<{
  id: string;
  topic: string;
  route?: string;
  slides: readonly PreviewSlide[];
  glossary: readonly { word: string; meaning: string }[];
}>;

function toPreviewSlide(slide: SlidePackage): PreviewSlide {
  return {
    id: slide.id,
    eyebrow: slide.eyebrow,
    title: slide.title,
    lede: slide.lede,
    narration: slide.narration,
    ...(slide.term !== undefined ? { term: slide.term } : {}),
    points: slide.points,
    caption: slide.caption,
    ...(slide.render?.kind === "scene"
      ? { sceneBlock: slide.render.scene }
      : {}),
  };
}

function toPreviewDeck(pkg: DeckPackage): PreviewDeck {
  return {
    id: pkg.id,
    topic: pkg.topic,
    route: pkg.route,
    slides: pkg.slides.map(toPreviewSlide),
    glossary: pkg.glossary,
  };
}

export const CELLULAR_ALGORITHMS_DECK = toPreviewDeck(
  cellularAlgorithms as DeckPackage,
);
export const CELLULAR_INTERNALS_DECK = toPreviewDeck(
  cellularInternals as DeckPackage,
);
export const DYNOSIM_DECK = toPreviewDeck(dynosim as DeckPackage);
export const RUST_ARCHITECTURE_ATLAS_DECK = toPreviewDeck(
  rustArchitectureAtlas as DeckPackage,
);
export const RUST_ARCHITECTURE_DECK = toPreviewDeck(
  rustArchitecture as DeckPackage,
);
export const SEGMENT_POOLS_DECK = toPreviewDeck(segmentPools as DeckPackage);
export const SLURM_VELO_DECK = toPreviewDeck(slurmVelo as DeckPackage);
export const VELO_DEEP_DIVE_DECK = toPreviewDeck(veloDeepDive as DeckPackage);

/** All flow-backed explainer packages, sorted by id. */
export const COMPILED_EXPLAINER_DECKS: readonly PreviewDeck[] = [
  CELLULAR_ALGORITHMS_DECK,
  CELLULAR_INTERNALS_DECK,
  DYNOSIM_DECK,
  RUST_ARCHITECTURE_ATLAS_DECK,
  RUST_ARCHITECTURE_DECK,
  SEGMENT_POOLS_DECK,
  SLURM_VELO_DECK,
  VELO_DEEP_DIVE_DECK,
].sort((a, b) => a.id.localeCompare(b.id));
