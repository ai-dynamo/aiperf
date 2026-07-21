/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import manifestJson from "./deck-manifest.generated.json";
import type { DeckHubMeta } from "./types";

/** Canonical registry ids and routes — must stay bookmark-stable. */
export const EXPECTED_DECK_ROUTES = [
  ["rust-architecture", "/rust-architecture"],
  ["rust-architecture-atlas", "/rust-architecture-atlas"],
  ["rust-architecture-deck-port", "/rust-architecture-deck-port"],
  ["segment-pools", "/segment-pools"],
  ["slurm-velo", "/slurm-velo"],
  ["velo-deep-dive", "/velo-deep-dive"],
  ["cellular-internals", "/cellular-internals"],
  ["cellular-algorithms", "/cellular-algorithms"],
  ["dynosim", "/dynosim"],
  ["steppable-replay-engine", "/steppable-replay-engine"],
  ["tstar-warmup", "/tstar-warmup"],
  ["synthetic-dataset-generator", "/synthetic-dataset-generator"],
  ["aiperf-vs-locust", "/aiperf-vs-locust"],
  ["flow-sdk-examples", "/flow-sdk-examples"],
  ["sdk-generic-catalog", "/sdk-generic-catalog"],
  ["sdk-diagram-catalog", "/sdk-diagram-catalog"],
] as const;

export type RegisteredDeckId = (typeof EXPECTED_DECK_ROUTES)[number][0];

/**
 * Lightweight per-deck metadata compiled once, server-side, by
 * `scripts/build-deck-artifacts.mjs` — never a full slide/scene compile.
 * The Hub screen and the app's route table run entirely off this; a deck's
 * full `DeckPackage` (slides, scenes, glossary) only compiles when its own
 * route is actually visited, via `load-deck-flows.ts`.
 */
export type DeckManifestEntry = {
  id: string;
  route: string;
  topic: string;
  eyebrowLabel: string;
  hub: DeckHubMeta;
  slideCount: number;
};

export const DECK_MANIFEST: readonly DeckManifestEntry[] = manifestJson;

export function deckManifestByRoute(route: string): DeckManifestEntry | undefined {
  return DECK_MANIFEST.find((entry) => entry.route === route);
}

export function deckManifestById(id: string): DeckManifestEntry | undefined {
  return DECK_MANIFEST.find((entry) => entry.id === id);
}

export function validateDeckManifest(
  manifest: readonly DeckManifestEntry[] = DECK_MANIFEST,
): string[] {
  const errors: string[] = [];
  const routes = new Set<string>();
  const ids = new Set<string>();

  if (manifest.length !== EXPECTED_DECK_ROUTES.length) {
    errors.push(
      `expected ${EXPECTED_DECK_ROUTES.length} decks, found ${manifest.length}`,
    );
  }

  for (const [expectedId, expectedRoute] of EXPECTED_DECK_ROUTES) {
    const entry = manifest.find((candidate) => candidate.id === expectedId);
    if (entry === undefined) {
      errors.push(`missing deck id: ${expectedId}`);
      continue;
    }
    if (entry.route !== expectedRoute) {
      errors.push(
        `${expectedId}: route "${entry.route}" does not match expected "${expectedRoute}"`,
      );
    }
  }

  for (const entry of manifest) {
    if (routes.has(entry.route)) errors.push(`duplicate route: ${entry.route}`);
    routes.add(entry.route);
    if (ids.has(entry.id)) errors.push(`duplicate id: ${entry.id}`);
    ids.add(entry.id);
    if (entry.slideCount === 0) errors.push(`${entry.id}: no slides`);
  }

  return errors;
}
