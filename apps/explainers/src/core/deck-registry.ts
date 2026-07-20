/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckDefinition } from "./types";
import {
  loadDeckFlowById,
  loadDeckPackages,
} from "./load-deck-flows";

/** Canonical registry ids and routes — must stay bookmark-stable. */
export const EXPECTED_DECK_ROUTES = [
  ["rust-architecture", "/rust-architecture"],
  ["rust-architecture-atlas", "/rust-architecture-atlas"],
  ["segment-pools", "/segment-pools"],
  ["slurm-velo", "/slurm-velo"],
  ["velo-deep-dive", "/velo-deep-dive"],
  ["cellular-internals", "/cellular-internals"],
  ["cellular-algorithms", "/cellular-algorithms"],
  ["dynosim", "/dynosim"],
  ["tstar-warmup", "/tstar-warmup"],
  ["synthetic-dataset-generator", "/synthetic-dataset-generator"],
  ["aiperf-vs-locust", "/aiperf-vs-locust"],
] as const;

export type RegisteredDeckId = (typeof EXPECTED_DECK_ROUTES)[number][0];

/**
 * Load a deck exclusively from its live Flow source.
 * Throws when the source is missing or id/route diverge from the bookmark map.
 */
export function deckFromFlow(
  id: RegisteredDeckId,
  expectedRoute: string,
): DeckDefinition {
  const fromFlow = loadDeckFlowById(id);
  if (fromFlow === undefined) {
    throw new Error(
      `Missing live Flow source for "${id}" under decks-flow ` +
        `(expected ${id}.flow compiled via compileExplainerSource)`,
    );
  }
  if (fromFlow.id !== id) {
    throw new Error(
      `Live Flow source for "${id}" has mismatched id "${fromFlow.id}"`,
    );
  }
  if (fromFlow.route !== expectedRoute) {
    throw new Error(
      `Live Flow source for "${id}" has route "${fromFlow.route}", expected "${expectedRoute}"`,
    );
  }
  return fromFlow;
}

export const DECK_REGISTRY: readonly DeckDefinition[] = EXPECTED_DECK_ROUTES.map(
  ([id, route]) => deckFromFlow(id, route),
);

export function deckByRoute(route: string): DeckDefinition | undefined {
  return DECK_REGISTRY.find((deck) => deck.route === route);
}

export function deckById(id: string): DeckDefinition | undefined {
  return DECK_REGISTRY.find((deck) => deck.id === id);
}

/** Whether any raw Flow source is currently discoverable and compilable. */
export function registryUsesLiveFlowSources(): boolean {
  return loadDeckPackages().length > 0;
}

export function validateDeckRegistry(decks: readonly DeckDefinition[] = DECK_REGISTRY): string[] {
  const errors: string[] = [];
  const routes = new Set<string>();
  const ids = new Set<string>();

  if (decks.length !== EXPECTED_DECK_ROUTES.length) {
    errors.push(
      `expected ${EXPECTED_DECK_ROUTES.length} decks, found ${decks.length}`,
    );
  }

  for (const [expectedId, expectedRoute] of EXPECTED_DECK_ROUTES) {
    const deck = decks.find((entry) => entry.id === expectedId);
    if (deck === undefined) {
      errors.push(`missing deck id: ${expectedId}`);
      continue;
    }
    if (deck.route !== expectedRoute) {
      errors.push(
        `${expectedId}: route "${deck.route}" does not match expected "${expectedRoute}"`,
      );
    }
  }

  for (const deck of decks) {
    if (routes.has(deck.route)) errors.push(`duplicate route: ${deck.route}`);
    routes.add(deck.route);
    if (ids.has(deck.id)) errors.push(`duplicate id: ${deck.id}`);
    ids.add(deck.id);
    if (deck.slides.length === 0) errors.push(`${deck.id}: no slides`);
    deck.slides.forEach((slide, index) => {
      if (!slide.narration.trim()) errors.push(`${deck.id}: slide ${index + 1} missing narration`);
      if (!slide.title.trim()) errors.push(`${deck.id}: slide ${index + 1} missing title`);
    });
  }

  return errors;
}
