/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckDefinition } from "./types.js";

const decks = new Map<string, DeckDefinition>();

export function registerDeck(deck: DeckDefinition): void {
  if (decks.has(deck.id)) {
    throw new Error(`Deck "${deck.id}" is already registered.`);
  }
  decks.set(deck.id, deck);
}

export function getDeck(id: string): DeckDefinition | undefined {
  return decks.get(id);
}

export function listDecks(): readonly DeckDefinition[] {
  return [...decks.values()];
}

/**
 * Test-only utility that empties the deck registry. Not used by any
 * production code path; exists so tests can isolate registrations between
 * cases without relying on `registerDeck`'s duplicate-id guard being lax.
 */
export function clearDecks(): void {
  decks.clear();
}
