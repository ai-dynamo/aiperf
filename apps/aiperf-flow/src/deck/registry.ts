/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckDefinition } from "./types.js";

const decks = new Map<string, DeckDefinition>();

export function registerDeck(deck: DeckDefinition): void {
  // Idempotent by design: re-registering the same deck id (HMR, repeated test
  // setup) replaces the prior definition instead of throwing.
  decks.set(deck.id, deck);
}

export function getDeck(id: string): DeckDefinition | undefined {
  return decks.get(id);
}

export function listDecks(): readonly DeckDefinition[] {
  return [...decks.values()];
}
