/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Navigate, useLocation } from "react-router-dom";
import { ExplainerShell } from "../core/ExplainerShell";
import { deckByRoute } from "../core/deck-registry";

/** Resolve the active deck from the pathname (matches DeckPackage.route). */
export function DeckRoute() {
  const { pathname } = useLocation();
  const deck = deckByRoute(pathname);

  if (!deck) {
    return <Navigate to="/" replace />;
  }

  // key remounts shell so slide/started/notes state cannot leak across decks
  // when React Router reuses the DeckRoute element between sibling routes.
  return <ExplainerShell key={deck.id} deck={deck} />;
}
