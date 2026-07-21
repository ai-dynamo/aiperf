/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { ExplainerShell } from "../core/ExplainerShell";
import { deckManifestByRoute } from "../core/deck-registry";
import { loadDeckFlowById } from "../core/load-deck-flows";
import type { DeckDefinition } from "../core/types";

/**
 * Resolve the active deck from the pathname and lazily compile only that
 * deck's `.flow` source (or fetch its precompiled JSON in prod) — visiting
 * one route never compiles the other decks.
 */
export function DeckRoute() {
  const { pathname } = useLocation();
  const manifestEntry = deckManifestByRoute(pathname);
  const [deck, setDeck] = useState<DeckDefinition | undefined>(undefined);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (manifestEntry === undefined) {
      return;
    }
    let cancelled = false;
    setDeck(undefined);
    setFailed(false);
    loadDeckFlowById(manifestEntry.id).then(
      (loaded) => {
        if (!cancelled) setDeck(loaded);
      },
      (error: unknown) => {
        console.error(error);
        if (!cancelled) setFailed(true);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [manifestEntry?.id]);

  if (manifestEntry === undefined || failed) {
    return <Navigate to="/" replace />;
  }

  if (deck === undefined) {
    return (
      <main className="ex-page ex-loading" aria-busy="true">
        Loading {manifestEntry.hub.title}…
      </main>
    );
  }

  // key remounts shell so slide/started/notes state cannot leak across decks
  // when React Router reuses the DeckRoute element between sibling routes.
  return <ExplainerShell key={deck.id} deck={deck} />;
}
