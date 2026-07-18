/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Link } from "react-router-dom";
import {
  DECK_REGISTRY,
  EXPECTED_DECK_ROUTES,
  deckById,
} from "../core/deck-registry";

/**
 * Home index: one card per canonical registry deck (bookmark-stable id/route),
 * titled from DeckPackage hub metadata.
 */
export function Hub() {
  const decks = EXPECTED_DECK_ROUTES.map(([id, route]) => {
    const deck = deckById(id) ?? DECK_REGISTRY.find((entry) => entry.route === route);
    return { id, route, deck };
  });

  const missing = decks.filter((entry) => entry.deck === undefined);

  return (
    <main className="ex-page ex-hub">
      <div className="ex-eyebrow" style={{ marginBottom: 10 }}>
        AIPerf · Explainers
      </div>
      <h1 className="ex-title" style={{ marginBottom: 12 }}>
        Interactive walkthroughs
      </h1>
      <p className="ex-lede" style={{ margin: "0 0 28px" }}>
        Short, narrated slideshows that explain how AIPerf pieces fit together. Pick a deck to start.
      </p>
      {missing.length > 0 ? (
        <p role="alert" className="ex-alert">
          Hub cannot list {missing.length} registered deck
          {missing.length === 1 ? "" : "s"}:{" "}
          {missing.map((entry) => entry.id).join(", ")}.
        </p>
      ) : null}
      <div className="ex-hub-grid">
        {decks.map(({ id, deck }) => {
          if (deck === undefined) {
            return null;
          }
          return (
            <Link
              key={id}
              to={deck.route}
              aria-label={`${deck.hub.highlight} ${deck.hub.title}`}
              className="ex-card"
            >
              <div className="ex-card__title">
                <span className="ex-card__highlight">{deck.hub.highlight}</span>{" "}
                {deck.hub.title}
              </div>
              <div className="ex-card__description">{deck.hub.description}</div>
            </Link>
          );
        })}
      </div>
    </main>
  );
}
