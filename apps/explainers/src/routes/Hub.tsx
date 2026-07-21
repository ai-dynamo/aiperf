/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Link } from "react-router-dom";
import {
  DECK_MANIFEST,
  EXPECTED_DECK_ROUTES,
  deckManifestById,
} from "../core/deck-registry";
import { BrandMark } from "../core/ui";

/**
 * Home index: one card per canonical registry deck (bookmark-stable id/route),
 * titled from the deck manifest — never compiles a deck's slides/scenes,
 * those only compile once that deck's own route is visited.
 */
export function Hub() {
  const decks = EXPECTED_DECK_ROUTES.map(([id, route]) => {
    const deck = deckManifestById(id) ?? DECK_MANIFEST.find((entry) => entry.route === route);
    return { id, route, deck };
  });

  const missing = decks.filter((entry) => entry.deck === undefined);

  return (
    <main className="ex-page ex-hub">
      <div className="ex-hub-mast">
        <BrandMark />
        <div className="ex-hub-mast__word">AIPERF</div>
        <div className="ex-hub-mast__tag">Explainers</div>
      </div>
      <h1 className="ex-title ex-title--hub">
        <span className="ex-title__light">Interactive</span>
        <span className="ex-title__bold">walkthroughs</span>
      </h1>
      <p className="ex-lede ex-lede--hub">
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
        {decks.map(({ id, deck }, index) => {
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
              <span className="ex-card__numeral" aria-hidden="true">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div className="ex-card__meta">
                <span>Deck {String(index + 1).padStart(2, "0")}</span>
                <span>{deck.slideCount} slides</span>
              </div>
              <div className="ex-card__title">
                <span className="ex-card__highlight">{deck.hub.highlight}</span>{" "}
                {deck.hub.title}
              </div>
              <div className="ex-card__description">{deck.hub.description}</div>
              <div className="ex-card__launch">
                Open presentation <span className="ex-card__arrow">→</span>
              </div>
            </Link>
          );
        })}
      </div>
    </main>
  );
}
