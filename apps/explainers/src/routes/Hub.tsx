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
import { useHostTheme } from "../core/ui";

/**
 * Home index: one card per canonical registry deck (bookmark-stable id/route),
 * titled from DeckPackage hub metadata.
 */
export function Hub() {
  const t = useHostTheme();

  const decks = EXPECTED_DECK_ROUTES.map(([id, route]) => {
    const deck = deckById(id) ?? DECK_REGISTRY.find((entry) => entry.route === route);
    return { id, route, deck };
  });

  const missing = decks.filter((entry) => entry.deck === undefined);

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "48px 24px" }}>
      <div
        style={{
          color: t.text.secondary,
          fontSize: 13,
          fontWeight: 650,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 10,
        }}
      >
        AIPerf · Explainers
      </div>
      <h1 style={{ margin: "0 0 12px", fontSize: 32, lineHeight: 1.15, color: t.text.primary }}>
        Interactive walkthroughs
      </h1>
      <p style={{ color: t.text.secondary, fontSize: 17, lineHeight: 1.55, margin: "0 0 28px" }}>
        Short, narrated slideshows that explain how AIPerf pieces fit together. Pick a deck to start.
      </p>
      {missing.length > 0 ? (
        <p
          role="alert"
          style={{
            color: t.text.primary,
            background: t.bg.elevated,
            border: `1px solid ${t.stroke.secondary}`,
            borderRadius: 10,
            padding: "14px 16px",
            margin: "0 0 16px",
            fontSize: 15,
            lineHeight: 1.5,
          }}
        >
          Hub cannot list {missing.length} registered deck
          {missing.length === 1 ? "" : "s"}:{" "}
          {missing.map((entry) => entry.id).join(", ")}.
        </p>
      ) : null}
      <div style={{ display: "grid", gap: 12 }}>
        {decks.map(({ id, deck }) => {
          if (deck === undefined) {
            return null;
          }
          return (
            <Link
              key={id}
              to={deck.route}
              aria-label={`${deck.hub.highlight} ${deck.hub.title}`}
              style={{
                display: "block",
                textDecoration: "none",
                color: "inherit",
                background: t.bg.elevated,
                border: `1px solid ${t.stroke.secondary}`,
                borderRadius: 10,
                padding: "18px 18px 16px",
              }}
            >
              <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>
                <span style={{ color: t.category.green }}>{deck.hub.highlight}</span>{" "}
                {deck.hub.title}
              </div>
              <div style={{ color: t.text.secondary, fontSize: 15, lineHeight: 1.5 }}>
                {deck.hub.description}
              </div>
            </Link>
          );
        })}
      </div>
    </main>
  );
}
