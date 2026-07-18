// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode } from "react";
import type { PreviewDeck } from "./deck-packages";

type ExplainerDeckPickerProps = Readonly<{
  decks: readonly PreviewDeck[];
  onDeckSelect: (deckId: string) => void;
}>;

/**
 * Converts deck ID to human-readable title.
 * Examples: 'rust-architecture' → "Rust Architecture"
 *          'slurm-velo' → "SLURM Velo"
 *          'dynosim' → "DynoSim"
 *          'aiperf-flow-system' → "AIPerf Flow System"
 */
function idToTitle(id: string): string {
  return id
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

type DeckCardProps = Readonly<{
  deck: PreviewDeck;
  onSelect: (deckId: string) => void;
}>;

function DeckCard({ deck, onSelect }: DeckCardProps): ReactNode {
  const slideCount = deck.slides.length;
  const description = deck.slides[0]?.lede ?? "Explore this explainer deck";

  return (
    <button
      className="deck-card"
      onClick={() => onSelect(deck.id)}
      type="button"
      aria-label={`Load ${idToTitle(deck.id)} explainer deck`}
    >
      <div className="deck-card-header">
        <h3 className="deck-card-title">{idToTitle(deck.id)}</h3>
        <span className="deck-card-slide-count">
          {slideCount} {slideCount === 1 ? "slide" : "slides"}
        </span>
      </div>
      <p className="deck-card-description">{description}</p>
      <span className="deck-card-badge">View deck</span>
    </button>
  );
}

/**
 * Explainer deck picker component for displaying available decks.
 * Renders a grid of clickable deck cards matching home page card style.
 *
 * @param decks - Array of ExplainerDefinition decks to display
 * @param onDeckSelect - Callback fired when user selects a deck
 */
export function ExplainerDeckPicker({
  decks,
  onDeckSelect,
}: ExplainerDeckPickerProps): ReactNode {
  return (
    <div className="explainer-deck-picker">
      <header className="explainer-header">
        <h1>Explainer Decks</h1>
        <p className="explainer-subtitle">
          Choose a walkthrough to understand the system
        </p>
      </header>

      <div className="deck-cards-grid">
        {decks.map((deck) => (
          <DeckCard
            key={deck.id}
            deck={deck}
            onSelect={onDeckSelect}
          />
        ))}
      </div>

      <style>{`
        .explainer-deck-picker {
          position: fixed;
          inset: 0;
          z-index: 9999;
          overflow-y: auto;
          padding: 2rem;
          background: var(--preview-board, #0d1117);
          color: var(--preview-chalk, #f0f6fc);
          min-height: 100vh;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .explainer-header {
          max-width: 1200px;
          margin: 0 auto 3rem;
          text-align: center;
          padding-bottom: 2rem;
          border-bottom: 1px solid var(--preview-guide, #596266);
        }

        .explainer-header h1 {
          font-size: 2.5rem;
          font-weight: 700;
          margin: 0 0 0.5rem;
          letter-spacing: -0.02em;
        }

        .explainer-subtitle {
          font-size: 1.1rem;
          color: var(--preview-muted, #9da6aa);
          margin: 0;
        }

        .deck-cards-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 1.5rem;
          max-width: 1200px;
          margin: 0 auto;
        }

        .deck-card {
          all: unset;
          cursor: pointer;
          padding: 1.5rem;
          background: var(--preview-panel, #24282b);
          border: 1px solid var(--preview-guide, #596266);
          border-radius: 8px;
          transition: all 0.2s ease;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          text-align: left;
        }

        .deck-card:hover {
          border-color: var(--preview-signal, #65d9de);
          background: var(--preview-raised, #2c3135);
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .deck-card:focus-visible {
          outline: 2px solid var(--preview-signal, #65d9de);
          outline-offset: 2px;
        }

        .deck-card-header {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .deck-card-title {
          font-size: 1.2rem;
          font-weight: 600;
          margin: 0;
          color: var(--preview-chalk, #f0f6fc);
        }

        .deck-card-slide-count {
          font-size: 0.8rem;
          font-weight: 500;
          color: var(--preview-signal, #65d9de);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .deck-card-description {
          font-size: 0.95rem;
          color: var(--preview-muted, #9da6aa);
          margin: 0;
          line-height: 1.5;
          flex: 1;
        }

        .deck-card-badge {
          align-self: flex-start;
          padding: 0.4rem 0.8rem;
          background: var(--preview-signal, #65d9de);
          color: var(--preview-board, #181b1d);
          font-size: 0.85rem;
          font-weight: 600;
          border-radius: 4px;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }

        @media (max-width: 860px) {
          .explainer-header h1 {
            font-size: 2rem;
          }

          .explainer-deck-picker {
            padding: 1rem;
          }

          .explainer-header {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
          }

          .deck-cards-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
          }
        }
      `}</style>
    </div>
  );
}
