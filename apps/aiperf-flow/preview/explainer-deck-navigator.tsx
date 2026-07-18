// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode } from "react";
import { COMPILED_EXPLAINER_DECKS } from "../packages/runtime/src/explainer/compiled-decks";
import { ExplainerSlideViewer } from "../packages/runtime/src/explainer/ui/ExplainerSlideViewer";

type ExplainerDeckNavigatorProps = Readonly<{
  deckId: string;
  deck?: any;
  slideIndex: number;
  onSlideChange(newIndex: number): void;
  onBackClick(): void;
}>;

/**
 * Converts deck ID to human-readable title.
 * Examples: 'rust-architecture' → "Rust Architecture"
 *          'slurm-velo' → "SLURM Velo"
 */
function idToTitle(id: string): string {
  return id
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Navigation wrapper for viewing and navigating through explainer deck slides.
 * Renders ExplainerSlideViewer with prev/next controls and slide counter.
 */
export function ExplainerDeckNavigator({
  deckId,
  deck,
  slideIndex,
  onSlideChange,
  onBackClick,
}: ExplainerDeckNavigatorProps): ReactNode {
  // Use passed deck object if available, otherwise look it up by ID
  // This ensures correct deck is rendered even if lookup is done elsewhere
  const resolvedDeck = deck ?? COMPILED_EXPLAINER_DECKS.find((d) => d.id === deckId);

  if (!resolvedDeck) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "var(--preview-chalk, #f0f6fc)",
        }}
      >
        <p>Deck not found: {deckId}</p>
        <button
          onClick={onBackClick}
          type="button"
          style={{
            marginTop: "1rem",
            padding: "0.5rem 1rem",
            backgroundColor: "var(--preview-signal, #3fb950)",
            color: "var(--preview-board, #0d1117)",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Back
        </button>
      </div>
    );
  }

  const slideCount = resolvedDeck.slides.length;
  const canGoPrev = slideIndex > 0;
  const canGoNext = slideIndex < slideCount - 1;

  const handlePrev = (): void => {
    if (canGoPrev) {
      onSlideChange(slideIndex - 1);
    }
  };

  const handleNext = (): void => {
    if (canGoNext) {
      onSlideChange(slideIndex + 1);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        backgroundColor: "var(--preview-board, #0d1117)",
        color: "var(--preview-chalk, #f0f6fc)",
      }}
    >
      {/* Header with deck title and close button */}
      <div
        style={{
          padding: "1rem 1.5rem",
          borderBottom: "1px solid var(--preview-guide, #30363d)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "1.3rem",
              fontWeight: 600,
              color: "var(--preview-chalk, #f0f6fc)",
            }}
          >
            {idToTitle(deckId)}
          </h2>
          <p
            style={{
              margin: "0.25rem 0 0 0",
              fontSize: "0.85rem",
              color: "var(--preview-muted, #8b949e)",
            }}
          >
            Slide {slideIndex + 1} of {slideCount}
          </p>
        </div>
        <button
          onClick={onBackClick}
          type="button"
          aria-label="Back to decks"
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: "var(--preview-control, #2a2a2a)",
            border: "1px solid var(--preview-guide, #30363d)",
            borderRadius: "4px",
            color: "var(--preview-chalk, #f0f6fc)",
            cursor: "pointer",
            fontSize: "0.9rem",
            fontWeight: 500,
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            const target = e.currentTarget as HTMLButtonElement;
            target.style.backgroundColor = "var(--preview-raised, #21262d)";
            target.style.borderColor = "var(--preview-signal, #3fb950)";
          }}
          onMouseLeave={(e) => {
            const target = e.currentTarget as HTMLButtonElement;
            target.style.backgroundColor = "var(--preview-control, #2a2a2a)";
            target.style.borderColor = "var(--preview-guide, #30363d)";
          }}
        >
          ← Back to scenes
        </button>
      </div>

      {/* Slide viewer */}
      <div
        style={{
          flex: 1,
          overflow: "auto",
          padding: "1.5rem",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <ExplainerSlideViewer
          deck={resolvedDeck}
          slideIndex={slideIndex}
          onSlideChange={onSlideChange}
        />
      </div>

      {/* Navigation footer */}
      <div
        style={{
          padding: "1rem 1.5rem",
          borderTop: "1px solid var(--preview-guide, #30363d)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "1rem",
        }}
      >
        <button
          onClick={handlePrev}
          disabled={!canGoPrev}
          type="button"
          aria-label="Previous slide"
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: canGoPrev
              ? "var(--preview-control, #2a2a2a)"
              : "var(--preview-guide, #30363d)",
            border: `1px solid ${
              canGoPrev
                ? "var(--preview-guide, #30363d)"
                : "var(--preview-guide, #30363d)"
            }`,
            borderRadius: "4px",
            color: canGoPrev
              ? "var(--preview-chalk, #f0f6fc)"
              : "var(--preview-muted, #8b949e)",
            cursor: canGoPrev ? "pointer" : "not-allowed",
            fontSize: "0.9rem",
            fontWeight: 500,
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            if (canGoPrev) {
              const target = e.currentTarget as HTMLButtonElement;
              target.style.backgroundColor = "var(--preview-raised, #21262d)";
              target.style.borderColor = "var(--preview-signal, #3fb950)";
            }
          }}
          onMouseLeave={(e) => {
            if (canGoPrev) {
              const target = e.currentTarget as HTMLButtonElement;
              target.style.backgroundColor = "var(--preview-control, #2a2a2a)";
              target.style.borderColor = "var(--preview-guide, #30363d)";
            }
          }}
        >
          ← Previous
        </button>

        <div
          style={{
            fontSize: "0.9rem",
            color: "var(--preview-muted, #8b949e)",
            fontWeight: 500,
          }}
        >
          Slide {slideIndex + 1} / {slideCount}
        </div>

        <button
          onClick={handleNext}
          disabled={!canGoNext}
          type="button"
          aria-label="Next slide"
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: canGoNext
              ? "var(--preview-control, #2a2a2a)"
              : "var(--preview-guide, #30363d)",
            border: `1px solid ${
              canGoNext
                ? "var(--preview-guide, #30363d)"
                : "var(--preview-guide, #30363d)"
            }`,
            borderRadius: "4px",
            color: canGoNext
              ? "var(--preview-chalk, #f0f6fc)"
              : "var(--preview-muted, #8b949e)",
            cursor: canGoNext ? "pointer" : "not-allowed",
            fontSize: "0.9rem",
            fontWeight: 500,
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            if (canGoNext) {
              const target = e.currentTarget as HTMLButtonElement;
              target.style.backgroundColor = "var(--preview-raised, #21262d)";
              target.style.borderColor = "var(--preview-signal, #3fb950)";
            }
          }}
          onMouseLeave={(e) => {
            if (canGoNext) {
              const target = e.currentTarget as HTMLButtonElement;
              target.style.backgroundColor = "var(--preview-control, #2a2a2a)";
              target.style.borderColor = "var(--preview-guide, #30363d)";
            }
          }}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
