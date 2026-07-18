// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { ExplainerDeckPicker } from "./explainer-deck-picker";
import { COMPILED_EXPLAINER_DECKS } from "./deck-packages";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ExplainerDeckPicker", () => {
  test("renders with header and subtitle", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    expect(screen.getByText(/Explainer Decks/i)).toBeTruthy();
    expect(screen.getByText(/Choose a walkthrough/i)).toBeTruthy();
  });

  test("renders all 4 compiled explainer decks as cards", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // Should render 4 deck cards
    const buttons = screen.getAllByRole("button", {
      name: /Load .* explainer deck/i,
    });
    expect(buttons.length).toBe(4);
  });

  test("displays correct deck titles", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // Check for human-readable titles
    expect(screen.getByText(/Rust Architecture/i)).toBeTruthy();
    expect(screen.getByText(/Slurm Velo/i)).toBeTruthy();
    expect(screen.getByRole("heading", { level: 3, name: /Dynosim/i })).toBeTruthy();
    expect(screen.getByText(/Aiperf Flow System/i)).toBeTruthy();
  });

  test("displays slide counts for each deck", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // Each deck should have a slide count displayed
    const slideCountTexts = screen.getAllByText(/slides?/i);
    expect(slideCountTexts.length).toBeGreaterThanOrEqual(4);
  });

  test("displays first slide lede as deck description", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // The first slide of rust-architecture deck should be visible as description
    const rustArchDeck = COMPILED_EXPLAINER_DECKS.find((d) => d.id === "rust-architecture");
    if (rustArchDeck && rustArchDeck.slides[0]) {
      const description = rustArchDeck.slides[0].lede;
      expect(screen.getByText(description)).toBeTruthy();
    }
  });

  test("calls onDeckSelect with correct deckId when card is clicked", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const rustArchButton = screen.getByRole("button", {
      name: /Load Rust Architecture/i,
    });
    fireEvent.click(rustArchButton);

    expect(mockSelect).toHaveBeenCalledWith("rust-architecture");
    expect(mockSelect).toHaveBeenCalledTimes(1);
  });

  test("calls onDeckSelect with correct deckId for each deck", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const expectedDecks = ["rust-architecture", "slurm-velo", "dynosim", "aiperf-flow-system"];

    expectedDecks.forEach((deckId) => {
      // Find the button matching this deck by its aria-label
      const allButtons = screen.getAllByRole("button", {
        name: /Load .* explainer deck/i,
      });

      const deckButton = allButtons.find((btn) => {
        const deckCardButton = btn as HTMLElement;
        return deckCardButton.getAttribute("aria-label")?.includes(deckId);
      });

      if (deckButton) {
        mockSelect.mockClear();
        fireEvent.click(deckButton);
        expect(mockSelect).toHaveBeenCalledWith(deckId);
      }
    });
  });

  test("deck cards are keyboard accessible", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const buttons = screen.getAllByRole("button", {
      name: /Load .* explainer deck/i,
    });

    buttons.forEach((button) => {
      expect(button).toHaveProperty("type", "button");
    });
  });

  test("renders with responsive grid layout", () => {
    const mockSelect = vi.fn();
    const { container } = render(
      <ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />
    );

    const grid = container.querySelector(".deck-cards-grid");
    expect(grid).toBeTruthy();

    // Check that grid uses CSS Grid
    const gridStyle = window.getComputedStyle(grid!);
    expect(gridStyle.display).toBe("grid");
  });

  test("displays 'View deck' badge on each card", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const badges = screen.getAllByText(/View deck/i);
    expect(badges.length).toBe(4);
  });

  test("properly formats deck id to title", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // Verify each ID is converted properly:
    // rust-architecture -> Rust Architecture
    expect(screen.getByText(/Rust Architecture/i)).toBeTruthy();
    // slurm-velo -> Slurm Velo
    expect(screen.getByText(/Slurm Velo/i)).toBeTruthy();
    // dynosim -> Dynosim
    expect(screen.getByRole("heading", { level: 3, name: /Dynosim/i })).toBeTruthy();
    // aiperf-flow-system -> Aiperf Flow System
    expect(screen.getByText(/Aiperf Flow System/i)).toBeTruthy();
  });

  test("handles singular/plural slide count correctly", () => {
    const mockSelect = vi.fn();
    const singleSlide = [
      {
        id: "single-slide-deck",
        route: "/single",
        topic: "test",
        eyebrowLabel: "Test",
        startGateTitle: "Start",
        slides: [
          {
            eyebrow: "Slide",
            title: "Only Slide",
            lede: "A single slide deck",
            narration: "Narration",
            points: [],
            caption: "",
          },
        ],
        scenesById: new Map(),
      },
    ];

    render(<ExplainerDeckPicker decks={singleSlide} onDeckSelect={mockSelect} />);

    expect(screen.getByText(/1 slide/i)).toBeTruthy();
  });

  test("renders empty state gracefully when no decks provided", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={[]} onDeckSelect={mockSelect} />);

    expect(screen.getByText(/Explainer Decks/i)).toBeTruthy();
    expect(screen.getByText(/Choose a walkthrough/i)).toBeTruthy();

    // No deck cards should be rendered
    const buttons = screen.queryAllByRole("button", {
      name: /Load .* explainer deck/i,
    });
    expect(buttons.length).toBe(0);
  });

  test("all deck cards have proper aria labels", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const buttons = screen.getAllByRole("button", {
      name: /Load .* explainer deck/i,
    });

    buttons.forEach((button) => {
      const label = button.getAttribute("aria-label");
      expect(label).toBeTruthy();
      expect(label).toMatch(/Load .* explainer deck/i);
    });
  });

  test("deck card structure is correct for each deck", () => {
    const mockSelect = vi.fn();
    const { container } = render(
      <ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />
    );

    const cards = container.querySelectorAll(".deck-card");
    expect(cards.length).toBe(4);

    cards.forEach((card) => {
      // Each card should have a header with title and slide count
      const header = card.querySelector(".deck-card-header");
      expect(header).toBeTruthy();

      const title = card.querySelector(".deck-card-title");
      expect(title).toBeTruthy();

      const slideCount = card.querySelector(".deck-card-slide-count");
      expect(slideCount).toBeTruthy();

      // Each card should have description and badge
      const description = card.querySelector(".deck-card-description");
      expect(description).toBeTruthy();

      const badge = card.querySelector(".deck-card-badge");
      expect(badge).toBeTruthy();
    });
  });
});
