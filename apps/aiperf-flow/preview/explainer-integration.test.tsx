// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { ExplainerDeckPicker } from "./explainer-deck-picker";
import { ExplainerDeckNavigator } from "./explainer-deck-navigator";
import { COMPILED_EXPLAINER_DECKS } from "./deck-packages";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Explainer Deck Navigation Integration", () => {
  test("ExplainerDeckPicker displays all decks", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // All 8 decks should be displayed
    expect(screen.getByRole("heading", { level: 3, name: /^Rust Architecture$/i })).toBeTruthy();
    expect(screen.getByText(/Slurm Velo/i)).toBeTruthy();
    expect(screen.getByRole("heading", { level: 3, name: /^Dynosim$/i })).toBeTruthy();
  });

  test("ExplainerDeckPicker calls onDeckSelect when deck clicked", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const deckButton = screen.getByRole("button", { name: /^Load Rust Architecture explainer deck$/i });
    fireEvent.click(deckButton);

    expect(mockSelect).toHaveBeenCalledWith("rust-architecture");
  });

  test("ExplainerDeckPicker calls onDeckSelect with correct deck ID", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // Get all deck selection buttons
    const buttons = screen.getAllByRole("button");
    const architectureButton = buttons.find((btn) =>
      btn.textContent?.includes("Rust Architecture")
    );

    if (architectureButton) {
      fireEvent.click(architectureButton);
      expect(mockSelect).toHaveBeenCalledWith("rust-architecture");
    }
  });

  test("ExplainerDeckNavigator advances slides when Next button is clicked", () => {
    const mockSlideChange = vi.fn();
    const mockBackClick = vi.fn();
    const deck = COMPILED_EXPLAINER_DECKS[0];

    if (!deck) {
      throw new Error("No test deck available");
    }

    render(
      <ExplainerDeckNavigator
        deckId={deck.id}
        slideIndex={0}
        onSlideChange={mockSlideChange}
        onBackClick={mockBackClick}
      />
    );

    // Verify initial slide is displayed
    expect(screen.getByText(/Slide 1 of/i)).toBeTruthy();

    // Click Next button
    const nextButton = screen.getByRole("button", { name: /Next/i });
    fireEvent.click(nextButton);

    // Verify callback was called with correct index
    expect(mockSlideChange).toHaveBeenCalledWith(1);
  });

  test("ExplainerDeckNavigator goes back slides when Previous button is clicked", () => {
    const mockSlideChange = vi.fn();
    const mockBackClick = vi.fn();
    const deck = COMPILED_EXPLAINER_DECKS[0];

    if (!deck) {
      throw new Error("No test deck available");
    }

    render(
      <ExplainerDeckNavigator
        deckId={deck.id}
        slideIndex={2}
        onSlideChange={mockSlideChange}
        onBackClick={mockBackClick}
      />
    );

    // Verify we're at slide 3
    expect(screen.getByText(/Slide 3 of/i)).toBeTruthy();

    // Click Previous button
    const prevButton = screen.getByRole("button", { name: /Previous/i });
    fireEvent.click(prevButton);

    // Verify callback was called with correct index
    expect(mockSlideChange).toHaveBeenCalledWith(1);
  });

  test("ExplainerDeckNavigator disables Previous on first slide", () => {
    const mockSlideChange = vi.fn();
    const mockBackClick = vi.fn();
    const deck = COMPILED_EXPLAINER_DECKS[0];

    if (!deck) {
      throw new Error("No test deck available");
    }

    render(
      <ExplainerDeckNavigator
        deckId={deck.id}
        slideIndex={0}
        onSlideChange={mockSlideChange}
        onBackClick={mockBackClick}
      />
    );

    // Previous button should be disabled on first slide
    const prevButton = screen.getByRole("button", { name: /Previous/i });
    expect((prevButton as HTMLButtonElement).disabled).toBe(true);
  });

  test("ExplainerDeckNavigator disables Next on last slide", () => {
    const mockSlideChange = vi.fn();
    const mockBackClick = vi.fn();
    const deck = COMPILED_EXPLAINER_DECKS[0];

    if (!deck) {
      throw new Error("No test deck available");
    }

    const lastSlideIndex = deck.slides.length - 1;

    render(
      <ExplainerDeckNavigator
        deckId={deck.id}
        slideIndex={lastSlideIndex}
        onSlideChange={mockSlideChange}
        onBackClick={mockBackClick}
      />
    );

    // Next button should be disabled on last slide
    const nextButton = screen.getByRole("button", { name: /Next/i });
    expect((nextButton as HTMLButtonElement).disabled).toBe(true);
  });
});
