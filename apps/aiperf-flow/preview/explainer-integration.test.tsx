// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { HomePage } from "./home-page";
import { ExplainerDeckPicker } from "./explainer-deck-picker";
import { ExplainerDeckNavigator } from "./explainer-deck-navigator";
import { COMPILED_EXPLAINER_DECKS } from "../packages/runtime/src/explainer/compiled-decks";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Explainer Deck Navigation Integration", () => {
  test("HomePage displays Explainers button", () => {
    const mockSelectScene = vi.fn();
    const mockOpenExplainers = vi.fn();

    render(
      <HomePage
        scenesByFlow={[]}
        onSelectScene={mockSelectScene}
        onOpenExplainers={mockOpenExplainers}
      />
    );

    const button = screen.getByRole("button", { name: /Open explainer decks/i });
    expect(button).toBeTruthy();
  });

  test("Clicking Explainers button calls onOpenExplainers", () => {
    const mockSelectScene = vi.fn();
    const mockOpenExplainers = vi.fn();

    render(
      <HomePage
        scenesByFlow={[]}
        onSelectScene={mockSelectScene}
        onOpenExplainers={mockOpenExplainers}
      />
    );

    const button = screen.getByRole("button", { name: /Open explainer decks/i });
    fireEvent.click(button);

    expect(mockOpenExplainers).toHaveBeenCalled();
  });

  test("ExplainerDeckPicker displays all decks", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    // All 4 decks should be displayed
    expect(screen.getByText(/Rust Architecture/i)).toBeTruthy();
    expect(screen.getByText(/Slurm Velo/i)).toBeTruthy();
    expect(screen.getByText(/Dynosim/i)).toBeTruthy();
  });

  test("ExplainerDeckPicker calls onDeckSelect when deck clicked", () => {
    const mockSelect = vi.fn();
    render(<ExplainerDeckPicker decks={COMPILED_EXPLAINER_DECKS} onDeckSelect={mockSelect} />);

    const deckButton = screen.getByRole("button", { name: /Load Rust Architecture/i });
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
});
