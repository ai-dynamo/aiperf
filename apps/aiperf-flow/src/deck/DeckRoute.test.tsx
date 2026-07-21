/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DeckRoute } from "./DeckRoute.js";
import { clearDecks, registerDeck } from "./registry.js";
import type { DeckDefinition } from "./types.js";

const testDeck: DeckDefinition = {
  id: "test-deck",
  title: "Test Deck",
  slides: [
    {
      id: "only-slide",
      eyebrow: "Eyebrow",
      title: "Only Slide",
      lede: "Lede text.",
      narration: "Narration text.",
      caption: "Caption text.",
      nodes: [],
      edges: [],
    },
  ],
};

describe("DeckRoute", () => {
  beforeEach(() => {
    registerDeck(testDeck);
  });

  afterEach(() => {
    clearDecks();
  });

  it("renders the deck's first slide by id", () => {
    render(
      <MemoryRouter initialEntries={["/test-deck"]}>
        <Routes>
          <Route path="/:deckId" element={<DeckRoute />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText("Only Slide")).toBeInTheDocument();
  });

  it("renders a not-found message for an unregistered deck id", () => {
    render(
      <MemoryRouter initialEntries={["/nonexistent"]}>
        <Routes>
          <Route path="/:deckId" element={<DeckRoute />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText(/no deck registered/i)).toBeInTheDocument();
  });
});
