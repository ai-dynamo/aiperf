/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearDecks, registerDeck } from "../../deck/registry.js";
import { DeckRoute } from "../../deck/DeckRoute.js";
import { ASYNC_DATAFLOW_ENGINE_DECK } from "./deck.js";

function renderDeck() {
  return render(
    <MemoryRouter initialEntries={["/async-dataflow-engine"]}>
      <Routes>
        <Route path="/:deckId" element={<DeckRoute />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("async dataflow engine deck", () => {
  beforeEach(() => {
    clearDecks();
    registerDeck(ASYNC_DATAFLOW_ENGINE_DECK);
    window.localStorage.clear();
  });

  afterEach(() => {
    clearDecks();
  });

  it("gives every slide narration and a source-bearing caption", () => {
    for (const slide of ASYNC_DATAFLOW_ENGINE_DECK.slides) {
      expect(slide.narration.trim().length, `${slide.id} narration`).toBeGreaterThan(80);
      // Captions cite the file and symbol behind the slide's claim, so a viewer
      // can verify it against the engine rather than trusting the deck.
      expect(slide.caption, `${slide.id} caption`).toMatch(/\.rs:\d+/);
    }
  });

  it("opens behind the start gate", async () => {
    renderDeck();

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("1 / 13")).toBeInTheDocument();
  });

  it("takes arrow keys straight after the gate, with no intervening click", () => {
    // Regression guard: dismissing the gate unmounts the focused button, so
    // without an explicit refocus the keydown handler never receives the event.
    renderDeck();

    fireEvent.click(screen.getByRole("button", { name: "Play without audio" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    // Whatever holds focus must be what the deck listens on — that is the bug.
    const focused = document.activeElement;
    expect(focused).not.toBe(document.body);

    fireEvent.keyDown(focused as Element, { key: "ArrowRight" });
    expect(screen.getByText("2 / 13")).toBeInTheDocument();
  });
});
