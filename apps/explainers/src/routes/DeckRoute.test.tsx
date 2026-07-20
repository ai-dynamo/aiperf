/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import type { DeckDefinition } from "../core/types";
import { DeckRoute } from "./DeckRoute";

function makeDeck(id: string, route: string, startGateTitle: string): DeckDefinition {
  return {
    id,
    route,
    storagePrefix: `ex-${id}`,
    classPrefix: id,
    eyebrowLabel: id,
    startGateTitle,
    hub: {
      title: id,
      highlight: "test",
      description: "Deck remount coverage.",
    },
    slides: [
      {
        eyebrow: "SLIDE",
        title: `${id} slide`,
        lede: "A short lede.",
        narration: "Alpha bravo",
        points: ["Point"],
        caption: "Caption",
      },
    ],
    glossary: [],
    MentalModel: () => <div data-testid={`mental-model-${id}`}>diagram</div>,
    css: "",
  };
}

const deckA = makeDeck("deck-a", "/deck-a", "Deck A start gate");
const deckB = makeDeck("deck-b", "/deck-b", "Deck B start gate");

vi.mock("../core/deck-registry", () => ({
  deckByRoute: (pathname: string) => {
    if (pathname === "/deck-a") return deckA;
    if (pathname === "/deck-b") return deckB;
    return undefined;
  },
}));

vi.mock("../core/narration", async () => {
  const actual = await vi.importActual<typeof import("../core/narration")>("../core/narration");
  return {
    ...actual,
    narrationSupported: () => true,
    unlockSpeech: () => true,
    stopNarration: () => undefined,
    speakNarration: () => () => undefined,
  };
});

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe("DeckRoute remounts ExplainerShell per deck", () => {
  it("resets started state when navigating to a different deck", () => {
    render(
      <MemoryRouter initialEntries={["/deck-a"]}>
        <Link to="/deck-b">Open deck B</Link>
        <Routes>
          <Route path="/deck-a" element={<DeckRoute />} />
          <Route path="/deck-b" element={<DeckRoute />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText("Deck A start gate")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /play without audio/i }));
    expect(screen.queryByText("Deck A start gate")).toBeNull();
    expect(screen.getByTestId("mental-model-deck-a")).toBeTruthy();

    fireEvent.click(screen.getByRole("link", { name: /open deck b/i }));

    // Without key={deck.id}, ExplainerShell keeps started=true across route reuse.
    expect(screen.getByText("Deck B start gate")).toBeTruthy();
    expect(screen.queryByText("Deck A start gate")).toBeNull();
  });
});
