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

function manifestEntryFor(deck: DeckDefinition) {
  return {
    id: deck.id,
    route: deck.route,
    topic: "test",
    eyebrowLabel: deck.eyebrowLabel,
    hub: deck.hub,
    slideCount: deck.slides.length,
  };
}

vi.mock("../core/deck-registry", () => ({
  deckManifestByRoute: (pathname: string) => {
    if (pathname === "/deck-a") return manifestEntryFor(deckA);
    if (pathname === "/deck-b") return manifestEntryFor(deckB);
    return undefined;
  },
}));

vi.mock("../core/load-deck-flows", () => ({
  loadDeckFlowById: (id: string) => {
    if (id === "deck-a") return Promise.resolve(deckA);
    if (id === "deck-b") return Promise.resolve(deckB);
    return Promise.reject(new Error(`unknown deck id "${id}"`));
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
  it("resets started state when navigating to a different deck", async () => {
    render(
      <MemoryRouter initialEntries={["/deck-a"]}>
        <Link to="/deck-b">Open deck B</Link>
        <Routes>
          <Route path="/deck-a" element={<DeckRoute />} />
          <Route path="/deck-b" element={<DeckRoute />} />
        </Routes>
      </MemoryRouter>,
    );

    // Lazy per-route compilation resolves asynchronously — wait for it.
    expect(await screen.findByText("Deck A start gate")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /play without audio/i }));
    expect(screen.queryByText("Deck A start gate")).toBeNull();
    expect(screen.getByTestId("mental-model-deck-a")).toBeTruthy();

    fireEvent.click(screen.getByRole("link", { name: /open deck b/i }));

    // Without key={deck.id}, ExplainerShell keeps started=true across route reuse.
    expect(await screen.findByText("Deck B start gate")).toBeTruthy();
    expect(screen.queryByText("Deck A start gate")).toBeNull();
  });
});
