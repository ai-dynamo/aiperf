/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { DECKS, Home, SIMULATIONS } from "./Home.js";

describe("Home", () => {
  it("lists every deck with a link to its route", () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    );
    const link = screen.getByRole("link", { name: /Segment Pools/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/segment-pools");
  });

  it("renders the page heading", () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "Explainer decks" })).toBeInTheDocument();
  });
});

describe("Home listings", () => {
  it("links every live simulation to its route", () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    );
    for (const sim of SIMULATIONS) {
      const link = screen.getByRole("link", { name: new RegExp(escapeRegExp(sim.title), "i") });
      expect(link).toHaveAttribute("href", sim.path);
    }
  });

  it("lists no route twice across both sections", () => {
    const paths = [...DECKS, ...SIMULATIONS].map((d) => d.path);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it("only lists routes the router actually serves", async () => {
    // A listing that 404s is worse than no listing: the card looks live and leads nowhere.
    //
    // Two ways a path can be served: an explicit <Route> in App.tsx, or a deck registered into
    // the registry and picked up by the generic DeckRoute catch-all. Both count.
    await import("../App.js");
    const { listDecks } = await import("../deck/registry.js");
    const registered = new Set(listDecks().map((deck) => `/${deck.id}`));
    // Vitest runs from the package root, so the router source is readable from there.
    const source = readFileSync("src/App.tsx", "utf8");

    for (const { path } of [...DECKS, ...SIMULATIONS]) {
      const served = source.includes(`path="${path}"`) || registered.has(path);
      expect(served, `nothing serves ${path}`).toBe(true);
    }
  });
});

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\—]/g, "\\$&");
}
