/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { WekaTrieBuildDeck } from "./WekaTrieBuildDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <WekaTrieBuildDeck />
    </ReactFlowProvider>,
  );
}

describe("WekaTrieBuildDeck", () => {
  it("renders the title and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Inside build_trie_graph")).toBeInTheDocument();
    expect(screen.getByText("_weka_trie_build.py")).toBeInTheDocument();
    expect(
      screen.getByText(/How one WekaTrace becomes a dependency-only ParsedGraph/),
    ).toBeInTheDocument();
  });

  it("renders all four passes with their real titles and copy", () => {
    renderDeck();
    expect(screen.getByText("build_trie_graph — four passes")).toBeInTheDocument();
    expect(screen.getByText("_flatten_requests")).toBeInTheDocument();
    expect(
      screen.getByText(/DFS every n\/s leaf in recorded t order/),
    ).toBeInTheDocument();
    expect(screen.getByText("_resolve_content_parents")).toBeInTheDocument();
    expect(
      screen.getByText(/incremental hash-id prefix trie/),
    ).toBeInTheDocument();
    expect(screen.getByText("_apply_idle_gap_warp")).toBeInTheDocument();
    expect(screen.getByText(/collapse true idle gaps to the cap/)).toBeInTheDocument();
    expect(screen.getByText("build node + edges")).toBeInTheDocument();
    expect(screen.getByText(/AND-fan-in inputs/)).toBeInTheDocument();
    expect(screen.getByText("Emitted IR is intentionally tiny")).toBeInTheDocument();
  });

  it("renders the prefix trie graph nodes from the canvas source", () => {
    renderDeck();
    expect(screen.getByText("root")).toBeInTheDocument();
    expect(screen.getByText("empty prefix")).toBeInTheDocument();
    expect(screen.getByText("hash A")).toBeInTheDocument();
    expect(screen.getByText("passer=r1 · terminal=r1")).toBeInTheDocument();
    expect(screen.getByText("hash B")).toBeInTheDocument();
    expect(screen.getByText("terminal=r2")).toBeInTheDocument();
    expect(screen.getByText("hash C")).toBeInTheDocument();
    expect(screen.getByText("terminal=r3")).toBeInTheDocument();
    expect(screen.getByText("hash D")).toBeInTheDocument();
    expect(screen.getByText("terminal=r4")).toBeInTheDocument();
  });

  it("renders the content-parent resolution picks", () => {
    renderDeck();
    expect(screen.getByText("Pass 2 — content-parent = hash-id prefix tree")).toBeInTheDocument();
    expect(screen.getByText("Resolution picks")).toBeInTheDocument();
    expect(screen.getByText("r2 [A, B]")).toBeInTheDocument();
    expect(screen.getAllByText("content_parent = r1").length).toBeGreaterThan(0);
    expect(
      screen.getByText("longest full prefix ([A] terminates at r1)"),
    ).toBeInTheDocument();
    expect(screen.getByText("r3 [A, C]")).toBeInTheDocument();
    expect(
      screen.getByText("no full prefix -> branch point via passer at depth 1"),
    ).toBeInTheDocument();
    expect(screen.getByText("r4 [A, B, D]")).toBeInTheDocument();
    expect(screen.getByText("content_parent = r2")).toBeInTheDocument();
    expect(screen.getByText("full prefix [A,B] beats [A]")).toBeInTheDocument();
    expect(screen.getByText("content_parent is content-only")).toBeInTheDocument();
  });

  it("renders the timing-edges section with all three cause cards", () => {
    renderDeck();
    expect(
      screen.getByText("Pass 4 — timing edges = completed-before waits-for"),
    ).toBeInTheDocument();
    expect(screen.getByText("Latest completed cause")).toBeInTheDocument();
    expect(
      screen.getByText(/delay_after_predecessor_us = max\(0, R.start − cause.end\)/),
    ).toBeInTheDocument();
    expect(screen.getByText("Other completed causes")).toBeInTheDocument();
    expect(screen.getByText(/one count=1 input on \{src\}_out/)).toBeInTheDocument();
    expect(screen.getByText("Nothing completed")).toBeInTheDocument();
    expect(
      screen.getByText(/Roots at START with min_start_delay_us = R.start/),
    ).toBeInTheDocument();
    expect(screen.getByText("Two planes, cleanly separated")).toBeInTheDocument();
  });
});
