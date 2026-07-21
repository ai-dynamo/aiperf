/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WekaTimingCausalityDeck } from "./WekaTimingCausalityDeck.js";

describe("WekaTimingCausalityDeck", () => {
  it("renders the title, design pills, and lossy-trace framing copy", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("Weka interval-order causality")).toBeInTheDocument();
    expect(screen.getByText("design")).toBeInTheDocument();
    expect(screen.getByText("synthesize mode")).toBeInTheDocument();
    expect(screen.getByText(/records timestamps, KV-block hashes, and/)).toBeInTheDocument();
  });

  it("renders both replaced failure modes", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("Racing siblings get chained")).toBeInTheDocument();
    expect(screen.getByText("Parentless spawn chains collapse")).toBeInTheDocument();
    expect(screen.getByText("One root assumption")).toBeInTheDocument();
  });

  it("defaults the interval-order lab to the AND-join case (C0 after A0 and B0)", () => {
    render(<WekaTimingCausalityDeck />);
    // Default c0start = 5.2: B0 [1.3,5.0] has finished, so C0 AND-joins A0 + B0.
    expect(screen.getByText("C0 starts after A0 and B0 — AND-join")).toBeInTheDocument();
    expect(screen.getByText("5.2s")).toBeInTheDocument();
    expect(screen.getByText("C0 fan-in width")).toBeInTheDocument();
    // Edge table binding-cause column reflects the worked example.
    expect(screen.getByText("Finished-before frontier")).toBeInTheDocument();
    expect(screen.getByText("delay_after_pred")).toBeInTheDocument();
  });

  it("flips C0 into overlap with B0 when the slider slides it earlier", () => {
    render(<WekaTimingCausalityDeck />);
    const slider = screen.getByLabelText("C0 start");
    // c0start = 4.0: C0 [4.0,5.8] overlaps B0 [1.3,5.0], so B0 -> C0 is dropped.
    fireEvent.change(slider, { target: { value: "4" } });
    expect(screen.getByText("C0 overlaps B0 — concurrent, no edge")).toBeInTheDocument();
  });

  it("marks B0 async so it is excluded as an out-of-subtree predecessor", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("Controls — all blocking")).toBeInTheDocument();
    const toggles = screen.getAllByRole("switch");
    // The first switch in the lab controls B0 async.
    fireEvent.click(toggles[0]);
    expect(screen.getByText("Controls — B0 async-launched")).toBeInTheDocument();
  });

  it("renders the rank tie-break demo, toggling between single edge and 2-cycle deadlock", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("single edge: X -> Y (rank by node_id)")).toBeInTheDocument();
    const rankToggle = screen.getByText("rank tie-break").closest("div")?.querySelector('[role="switch"]');
    expect(rankToggle).not.toBeNull();
    fireEvent.click(rankToggle as Element);
    expect(screen.getByText("2-cycle: await_inputs deadlock")).toBeInTheDocument();
  });

  it("renders the content-contract section with old/new headers and the divergence legend", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("advance_turn relabels block 20 — per-turn (old)")).toBeInTheDocument();
    expect(screen.getByText("role fixed at creation — frozen per-block (new)")).toBeInTheDocument();
    expect(screen.getByText("block 20 (divergence point)")).toBeInTheDocument();
    expect(screen.getByText("Message-unit emission (not block-unit)")).toBeInTheDocument();
  });

  it("renders the data-flow pipeline nodes for both timing and content lanes", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("recorded requests")).toBeInTheDocument();
    expect(screen.getByText("global rank")).toBeInTheDocument();
    expect(screen.getByText("interval order")).toBeInTheDocument();
    expect(screen.getByText("scope-blind trie")).toBeInTheDocument();
    expect(screen.getByText("block-tag pass")).toBeInTheDocument();
    expect(screen.getByText("message-unit emission")).toBeInTheDocument();
  });

  it("renders the locked-regressions table with the 57f2a77e receipt topology row", () => {
    render(<WekaTimingCausalityDeck />);
    expect(screen.getByText("Locked regressions")).toBeInTheDocument();
    expect(screen.getByText("Zero-duration coincidence")).toBeInTheDocument();
    expect(screen.getByText("async_launched exclusion")).toBeInTheDocument();
    expect(screen.getByText(/57f2a77e receipt topology/)).toBeInTheDocument();
  });
});
