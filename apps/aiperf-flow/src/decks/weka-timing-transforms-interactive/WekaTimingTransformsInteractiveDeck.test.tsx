/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WekaTimingTransformsInteractiveDeck } from "./WekaTimingTransformsInteractiveDeck.js";

describe("WekaTimingTransformsInteractiveDeck", () => {
  it("renders the title and every source file reference", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    expect(screen.getByText("Weka timing transforms")).toBeInTheDocument();
    expect(screen.getByText("_weka_trie_build.py")).toBeInTheDocument();
    expect(screen.getByText("graph_ir_replay.py")).toBeInTheDocument();
    expect(screen.getByText("step_emit_weka.py")).toBeInTheDocument();
  });

  it("renders the six section headings from the source canvas", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    expect(screen.getByText("The pipeline, stage by stage")).toBeInTheDocument();
    expect(screen.getByText("Idle-gap warp lab")).toBeInTheDocument();
    expect(screen.getByText("Interval-order edges & binding delay")).toBeInTheDocument();
    // "t* snapshot chop" is also a pipeline stage name (StageExplorer's detail card can echo
    // it), so disambiguate via the section <h2> specifically.
    expect(screen.getByRole("heading", { level: 2, name: "t* snapshot chop" })).toBeInTheDocument();
    expect(screen.getByText("Independent t* across three traces")).toBeInTheDocument();
    expect(screen.getByText("Combined timeline — all traces aligned at t*")).toBeInTheDocument();
  });

  it("defaults to the agent scenario with its default 60s cap", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    expect(screen.getByText("cap = 60s")).toBeInTheDocument();
    expect(screen.getByText("60s (default)")).toBeInTheDocument();
  });

  it("switching scenario via the Select updates the idle-gap-cut stat", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    // agent scenario has one idle gap cut at the default 60s cap.
    expect(screen.getByText("idle gaps cut")).toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue("Agent session (subagents past a long idle)"), {
      target: { value: "dense" },
    });
    // dense scenario has no gap over the default cap -> 0 idle gaps cut.
    const stats = screen.getAllByText("0");
    expect(stats.length).toBeGreaterThan(0);
  });

  it("toggling Warp off relabels the timeline as no-cap passthrough", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    expect(screen.getByText("active-interval idle capping on")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("switch"));
    expect(screen.getByText("no-cap passthrough (warped_start = raw t)")).toBeInTheDocument();
    expect(screen.getByText("warp off")).toBeInTheDocument();
  });

  it("renders one MiniTraceChop card per mini trace with independent labels", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    expect(screen.getByText("Linear chat (single lane)")).toBeInTheDocument();
    expect(screen.getByText("One subagent")).toBeInTheDocument();
    expect(screen.getByText("Two overlapping subagents")).toBeInTheDocument();
  });

  it("Reset restores the idle cap, warp toggle, and t* to their defaults", () => {
    render(<WekaTimingTransformsInteractiveDeck />);
    fireEvent.click(screen.getByRole("switch"));
    expect(screen.getByText("warp off")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("cap = 60s")).toBeInTheDocument();
  });
});
