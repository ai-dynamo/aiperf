/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RustPortFlowDeck, STAGES } from "./RustPortFlowDeck.js";
import { DECKS } from "../../routes/Home.js";

const STAGE_LABELS = [
  "Big Picture",
  "Runtime & self-exec",
  "Dataset loading",
  "Sharing the dataset",
  "Workers sync & connect",
  "Clock seam",
  "Transport seam",
  "Request hot-path",
  "Aggregation → results",
];

const LANE_LABELS = ["Dataset", "Scheduler / Workload", "Transport", "Server", "Aggregate", "Export"];

describe("RustPortFlowDeck (v2 swimlane-timeline)", () => {
  it("renders the timeline overview with all six subsystem lanes", () => {
    render(<RustPortFlowDeck />);
    for (const lane of LANE_LABELS) {
      expect(screen.getAllByText(lane).length).toBeGreaterThan(0);
    }
  });

  it("renders all nine spine stage region labels on the timeline", () => {
    render(<RustPortFlowDeck />);
    for (const label of STAGE_LABELS) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("renders the Workload + Transport seam frames grouping the track", () => {
    render(<RustPortFlowDeck />);
    // The Workload seam frames the scheduler admission segment; its label is unique to the frame.
    expect(screen.getAllByText("Workload").length).toBeGreaterThan(0);
    // The Transport seam frame shares its label with the lane + toggle — all present.
    expect(screen.getAllByText("Transport").length).toBeGreaterThan(0);
  });

  it("exposes exactly nine stages, one per spine ordinal 0-8", () => {
    expect(STAGES).toHaveLength(9);
    expect([...STAGES].map((s) => s.order).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("renders the Clock and Transport seam toggles and play controls", () => {
    render(<RustPortFlowDeck />);
    expect(screen.getByRole("button", { name: "RealClock" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "SimClock" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "HTTP" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "gRPC" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
  });

  it("rides the request line: Next advances the play head through the 16 request events", () => {
    render(<RustPortFlowDeck />);
    expect(screen.getAllByText("step 1/16").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getAllByText("step 2/16").length).toBeGreaterThan(0);
    // Step 2 is the runtime self-exec event; its caption names the real re-exec composition root.
    expect(screen.getAllByText(/re-exec of the same binary/).length).toBeGreaterThan(0);
  });

  it("rescales the timeline x-axis when the Clock seam changes (real wall-ms ↔ virtual ticks)", () => {
    render(<RustPortFlowDeck />);
    expect(screen.getByText("RealClock · wall-ms")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "SimClock" }));
    expect(screen.getByText("SimClock · virtual ticks")).toBeInTheDocument();
  });

  it("drills into a stage on region click, revealing its real source anchor", () => {
    render(<RustPortFlowDeck />);
    fireEvent.click(screen.getAllByText("Clock seam")[0]!);
    // Level-1 evidence row shows the verified Clock trait anchor.
    expect(screen.getByText("runtime/src/clock/clock.rs:12")).toBeInTheDocument();
    // Breadcrumb now offers a way back to the overview.
    expect(screen.getByRole("button", { name: "Big-picture request lifecycle" })).toBeInTheDocument();
  });

  it("re-routes the dispatch event caption when the Transport seam changes", () => {
    render(<RustPortFlowDeck />);
    // Advance the play head to the transport dispatch event (path index 8 → step 9/16).
    for (let i = 0; i < 8; i++) {
      fireEvent.click(screen.getByRole("button", { name: "Next" }));
    }
    expect(screen.getAllByText("step 9/16").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/TransportSink \(hyper, streaming\)/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "gRPC" }));
    expect(screen.getAllByText(/GrpcTransportSink \(Tonic, non-streaming\)/).length).toBeGreaterThan(0);
  });

  it("is registered on Home's deck listing", () => {
    expect(DECKS.some((deck) => deck.path === "/rust-port-flow")).toBe(true);
  });

  it("tiles stage blocks so none overlap within a lane (real wall-ms scale)", () => {
    const { container } = render(<RustPortFlowDeck />);
    // Each stage block is a <g data-testid="stage-region"> wrapping one <rect>. Group the rects by
    // their y (which encodes the lane row) and assert the x-intervals in each row are disjoint —
    // this is the regression guard for the wall-ms "left-crush" that stacked the setup blocks.
    const rects = Array.from(
      container.querySelectorAll<SVGRectElement>('[data-testid="stage-region"] > rect'),
    ).map((r) => ({
      x: Number(r.getAttribute("x")),
      w: Number(r.getAttribute("width")),
      y: Number(r.getAttribute("y")),
    }));
    expect(rects.length).toBe(STAGE_LABELS.length);
    const byRow = new Map<number, { x: number; w: number }[]>();
    for (const r of rects) {
      const row = byRow.get(r.y) ?? [];
      row.push({ x: r.x, w: r.w });
      byRow.set(r.y, row);
    }
    for (const row of byRow.values()) {
      const sorted = [...row].sort((a, b) => a.x - b.x);
      for (let i = 1; i < sorted.length; i++) {
        // The left edge of each block starts at/after the previous block's right edge.
        expect(sorted[i]!.x).toBeGreaterThanOrEqual(sorted[i - 1]!.x + sorted[i - 1]!.w - 0.5);
      }
    }
  });
});
