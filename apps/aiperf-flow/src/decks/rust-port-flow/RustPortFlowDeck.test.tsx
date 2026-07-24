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

describe("RustPortFlowDeck", () => {
  it("renders the overview with all nine spine stage labels", () => {
    render(<RustPortFlowDeck />);
    for (const label of STAGE_LABELS) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
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

  it("advances the play head through the pipeline when Next is clicked", () => {
    render(<RustPortFlowDeck />);
    expect(screen.getAllByText("step 1/9").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getAllByText("step 2/9").length).toBeGreaterThan(0);
    // The now-active step caption names the real Runtime stage content.
    expect(screen.getAllByText(/re-exec of the same binary/).length).toBeGreaterThan(0);
  });

  it("drills into a stage on node click, revealing its real source anchor", () => {
    render(<RustPortFlowDeck />);
    fireEvent.click(screen.getAllByText("Clock seam")[0]!);
    // Level-1 evidence row shows the verified Clock trait anchor.
    expect(screen.getByText("runtime/src/clock/clock.rs:12")).toBeInTheDocument();
    // Breadcrumb now offers a way back to the overview.
    expect(screen.getByRole("button", { name: "Big-picture request lifecycle" })).toBeInTheDocument();
  });

  it("re-routes the transport step caption when the Transport seam changes", () => {
    render(<RustPortFlowDeck />);
    // Jump the play head to the Transport stage (spine order 6).
    for (let i = 0; i < 6; i++) {
      fireEvent.click(screen.getByRole("button", { name: "Next" }));
    }
    expect(screen.getAllByText("step 7/9").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/TransportSink \(hyper, streaming\)/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "gRPC" }));
    expect(screen.getAllByText(/GrpcTransportSink \(Tonic, non-streaming\)/).length).toBeGreaterThan(0);
  });

  it("is registered on Home's deck listing", () => {
    expect(DECKS.some((deck) => deck.path === "/rust-port-flow")).toBe(true);
  });

  it("plays a full request lifecycle assembled from the real hot-path + aggregation fragments", () => {
    render(<RustPortFlowDeck />);
    // The lifecycle particle starts on the issue hop (10 hops: issue → … → terminal report).
    expect(screen.getAllByText("hop 1/10").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(/RequestRateWorkload::execute issues the next scheduled turn/).length,
    ).toBeGreaterThan(0);
    // Advance to the SlotPool admission hop.
    fireEvent.click(screen.getByRole("button", { name: "Next token" }));
    expect(screen.getAllByText(/SlotPool grants a concurrency credit/).length).toBeGreaterThan(0);
    // …to the Dispatcher hop, then the chosen (default HTTP) sink hop — SSE tokens + TTFT.
    fireEvent.click(screen.getByRole("button", { name: "Next token" }));
    fireEvent.click(screen.getByRole("button", { name: "Next token" }));
    expect(screen.getAllByText(/TransportSink \(hyper\) streams SSE tokens/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("TransportSink (HTTP · hyper)").length).toBeGreaterThan(0);
  });

  it("reroutes the request-lifecycle sink hop when the Transport seam changes", () => {
    render(<RustPortFlowDeck />);
    // Swap the transport target, then step the lifecycle particle to its sink hop.
    fireEvent.click(screen.getByRole("button", { name: "dynosim" }));
    fireEvent.click(screen.getByRole("button", { name: "Next token" }));
    fireEvent.click(screen.getByRole("button", { name: "Next token" }));
    fireEvent.click(screen.getByRole("button", { name: "Next token" }));
    // The same particle is now routed through the dynosim SteppableEngine sink.
    expect(
      screen.getAllByText(/NativeDynamoEngineFactory builds a SteppableEngine/).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("SteppableEngine (dynosim)").length).toBeGreaterThan(0);
  });
});
