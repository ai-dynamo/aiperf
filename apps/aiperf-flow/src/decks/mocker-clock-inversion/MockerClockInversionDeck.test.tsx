/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MockerClockInversionDeck } from "./MockerClockInversionDeck.js";

function renderDeck(): void {
  render(<MockerClockInversionDeck />);
}

describe("MockerClockInversionDeck", () => {
  it("renders the title and framing copy", () => {
    renderDeck();
    expect(screen.getByText("AIPerf takes ownership of simulation time")).toBeInTheDocument();
    expect(screen.getByText(/The key change is an outer-loop inversion/)).toBeInTheDocument();
    expect(screen.getByText("AIPerf ajc/rust")).toBeInTheDocument();
  });

  it("renders the before/after loop diagrams", () => {
    renderDeck();
    expect(screen.getByText("Mocker legacy offline replay")).toBeInTheDocument();
    expect(screen.getByText("AIPerf dynosim_offline")).toBeInTheDocument();
    expect(screen.getAllByText("Static arrival queue").length).toBeGreaterThan(0);
    expect(screen.getAllByText("run_to_completion()").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TraceSimulationReport").length).toBeGreaterThan(0);
    expect(screen.getAllByText("run_paced_with_backend").length).toBeGreaterThan(0);
    expect(screen.getAllByText("DynosimSink + EngineHost").length).toBeGreaterThan(0);
    expect(screen.getAllByText("SteppableReplay").length).toBeGreaterThan(0);
  });

  it("renders the inversion callout", () => {
    renderDeck();
    expect(screen.getByText("What inversion means here")).toBeInTheDocument();
    expect(screen.getByText(/AIPerf owns the clock object and arbitration policy/)).toBeInTheDocument();
  });

  it("renders the first arbitration frame by default", () => {
    renderDeck();
    expect(screen.getByText("Example: R1 is running; AIPerf schedules R2 for 10 ms")).toBeInTheDocument();
    expect(screen.getByText("AIPerf runs until its futures park")).toBeInTheDocument();
    expect(screen.getByText("1 / 7")).toBeInTheDocument();
    expect(screen.getByText("R2 arrival · 10 ms")).toBeInTheDocument();
    expect(screen.getByText("R1 work · 0 ms")).toBeInTheDocument();
    expect(
      screen.getByText(/run_paced_with_backend dispatches R1 through DynosimSink/),
    ).toBeInTheDocument();
  });

  it("steps forward through the arbitration cycle on Next event", () => {
    renderDeck();
    fireEvent.click(screen.getByRole("button", { name: "Next event" }));
    expect(screen.getByText("The engine is ready before AIPerf’s next timer")).toBeInTheDocument();
    expect(screen.getByText("2 / 7")).toBeInTheDocument();
    expect(
      screen.getByText(/The pump compares SimClock::next_event_time with EngineHost::next_event_ns/),
    ).toBeInTheDocument();
  });

  it("steps back and resets", () => {
    renderDeck();
    fireEvent.click(screen.getByRole("button", { name: "Next event" }));
    fireEvent.click(screen.getByRole("button", { name: "Next event" }));
    expect(screen.getByText("3 / 7")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(screen.getByText("2 / 7")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("1 / 7")).toBeInTheDocument();
  });

  it("reaches the final frame and disables Next event", () => {
    renderDeck();
    for (let i = 0; i < 6; i++) {
      fireEvent.click(screen.getByRole("button", { name: "Next event" }));
    }
    expect(screen.getByText("Repeat until the workload resolves, then drain")).toBeInTheDocument();
    expect(screen.getByText("7 / 7")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next event" })).toBeDisabled();
  });

  it("renders the actual ajc/rust composition chain", () => {
    renderDeck();
    expect(screen.getByText("The actual `ajc/rust` composition")).toBeInTheDocument();
    expect(screen.getByText("run_paced_offline")).toBeInTheDocument();
    expect(screen.getAllByText("drive_sim_with_source").length).toBeGreaterThan(0);
    expect(screen.getAllByText("EngineHost").length).toBeGreaterThan(0);
    expect(screen.getByText("AIPerf keeps")).toBeInTheDocument();
    expect(screen.getByText("Mocker keeps")).toBeInTheDocument();
    expect(screen.getByText("The seam adds")).toBeInTheDocument();
    expect(screen.getByText(/Workload policy, pacing, concurrency, phases/)).toBeInTheDocument();
    expect(screen.getByText(/Dynamic submit, next-event visibility/)).toBeInTheDocument();
  });

  it("renders the source file references", () => {
    renderDeck();
    expect(screen.getByText(/AIPerf sim pump — rust\/runtime\/src\/graph\/runtime\.rs:205/)).toBeInTheDocument();
    expect(screen.getByText(/dynosim_offline — rust\/runtime\/src\/dynosim\.rs:2387/)).toBeInTheDocument();
    expect(screen.getByText(/Mocker seam — lib\/mocker\/src\/loadgen\/steppable\.rs:77/)).toBeInTheDocument();
  });
});
