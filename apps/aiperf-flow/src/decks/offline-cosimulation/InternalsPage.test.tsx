/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { InternalsPage } from "./InternalsPage.js";

function renderInternals() {
  return render(
    <ReactFlowProvider>
      <InternalsPage />
    </ReactFlowProvider>,
  );
}

describe("InternalsPage", () => {
  it("renders the layered AIPERF -> DYNAMO MOCKER stack", () => {
    renderInternals();
    expect(screen.getByText("AIPERF")).toBeInTheDocument();
    expect(screen.getByText("DYNAMO MOCKER (PASSIVE)")).toBeInTheDocument();
    expect(screen.getAllByText("ScheduledRuntime").length).toBeGreaterThan(0);
    expect(screen.getByText("EngineHost : SimEventSource")).toBeInTheDocument();
    expect(screen.getByText("next_event_ns · set_time_ns · step · route")).toBeInTheDocument();
    expect(screen.getByText("SteppableAgg")).toBeInTheDocument();
    expect(screen.getByText("SteppableDisagg")).toBeInTheDocument();
    expect(screen.getByText("scalar now")).toBeInTheDocument();
    expect(
      screen.getByText(/scheduler math unchanged; only driver loops/),
    ).toBeInTheDocument();
  });

  it("starts the drive loop at the Poll frame", () => {
    renderInternals();
    expect(screen.getByText("1 / 8")).toBeInTheDocument();
    // "Poll" appears both as the stage tile and the active-frame callout title.
    expect(screen.getAllByText("Poll").length).toBeGreaterThanOrEqual(2);
    expect(
      screen.getByText(/Admitted turns submit token arrays into the steppable engine/),
    ).toBeInTheDocument();
  });

  it("advances the drive loop through its frames on Step", () => {
    renderInternals();
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(screen.getByText("2 / 8")).toBeInTheDocument();
    expect(screen.getAllByText("Compare").length).toBeGreaterThanOrEqual(2);
    expect(
      screen.getByText(/compares the next clock sleeper against EngineHost.next_event_ns/),
    ).toBeInTheDocument();
  });

  it("Back is disabled at the first frame and Reset returns to it", () => {
    renderInternals();
    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(screen.getByText("3 / 8")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("1 / 8")).toBeInTheDocument();
  });

  it("renders the virtual-time ruler with a SimClock pill", () => {
    renderInternals();
    expect(screen.getByText("Virtual time")).toBeInTheDocument();
    expect(screen.getByText("SimClock")).toBeInTheDocument();
    expect(screen.getByText("22 ms")).toBeInTheDocument();
  });

  it("renders the Level-B observer pipeline with on_token emphasized", () => {
    renderInternals();
    expect(screen.getByText("Level-B observer contract")).toBeInTheDocument();
    expect(screen.getByText("on_arrival")).toBeInTheDocument();
    expect(screen.getByText("on_admit")).toBeInTheDocument();
    expect(screen.getByText("on_token")).toBeInTheDocument();
    expect(screen.getByText("TTFT = first")).toBeInTheDocument();
    expect(screen.getByText("on_usage")).toBeInTheDocument();
    expect(screen.getByText("on_terminal")).toBeInTheDocument();
    expect(
      screen.getByText(/ObserverTee fans each callback to CollectorObserver/),
    ).toBeInTheDocument();
  });

  it("renders the two closing callouts", () => {
    renderInternals();
    expect(screen.getByText("Engine never sees Clock")).toBeInTheDocument();
    expect(screen.getByText("Live, not post-hoc")).toBeInTheDocument();
  });
});
