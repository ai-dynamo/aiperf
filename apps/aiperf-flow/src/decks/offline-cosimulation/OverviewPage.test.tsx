/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { OverviewPage } from "./OverviewPage.js";

function renderOverview() {
  return render(
    <ReactFlowProvider>
      <OverviewPage />
    </ReactFlowProvider>,
  );
}

describe("OverviewPage", () => {
  it("renders the orchestration band and request-path nodes in offline mode", () => {
    renderOverview();
    expect(
      screen.getByText("AIPERF OWNS ORCHESTRATION, CLOCK, AND MEASUREMENT"),
    ).toBeInTheDocument();
    expect(screen.getByText("Config v2")).toBeInTheDocument();
    expect(screen.getByText("transport.type: dynosim_offline")).toBeInTheDocument();
    expect(screen.getByText("AIPerf run loop")).toBeInTheDocument();
    expect(screen.getByText("SimClock")).toBeInTheDocument();
    expect(screen.getByText("integer-ns virtual time")).toBeInTheDocument();
  });

  it("swaps the clock and transport.type when switching to online mode", () => {
    renderOverview();
    fireEvent.click(screen.getByRole("button", { name: "dynosim_online" }));
    expect(screen.getByText("transport.type: dynosim_online")).toBeInTheDocument();
    expect(screen.getByText("RealClock")).toBeInTheDocument();
    expect(screen.getByText("wall-clock replay")).toBeInTheDocument();
  });

  it("renders the engine boundary and observer nodes", () => {
    renderOverview();
    expect(screen.getByText("Steppable engine boundary")).toBeInTheDocument();
    expect(screen.getByText("step_until · next_event_ms")).toBeInTheDocument();
    expect(screen.getByText("RequestObserver")).toBeInTheDocument();
    expect(screen.getByText("shared Level-B contract")).toBeInTheDocument();
    expect(screen.getByText("No sockets")).toBeInTheDocument();
    expect(screen.getByText("in-process Dynamo mocker")).toBeInTheDocument();
  });

  it("renders the four observer-stream consumers", () => {
    renderOverview();
    expect(
      screen.getByText("THE SAME OBSERVER STREAM POWERS EVERY CONSUMER"),
    ).toBeInTheDocument();
    expect(screen.getByText("Native v2 report")).toBeInTheDocument();
    expect(screen.getByText("Streaming metrics")).toBeInTheDocument();
    expect(screen.getByText("Adaptive windows")).toBeInTheDocument();
    expect(screen.getByText("Live dashboard")).toBeInTheDocument();
  });

  it("renders the TTFT footer callout and the two calibration callouts", () => {
    renderOverview();
    expect(
      screen.getByText("first on_token = TTFT + prefill release + graph first-token gate"),
    ).toBeInTheDocument();
    expect(screen.getByText("One driver contract, two clocks")).toBeInTheDocument();
    expect(screen.getByText("Acceptance invariant")).toBeInTheDocument();
    expect(screen.getByText(/bit-for-bit on handoff fixtures/)).toBeInTheDocument();
  });
});
