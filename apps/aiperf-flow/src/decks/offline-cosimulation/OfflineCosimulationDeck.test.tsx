/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { OfflineCosimulationDeck } from "./OfflineCosimulationDeck.js";

describe("OfflineCosimulationDeck", () => {
  it("renders the top bar, title, subtitle, and dynosim pill", () => {
    render(<OfflineCosimulationDeck />);
    expect(screen.getByText("Offline Co-simulation")).toBeInTheDocument();
    expect(screen.getByText("Offline co-simulation")).toBeInTheDocument();
    expect(
      screen.getByText("Socket-free Dynamo execution through AIPerf's native measurement path"),
    ).toBeInTheDocument();
    expect(screen.getByText("dynosim feature")).toBeInTheDocument();
  });

  it("defaults to the Overview page", () => {
    render(<OfflineCosimulationDeck />);
    expect(
      screen.getByText("AIPERF OWNS ORCHESTRATION, CLOCK, AND MEASUREMENT"),
    ).toBeInTheDocument();
  });

  it("switches to the Internals page via the tabs", () => {
    render(<OfflineCosimulationDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Internals" }));
    expect(screen.getByText("EngineHost : SimEventSource")).toBeInTheDocument();
    expect(screen.getByText("Level-B observer contract")).toBeInTheDocument();
  });
});
