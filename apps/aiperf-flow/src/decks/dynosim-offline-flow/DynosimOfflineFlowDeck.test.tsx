/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { DynosimOfflineFlowDeck } from "./DynosimOfflineFlowDeck.js";

describe("DynosimOfflineFlowDeck", () => {
  it("renders the Overview page by default", () => {
    render(<DynosimOfflineFlowDeck />);
    expect(screen.getByText("How it fits together")).toBeInTheDocument();
  });

  it("switches to the Launch page when its tab is clicked", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));
    expect(screen.getByText("Launch & preflight")).toBeInTheDocument();
  });

  it("switches to the Architecture page when its tab is clicked", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Architecture" }));
    expect(screen.getByText("System architecture — the two seams")).toBeInTheDocument();
  });

  it("switches to the Loop page when its tab is clicked", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Loop" }));
    expect(screen.getByText("The simulation loop")).toBeInTheDocument();
  });

  it("switches to the Dispatch page when its tab is clicked", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Dispatch" }));
    expect(screen.getByText("Request → tokens → engine")).toBeInTheDocument();
  });

  it("switches to the Parity page when its tab is clicked", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Parity" }));
    expect(screen.getByText("The verification gate")).toBeInTheDocument();
  });

  it("switches to the Engine page when its tab is clicked", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Engine" }));
    expect(screen.getByText("Engine internals — topology builder")).toBeInTheDocument();
  });

  it("the detail toggle changes visible content across pages", () => {
    render(<DynosimOfflineFlowDeck />);
    fireEvent.click(screen.getByRole("button", { name: "maintainer" }));
    expect(screen.getByText("load.rs / yaml.rs")).toBeInTheDocument();
  });
});
