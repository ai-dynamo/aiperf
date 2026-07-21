/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SeamTheaterSection } from "./SeamTheaterSection.js";

describe("SeamTheaterSection", () => {
  it("renders the workload orchestration and default HTTP sink", () => {
    render(<SeamTheaterSection detail="engineering" />);
    expect(screen.getByText("Clock and request dispatch are concrete substitution points")).toBeInTheDocument();
    expect(screen.getByText("Workload orchestration")).toBeInTheDocument();
    expect(screen.getByText("TransportSink")).toBeInTheDocument();
    expect(screen.getByText("RequestSink<Request>")).toBeInTheDocument();
  });

  it("swaps the sink and dispatch seam when DynoSim is picked", () => {
    render(<SeamTheaterSection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "DynoSim" }));
    expect(screen.getByText("DynosimExecutor")).toBeInTheDocument();
    expect(screen.getByText("PreparedRunnerOperation")).toBeInTheDocument();
  });

  it("renders the TTFT-derivation callout", () => {
    render(<SeamTheaterSection detail="engineering" />);
    expect(screen.getByText("TTFT derivation")).toBeInTheDocument();
  });
});
