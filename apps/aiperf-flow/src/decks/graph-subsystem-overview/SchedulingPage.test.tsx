/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SchedulingPage } from "./SchedulingPage.js";

describe("SchedulingPage", () => {
  it("shows the default timeline predecessors for r4 and updates on selection", () => {
    render(<SchedulingPage />);
    // r4 default: predecessors r1, r2.
    expect(screen.getByText(/r4's finished-before predecessors are/)).toBeInTheDocument();
    expect(screen.getByText("r1, r2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "r1" }));
    expect(screen.getByText(/none — it roots at START/)).toBeInTheDocument();
  });

  it("renders the interval frontier explanation", () => {
    render(<SchedulingPage />);
    expect(screen.getByText(/transitively covered/)).toBeInTheDocument();
    expect(screen.getByText("binding")).toBeInTheDocument();
  });

  it("toggles the idle-gap warp", () => {
    render(<SchedulingPage />);
    expect(screen.getByText("Idle gaps capped")).toBeInTheDocument();
    const warpToggle = screen.getAllByRole("switch")[0]!;
    fireEvent.click(warpToggle);
    expect(screen.getByText("Raw recorded gaps")).toBeInTheDocument();
  });

  it("changes the t* split point", () => {
    render(<SchedulingPage />);
    fireEvent.click(screen.getByRole("button", { name: "8" }));
    expect(screen.getByText(/Nodes before/)).toBeInTheDocument();
  });
});
