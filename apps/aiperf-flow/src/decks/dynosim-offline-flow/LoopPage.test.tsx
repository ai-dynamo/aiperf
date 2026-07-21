/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LoopPage } from "./LoopPage.js";

describe("LoopPage", () => {
  it("starts on frame 1 of 8 with the Poll caption", () => {
    render(<LoopPage level="developer" />);
    expect(screen.getByText("1 / 8")).toBeInTheDocument();
    expect(screen.getByText(/Poll the workload to quiescence/)).toBeInTheDocument();
  });

  it("clicking Step advances the frame and caption", () => {
    render(<LoopPage level="developer" />);
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(screen.getByText("2 / 8")).toBeInTheDocument();
    expect(screen.getByText(/Compare the next clock sleeper/)).toBeInTheDocument();
  });

  it("Reset returns to frame 1 after stepping forward", () => {
    render(<LoopPage level="developer" />);
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("1 / 8")).toBeInTheDocument();
  });

  it("shows maintainer stage sub-labels only at maintainer level", () => {
    render(<LoopPage level="developer" />);
    expect(screen.queryByText("wake waiters")).not.toBeInTheDocument();

    render(<LoopPage level="maintainer" />);
    expect(screen.getByText("wake waiters")).toBeInTheDocument();
  });

  it("reaches the final Route frame after stepping to the end", () => {
    render(<LoopPage level="developer" />);
    for (let i = 0; i < 7; i++) {
      fireEvent.click(screen.getByRole("button", { name: "Step" }));
    }
    expect(screen.getByText("8 / 8")).toBeInTheDocument();
    expect(screen.getByText(/StopChecker's bound is met/)).toBeInTheDocument();
  });
});
