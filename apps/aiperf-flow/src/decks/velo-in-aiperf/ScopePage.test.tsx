/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScopePage } from "./ScopePage.js";

describe("ScopePage", () => {
  it("starts at four ticks with issued/completed counters and full liveness", () => {
    render(<ScopePage />);
    expect(screen.getByText("48")).toBeInTheDocument(); // issued = 4 * 12
    expect(screen.getByText("44")).toBeInTheDocument(); // completed = 4 * 11
    expect(screen.getByText("3 / 3")).toBeInTheDocument();
    expect(screen.getByText("CH 0 / cell 0")).toBeInTheDocument();
  });

  it("increments counters when a heartbeat is emitted", () => {
    render(<ScopePage />);
    fireEvent.click(screen.getByRole("button", { name: "Emit heartbeat" }));
    expect(screen.getByText("60")).toBeInTheDocument(); // 5 * 12
    expect(screen.getByText("55")).toBeInTheDocument(); // 5 * 11
  });

  it("marks cell 2 as missing when failed", () => {
    render(<ScopePage />);
    fireEvent.click(screen.getByRole("button", { name: "Fail cell 2" }));
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
    expect(screen.getByText("lag ↑ · pulse missing")).toBeInTheDocument();
  });
});
