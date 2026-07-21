/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WorkloadForkSection } from "./WorkloadForkSection.js";

describe("WorkloadForkSection", () => {
  it("renders both workload columns and the join node", () => {
    render(<WorkloadForkSection detail="engineering" />);
    expect(screen.getByText("One execution core, two workload shapes")).toBeInTheDocument();
    expect(screen.getByText("Scheduled conversations")).toBeInTheDocument();
    expect(screen.getByText("Compiled trace graph")).toBeInTheDocument();
    expect(screen.getByText("TurnDispatcher")).toBeInTheDocument();
    expect(screen.getByText("GraphInputBundle")).toBeInTheDocument();
    expect(screen.getByText("worker-local dispatch")).toBeInTheDocument();
  });

  it("swaps the engineering footnote when graph is selected", () => {
    render(<WorkloadForkSection detail="engineering" />);
    expect(screen.getByText(/Warmup and profiling share/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Graph" }));
    expect(screen.getByText(/The handoff frontier is consumed once/)).toBeInTheDocument();
  });
});
