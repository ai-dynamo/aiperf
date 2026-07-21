/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SchedulerPage } from "./SchedulerPage.js";

describe("SchedulerPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <SchedulerPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Scheduler and cache")).toBeInTheDocument();
    expect(screen.getByText("Prefill and decode stepping")).toBeInTheDocument();
    expect(screen.getByText("Batch saturation")).toBeInTheDocument();
    expect(
      screen.getAllByText(
        "Scheduler ticks admit prefill work and emit decode tokens under configured capacities.",
      ).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("rust/mock-server/src/prefix_cache.rs").length).toBeGreaterThan(0);
  });
});
