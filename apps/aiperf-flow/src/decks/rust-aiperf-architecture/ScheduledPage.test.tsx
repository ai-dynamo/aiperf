/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ScheduledPage } from "./ScheduledPage.js";

describe("ScheduledPage", () => {
  it("renders the intro and workload nodes", () => {
    render(
      <ReactFlowProvider>
        <ScheduledPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Paced workload path")).toBeInTheDocument();
    expect(screen.getByText("PhaseOrchestrator")).toBeInTheDocument();
    expect(screen.getByText("SlotPool + StopChecker")).toBeInTheDocument();
    expect(screen.getByText("workers > 1")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <ScheduledPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Same workload ID")).toBeInTheDocument();
    expect(screen.getByText("Local hot path")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/phase_runtime.rs")).toBeInTheDocument();
  });
});
