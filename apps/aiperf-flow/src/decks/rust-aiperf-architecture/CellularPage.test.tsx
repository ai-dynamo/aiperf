/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { CellularPage } from "./CellularPage.js";

describe("CellularPage", () => {
  it("renders the intro and cellular nodes", () => {
    render(
      <ReactFlowProvider>
        <CellularPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Multi-process scale")).toBeInTheDocument();
    expect(screen.getByText("cell launcher")).toBeInTheDocument();
    expect(screen.getByText("controller merge")).toBeInTheDocument();
    expect(screen.getByText("aiperf --cell 0")).toBeInTheDocument();
  });

  it("renders the numbered callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <CellularPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("S1")).toBeInTheDocument();
    expect(screen.getByText("S4")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/cellular/mod.rs")).toBeInTheDocument();
  });
});
