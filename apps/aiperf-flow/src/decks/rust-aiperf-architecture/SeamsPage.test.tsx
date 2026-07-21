/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SeamsPage } from "./SeamsPage.js";

describe("SeamsPage", () => {
  it("renders the intro and both sub-diagram headings", () => {
    render(
      <ReactFlowProvider>
        <SeamsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Extension internals")).toBeInTheDocument();
    expect(screen.getByText("Compile-time extension universe")).toBeInTheDocument();
    expect(screen.getByText("Execution substitution")).toBeInTheDocument();
    expect(screen.getByText("Cellular scaling wraps the same run core")).toBeInTheDocument();
  });

  it("renders the wide cellular-scaling nodes, callouts, and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <SeamsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getAllByText("optional aggregators").length).toBeGreaterThan(0);
    expect(screen.getAllByText("controller process").length).toBeGreaterThan(0);
    expect(screen.getByText("No runtime plugin discovery")).toBeInTheDocument();
    expect(screen.getByText("No pair matrix")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/extensions/mod.rs")).toBeInTheDocument();
  });
});
