/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { MetricsPage } from "./MetricsPage.js";

describe("MetricsPage", () => {
  it("renders the intro and measurement nodes", () => {
    render(
      <ReactFlowProvider>
        <MetricsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Measurement and exports")).toBeInTheDocument();
    expect(screen.getByText("ObserverTee")).toBeInTheDocument();
    expect(screen.getByText("MetricsAccumulator")).toBeInTheDocument();
    expect(screen.getByText("NativeReport")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <MetricsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Exact mode")).toBeInTheDocument();
    expect(screen.getByText("Separate artifact path")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/report.rs")).toBeInTheDocument();
  });
});
