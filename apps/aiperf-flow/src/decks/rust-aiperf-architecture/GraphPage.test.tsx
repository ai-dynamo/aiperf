/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GraphPage } from "./GraphPage.js";

describe("GraphPage", () => {
  it("renders the intro and graph nodes", () => {
    render(
      <ReactFlowProvider>
        <GraphPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Trace replay path")).toBeInTheDocument();
    expect(screen.getByText("GraphInputAdapterResolver")).toBeInTheDocument();
    expect(screen.getByText("TStarSampler")).toBeInTheDocument();
    expect(screen.getByText("graph executor")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <GraphPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("One compiler")).toBeInTheDocument();
    expect(screen.getByText("Warmup failure")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/graph/executor.rs")).toBeInTheDocument();
  });
});
