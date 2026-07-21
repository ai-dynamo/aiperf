/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SlurmDiagram, DIAGRAMS } from "./SlurmDiagram.js";
import { STEPS } from "./steps-data.js";

describe("SlurmDiagram", () => {
  it("has one diagram per walkthrough step", () => {
    expect(DIAGRAMS.length).toBe(STEPS.length);
    for (const diagram of DIAGRAMS) {
      expect(diagram.nodes.length).toBeGreaterThan(0);
    }
  });

  it("renders the fan-out scene's cell-slice nodes (step 13)", () => {
    render(
      <ReactFlowProvider>
        <SlurmDiagram stepIndex={13} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Cell 0 · slice 0")).toBeInTheDocument();
    expect(screen.getByText("Cell 1 · slice 1")).toBeInTheDocument();
    expect(screen.getByText("Cell 2 · slice 2")).toBeInTheDocument();
    expect(screen.getByText("GLOBAL PLAN")).toBeInTheDocument();
  });

  it("renders the three traffic-plane lanes (step 9)", () => {
    render(
      <ReactFlowProvider>
        <SlurmDiagram stepIndex={9} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("VELO — control")).toBeInTheDocument();
    expect(screen.getByText("HTTP / gRPC — load")).toBeInTheDocument();
    expect(screen.getByText("HTTP/1 + zstd — bulk files")).toBeInTheDocument();
  });
});
