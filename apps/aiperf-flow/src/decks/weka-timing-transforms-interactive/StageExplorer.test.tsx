/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { StageExplorer, STAGES } from "./StageExplorer.js";

describe("StageExplorer", () => {
  it("renders all nine stage names as pipeline nodes", () => {
    render(
      <ReactFlowProvider>
        <StageExplorer />
      </ReactFlowProvider>,
    );
    expect(STAGES).toHaveLength(9);
    for (const stage of STAGES) {
      // The default-selected stage's name is echoed twice (pipeline node + detail-card
      // heading), every other stage's name appears once as its pipeline node.
      expect(screen.getAllByText(stage.name).length).toBeGreaterThanOrEqual(1);
    }
  });

  it("defaults the detail card to the idle-gap warp stage", () => {
    render(
      <ReactFlowProvider>
        <StageExplorer />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("_ActiveIdleWarp.map")).toBeInTheDocument();
    expect(screen.getByText(/Collapses true dead air/)).toBeInTheDocument();
  });

  it("clicking a stage node updates the detail card", () => {
    render(
      <ReactFlowProvider>
        <StageExplorer />
      </ReactFlowProvider>,
    );
    fireEvent.click(screen.getByText("Flatten requests"));
    expect(screen.getByText("_flatten_requests")).toBeInTheDocument();
    expect(screen.getByText(/DFS over requests/)).toBeInTheDocument();
  });
});
