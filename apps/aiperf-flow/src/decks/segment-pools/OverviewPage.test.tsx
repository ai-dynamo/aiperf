/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { OverviewPage } from "./OverviewPage.js";

describe("OverviewPage", () => {
  it("renders the pipeline heading and framing copy", () => {
    render(
      <ReactFlowProvider>
        <OverviewPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("The pipeline: rows in → wire bytes out")).toBeInTheDocument();
    expect(screen.getByText(/serialize content once, splice bytes forever/)).toBeInTheDocument();
  });

  it("renders the BUILD-band nodes from the canvas source", () => {
    render(
      <ReactFlowProvider>
        <OverviewPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Dataset source")).toBeInTheDocument();
    expect(screen.getByText("JSON / CSV / HF / trace")).toBeInTheDocument();
    expect(screen.getByText("Composer.compose")).toBeInTheDocument();
    expect(screen.getByText("intern rows → pool")).toBeInTheDocument();
    expect(screen.getByText("apply_common_contexts")).toBeInTheDocument();
    expect(screen.getByText("system / user_context")).toBeInTheDocument();
  });

  it("renders the FREEZE-band SegmentPool and InMemorySegmentStore nodes", () => {
    render(
      <ReactFlowProvider>
        <OverviewPage />
      </ReactFlowProvider>,
    );
    expect(screen.getAllByText("SegmentPool").length).toBeGreaterThan(0);
    expect(screen.getByText("arena: Vec<Segment>")).toBeInTheDocument();
    expect(screen.getAllByText("InMemorySegmentStore").length).toBeGreaterThan(0);
    expect(screen.getByText("Box<[Segment]> (frozen)")).toBeInTheDocument();
  });

  it("renders the DISPATCH-band nodes from the canvas source", () => {
    render(
      <ReactFlowProvider>
        <OverviewPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    expect(screen.getByText("Arc<dyn SegmentStore>")).toBeInTheDocument();
    expect(screen.getByText("precompute_body_plans")).toBeInTheDocument();
    expect(screen.getByText("BodyPlan per static turn")).toBeInTheDocument();
    expect(screen.getByText("JsonBodyMaterializer")).toBeInTheDocument();
    expect(screen.getByText("splice handles → Bytes")).toBeInTheDocument();
    expect(screen.getByText("Transport")).toBeInTheDocument();
    expect(screen.getByText("HTTP / gRPC dispatch")).toBeInTheDocument();
  });

  it("renders the two-representations callout", () => {
    render(
      <ReactFlowProvider>
        <OverviewPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText(/The two representations/)).toBeInTheDocument();
  });
});
