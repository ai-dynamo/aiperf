/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { RuntimePage } from "./RuntimePage.js";

describe("RuntimePage", () => {
  it("renders the intro and hot-path nodes", () => {
    render(
      <ReactFlowProvider>
        <RuntimePage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("One request, end to end")).toBeInTheDocument();
    expect(screen.getByText("RunnerApplication::stock")).toBeInTheDocument();
    expect(screen.getByText("RequestSink<R>::dispatch")).toBeInTheDocument();
    expect(screen.getByText("Metrics accumulator")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <RuntimePage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Startup vs hot path")).toBeInTheDocument();
    expect(screen.getByText("Lock avoidance")).toBeInTheDocument();
    expect(screen.getByText("rust/loadgen-core/src/sink.rs")).toBeInTheDocument();
  });
});
