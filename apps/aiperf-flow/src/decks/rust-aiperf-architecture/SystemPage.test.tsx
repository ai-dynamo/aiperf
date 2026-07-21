/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SystemPage } from "./SystemPage.js";

describe("SystemPage", () => {
  it("renders the intro and band nodes", () => {
    render(
      <ReactFlowProvider>
        <SystemPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("One binary, two roles")).toBeInTheDocument();
    expect(screen.getAllByText("aiperf --execute").length).toBeGreaterThan(0);
    expect(screen.getByText("aiperf-mock-server")).toBeInTheDocument();
    expect(screen.getByText("Dynamo SteppableReplay")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <SystemPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Product boundary")).toBeInTheDocument();
    expect(screen.getByText("Feature gate")).toBeInTheDocument();
    expect(screen.getByText("rust/cli/src/dispatch.rs")).toBeInTheDocument();
  });
});
