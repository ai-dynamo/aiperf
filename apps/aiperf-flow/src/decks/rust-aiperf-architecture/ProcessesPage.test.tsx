/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ProcessesPage } from "./ProcessesPage.js";

describe("ProcessesPage", () => {
  it("renders the intro and library nodes", () => {
    render(
      <ReactFlowProvider>
        <ProcessesPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Crates and boundaries")).toBeInTheDocument();
    expect(screen.getAllByText("loadgen-core").length).toBeGreaterThan(0);
    expect(screen.getByText("e2e harness")).toBeInTheDocument();
  });

  it("renders the dependency-direction and packaging notes", () => {
    render(
      <ReactFlowProvider>
        <ProcessesPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Dependency direction")).toBeInTheDocument();
    expect(screen.getByText("Packaging")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/lib.rs")).toBeInTheDocument();
  });
});
