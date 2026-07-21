/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SemanticsPage } from "./SemanticsPage.js";

describe("SemanticsPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <SemanticsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Faults and semantics")).toBeInTheDocument();
    expect(screen.getByText("Mid-stream SSE failure")).toBeInTheDocument();
    expect(screen.getByText("Tool-call emission")).toBeInTheDocument();
    expect(
      screen.getAllByText("A stream can fail after generated output, preserving partial-response evidence.")
        .length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("rust/e2e/tests/test_error_fidelity.rs").length).toBeGreaterThan(0);
  });
});
