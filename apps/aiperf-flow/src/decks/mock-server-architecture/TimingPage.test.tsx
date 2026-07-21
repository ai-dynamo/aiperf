/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { TimingPage } from "./TimingPage.js";

describe("TimingPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <TimingPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Timing and generation")).toBeInTheDocument();
    expect(screen.getByText("TTFT and ITL pacing")).toBeInTheDocument();
    expect(screen.getByText("Character and corpus tokenization")).toBeInTheDocument();
    expect(
      screen.getAllByText("First-token delay and generated-token gaps are independently paced.")
        .length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("rust/mock-server/src/latency.rs").length).toBeGreaterThan(0);
  });
});
