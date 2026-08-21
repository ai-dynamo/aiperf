/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it, vi } from "vitest";
import { IndexPage, MECHANISMS } from "./IndexPage.js";

describe("IndexPage", () => {
  it("renders the eyebrow, title, and constellation core", () => {
    render(
      <ReactFlowProvider>
        <IndexPage onSelect={vi.fn()} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("AIPerf cellular transport")).toBeInTheDocument();
    expect(screen.getByText("Velo mechanisms")).toBeInTheDocument();
    expect(screen.getByText("Velo plane")).toBeInTheDocument();
  });

  it("renders all ten mechanism cards with titles and marks", () => {
    render(
      <ReactFlowProvider>
        <IndexPage onSelect={vi.fn()} />
      </ReactFlowProvider>,
    );
    expect(MECHANISMS).toHaveLength(10);
    expect(screen.getByText("Connection radar")).toBeInTheDocument();
    expect(screen.getByText("Hierarchy refusal")).toBeInTheDocument();
    expect(screen.getByText("R / 01")).toBeInTheDocument();
    expect(screen.getByText("T / 10")).toBeInTheDocument();
  });
});
