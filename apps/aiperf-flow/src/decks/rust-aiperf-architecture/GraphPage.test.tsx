/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GraphPage } from "./GraphPage.js";

describe("GraphPage", () => {
  it("renders the intro and the hub", () => {
    render(<GraphPage />);
    expect(screen.getByText("Trace replay path")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How are traces replayed?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<GraphPage />);
    expect(screen.getAllByText("Resolve source").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Execute graph").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<GraphPage />);
    expect(screen.getByText("rust/aiperf/src/graph/executor.rs")).toBeInTheDocument();
  });
});
