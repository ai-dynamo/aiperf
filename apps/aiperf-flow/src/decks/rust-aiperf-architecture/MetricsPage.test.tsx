/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricsPage } from "./MetricsPage.js";

describe("MetricsPage", () => {
  it("renders the intro and the hub", () => {
    render(<MetricsPage />);
    expect(screen.getByText("Measurement and exports")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How is each request measured?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards with their titles", () => {
    render(<MetricsPage />);
    expect(screen.getAllByText("MetricsAccumulator").length).toBeGreaterThan(0);
    expect(screen.getAllByText("NativeReport").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<MetricsPage />);
    expect(screen.getByText("rust/aiperf/src/report.rs")).toBeInTheDocument();
  });
});
