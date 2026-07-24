/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CellularPage } from "./CellularPage.js";

describe("CellularPage", () => {
  it("renders the intro and the hub", () => {
    render(<CellularPage />);
    expect(screen.getByText("Multi-process scale")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How does one run scale out?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards with their titles", () => {
    render(<CellularPage />);
    expect(screen.getAllByText("cell launcher").length).toBeGreaterThan(0);
    expect(screen.getAllByText("controller merge").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<CellularPage />);
    expect(screen.getByText("rust/aiperf/src/cellular/mod.rs")).toBeInTheDocument();
  });
});
