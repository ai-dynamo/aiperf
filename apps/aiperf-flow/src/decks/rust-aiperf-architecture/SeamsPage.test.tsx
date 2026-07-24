/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SeamsPage } from "./SeamsPage.js";

describe("SeamsPage", () => {
  it("renders the intro and the hub", () => {
    render(<SeamsPage />);
    expect(screen.getByText("Extension internals")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("Where does it stay open?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<SeamsPage />);
    expect(screen.getAllByText("Extension registration").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Cellular scaling").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<SeamsPage />);
    expect(screen.getByText("rust/aiperf/src/extensions/mod.rs")).toBeInTheDocument();
  });
});
