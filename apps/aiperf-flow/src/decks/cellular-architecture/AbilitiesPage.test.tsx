/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AbilitiesPage } from "./AbilitiesPage.js";

describe("AbilitiesPage", () => {
  it("renders the ability matrix with verbatim dimensions and statuses", () => {
    render(<AbilitiesPage />);
    expect(screen.getByText("Ability map")).toBeInTheDocument();
    expect(screen.getByText("Online transport")).toBeInTheDocument();
    expect(screen.getByText("DynoSim transport")).toBeInTheDocument();
    expect(screen.getAllByText("Rejected").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Approximation").length).toBeGreaterThan(0);
    expect(
      screen.getByText("t-digest + exact moments"),
    ).toBeInTheDocument();
  });

  it("hides Planned/Partial rows when the roadmap toggle is turned off", () => {
    render(<AbilitiesPage />);
    // "Graph adaptive" is a Planned dimension, visible by default.
    expect(screen.getByText("Graph adaptive")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("switch"));
    expect(screen.queryByText("Graph adaptive")).not.toBeInTheDocument();
    // A Built dimension stays visible.
    expect(screen.getByText("Work unit")).toBeInTheDocument();
  });
});
