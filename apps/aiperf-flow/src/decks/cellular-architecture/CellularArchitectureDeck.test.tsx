/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CellularArchitectureDeck } from "./CellularArchitectureDeck.js";

describe("CellularArchitectureDeck", () => {
  it("opens on the Story tab", () => {
    render(<CellularArchitectureDeck />);
    expect(screen.getByText("Cellular Architecture")).toBeInTheDocument();
    expect(screen.getByText("One run. Many cells. One report.")).toBeInTheDocument();
  });

  it("switches to the recipe atlas tab", () => {
    render(<CellularArchitectureDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Recipe atlas" }));
    expect(
      screen.getByText("One benchmark. Many autonomous cells. One measurement contract."),
    ).toBeInTheDocument();
  });

  it("switches to the abilities tab", () => {
    render(<CellularArchitectureDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Abilities" }));
    expect(screen.getByText("Ability map")).toBeInTheDocument();
  });
});
