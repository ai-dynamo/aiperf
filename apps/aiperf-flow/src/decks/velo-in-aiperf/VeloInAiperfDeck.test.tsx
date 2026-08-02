/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { VeloInAiperfDeck } from "./VeloInAiperfDeck.js";

describe("VeloInAiperfDeck", () => {
  it("renders the constellation index by default", () => {
    render(<VeloInAiperfDeck />);
    expect(screen.getByText("Velo mechanisms")).toBeInTheDocument();
    expect(
      screen.getByText(/Ten interactive instruments expose how cellular identity/),
    ).toBeInTheDocument();
  });

  it("switches to the Radar mechanism when its tab is clicked", () => {
    render(<VeloInAiperfDeck />);
    fireEvent.click(screen.getByRole("button", { name: "R · Radar" }));
    expect(screen.getByText("Resolve a controller")).toBeInTheDocument();
  });

  it("switches to the Press mechanism when its tab is clicked", () => {
    render(<VeloInAiperfDeck />);
    fireEvent.click(screen.getByRole("button", { name: "P · Press" }));
    expect(screen.getByText("Typed state becomes raw bytes")).toBeInTheDocument();
  });

  it("switches to the Tree mechanism when its tab is clicked", () => {
    render(<VeloInAiperfDeck />);
    fireEvent.click(screen.getByRole("button", { name: "T · Tree" }));
    expect(screen.getByText("Collapse payload upward")).toBeInTheDocument();
  });
});
