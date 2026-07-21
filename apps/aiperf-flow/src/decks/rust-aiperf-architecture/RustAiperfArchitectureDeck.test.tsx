/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RustAiperfArchitectureDeck } from "./RustAiperfArchitectureDeck.js";

const TAB_LABELS = [
  "1 · System",
  "2 · Processes",
  "3 · Runtime",
  "4 · Protocol",
  "5 · Scheduled",
  "6 · Graph",
  "7 · Endpoints",
  "8 · Metrics",
  "9 · Cellular",
  "10 · Builds",
  "11 · Seams",
];

describe("RustAiperfArchitectureDeck", () => {
  it("renders the deck header and all eleven tab labels", () => {
    render(<RustAiperfArchitectureDeck />);
    expect(screen.getByText("Rust AIPerf architecture")).toBeInTheDocument();
    for (const label of TAB_LABELS) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("shows the System page by default", () => {
    render(<RustAiperfArchitectureDeck />);
    expect(screen.getByText("One binary, two roles")).toBeInTheDocument();
  });

  it("switches to the Scheduled page when its tab is clicked", () => {
    render(<RustAiperfArchitectureDeck />);
    fireEvent.click(screen.getByRole("button", { name: "5 · Scheduled" }));
    expect(screen.getByText("Paced workload path")).toBeInTheDocument();
  });

  it("switches to the Seams page when its tab is clicked", () => {
    render(<RustAiperfArchitectureDeck />);
    fireEvent.click(screen.getByRole("button", { name: "11 · Seams" }));
    expect(screen.getByText("Extension internals")).toBeInTheDocument();
  });
});
