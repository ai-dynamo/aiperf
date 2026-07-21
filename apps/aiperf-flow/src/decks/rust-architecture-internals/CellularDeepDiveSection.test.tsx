/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CellularDeepDiveSection } from "./CellularDeepDiveSection.js";

describe("CellularDeepDiveSection", () => {
  it("renders the controller, cells, and default star merge", () => {
    render(<CellularDeepDiveSection detail="engineering" />);
    expect(screen.getByText("Cellular wraps the run core with ownership and merge planes")).toBeInTheDocument();
    expect(screen.getByText("cellular controller")).toBeInTheDocument();
    expect(screen.getAllByText("ordinary run core").length).toBe(3);
    expect(screen.getByText("controller merge")).toBeInTheDocument();
    expect(screen.getByText(/Velo start event/)).toBeInTheDocument();
  });

  it("switches to the T2 fold tree with aggregators", () => {
    render(<CellularDeepDiveSection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "T2 fold tree" }));
    expect(screen.getByText("aggregator 0")).toBeInTheDocument();
    expect(screen.getByText("aggregator 1")).toBeInTheDocument();
    expect(screen.getByText("controller store merge")).toBeInTheDocument();
  });

  it("shows the Phaser start generation caption in phaser focus", () => {
    render(<CellularDeepDiveSection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "Phaser opt-in" }));
    expect(screen.getByText(/Phaser Started generation/)).toBeInTheDocument();
  });
});
