/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProcessesPage } from "./ProcessesPage.js";

describe("ProcessesPage", () => {
  it("renders the intro and the hub", () => {
    render(<ProcessesPage />);
    expect(screen.getByText("Crates and boundaries")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How is the workspace wired?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<ProcessesPage />);
    expect(screen.getAllByText("loadgen-core").length).toBeGreaterThan(0);
    expect(screen.getAllByText("External boundaries").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<ProcessesPage />);
    expect(screen.getByText("rust/aiperf/src/lib.rs")).toBeInTheDocument();
  });
});
