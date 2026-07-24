/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FeaturesPage } from "./FeaturesPage.js";

describe("FeaturesPage", () => {
  it("renders the intro and the hub", () => {
    render(<FeaturesPage />);
    expect(screen.getByText("Feature composition")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("What can this image do?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<FeaturesPage />);
    expect(screen.getAllByText("Lean base").length).toBeGreaterThan(0);
    expect(screen.getAllByText("velo").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<FeaturesPage />);
    expect(screen.getByText("rust/cli/Cargo.toml")).toBeInTheDocument();
  });
});
