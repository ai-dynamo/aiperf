/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SystemPage } from "./SystemPage.js";

describe("SystemPage", () => {
  it("renders the intro and the hub", () => {
    render(<SystemPage />);
    expect(screen.getByText("One binary, two roles")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("What runs the benchmark?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards with their mini-diagram labels", () => {
    render(<SystemPage />);
    expect(screen.getAllByText("Author and launch").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Dispatch target").length).toBeGreaterThan(0);
    expect(screen.getAllByText("--execute").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<SystemPage />);
    expect(screen.getByText("rust/cli/src/dispatch.rs")).toBeInTheDocument();
  });
});
