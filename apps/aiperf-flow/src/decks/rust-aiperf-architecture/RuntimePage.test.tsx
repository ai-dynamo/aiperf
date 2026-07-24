/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RuntimePage } from "./RuntimePage.js";

describe("RuntimePage", () => {
  it("renders the intro and the hub", () => {
    render(<RuntimePage />);
    expect(screen.getByText("One request, end to end")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("What happens per run?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<RuntimePage />);
    expect(screen.getAllByText("Author and bootstrap").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Dispatch and observe").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<RuntimePage />);
    expect(screen.getByText("rust/loadgen-core/src/sink.rs")).toBeInTheDocument();
  });
});
