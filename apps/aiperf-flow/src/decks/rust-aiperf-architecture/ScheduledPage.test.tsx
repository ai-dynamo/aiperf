/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScheduledPage } from "./ScheduledPage.js";

describe("ScheduledPage", () => {
  it("renders the intro and the hub", () => {
    render(<ScheduledPage />);
    expect(screen.getByText("Paced workload path")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How is load paced?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<ScheduledPage />);
    expect(screen.getAllByText("Lower dataset").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Worker topology").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<ScheduledPage />);
    expect(screen.getByText("rust/aiperf/src/phase_runtime.rs")).toBeInTheDocument();
  });
});
