/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EndpointsPage } from "./EndpointsPage.js";

describe("EndpointsPage", () => {
  it("renders the intro and the hub", () => {
    render(<EndpointsPage />);
    expect(screen.getByText("Dialect preparation")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How are dialects prepared?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<EndpointsPage />);
    expect(screen.getAllByText("Resolve profiles").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Dialect families").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<EndpointsPage />);
    expect(screen.getByText("rust/aiperf/src/endpoints/endpoints.rs")).toBeInTheDocument();
  });
});
