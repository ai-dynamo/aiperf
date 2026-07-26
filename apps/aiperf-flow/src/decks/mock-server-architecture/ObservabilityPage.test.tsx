/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ObservabilityPage } from "./ObservabilityPage.js";

describe("ObservabilityPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <ObservabilityPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Observability and deployment")).toBeInTheDocument();
    expect(screen.getByText("Prometheus backend dialects")).toBeInTheDocument();
    expect(screen.getByText("Multi-process L4 balancer")).toBeInTheDocument();
    expect(
      screen.getAllByText(
        "Synthetic GPU load follows observed request throughput within the configured window.",
      ).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("rust/e2e-tests/tests/test_server_metrics.rs")).toBeInTheDocument();
  });
});
