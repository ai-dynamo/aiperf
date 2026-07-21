/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { EndpointsPage } from "./EndpointsPage.js";

describe("EndpointsPage", () => {
  it("renders the intro and dialect nodes", () => {
    render(
      <ReactFlowProvider>
        <EndpointsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Dialect preparation")).toBeInTheDocument();
    expect(screen.getByText("EndpointRegistry")).toBeInTheDocument();
    expect(screen.getByText("PreparedEndpointTable")).toBeInTheDocument();
    expect(screen.getByText("OpenAI + Anthropic")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <EndpointsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Open registry")).toBeInTheDocument();
    expect(screen.getByText("Usage authority")).toBeInTheDocument();
    expect(screen.getByText("rust/aiperf/src/endpoints/endpoints.rs")).toBeInTheDocument();
  });
});
