/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { FeaturesPage } from "./FeaturesPage.js";

describe("FeaturesPage", () => {
  it("renders the intro and feature nodes", () => {
    render(
      <ReactFlowProvider>
        <FeaturesPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Feature composition")).toBeInTheDocument();
    expect(screen.getByText("aiperf-cli default = []")).toBeInTheDocument();
    expect(screen.getByText("dynamo-full")).toBeInTheDocument();
    expect(screen.getByText("cell count policy")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <FeaturesPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Fail closed")).toBeInTheDocument();
    expect(screen.getByText("DynoSim dependency")).toBeInTheDocument();
    expect(screen.getByText("Makefile")).toBeInTheDocument();
  });
});
