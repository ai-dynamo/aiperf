/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { OrientationPage } from "./OrientationPage.js";

describe("OrientationPage", () => {
  it("renders the chapter heading and signature walkthrough control", () => {
    render(
      <ReactFlowProvider>
        <OrientationPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Orientation")).toBeInTheDocument();
    expect(screen.getByText("Advance specimen")).toBeInTheDocument();
  });

  it("renders catalog titles and verbatim proof paths from the source canvas", () => {
    render(
      <ReactFlowProvider>
        <OrientationPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Standalone target boundary")).toBeInTheDocument();
    expect(screen.getByText("One request end to end")).toBeInTheDocument();
    expect(screen.getByText("rust/e2e/tests/test_chat_endpoint.rs")).toBeInTheDocument();
    expect(
      screen.getAllByText(
        "A request crosses parsing, token budgeting, latency, streaming, and accounting in one server process.",
      ).length,
    ).toBeGreaterThan(0);
  });
});
