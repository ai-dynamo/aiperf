/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { RadarPage } from "./RadarPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <RadarPage />
    </ReactFlowProvider>,
  );
}

describe("RadarPage", () => {
  it("starts with zero sectors resolved", () => {
    renderPage();
    expect(screen.getByText("0/4 sectors resolved · each sweep is user-triggered")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sweep sector" })).toBeInTheDocument();
  });

  it("resolves an endpoint after sweeping", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Sweep sector" }));
    expect(screen.getByText("1/4 sectors resolved · each sweep is user-triggered")).toBeInTheDocument();
    expect(screen.getByText("tcp://host:port")).toBeInTheDocument();
  });

  it("resolves all four sectors and disables further sweeping", () => {
    renderPage();
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByRole("button", { name: /Sweep sector|All resolved/ }));
    }
    expect(screen.getByRole("button", { name: "All resolved" })).toBeDisabled();
    expect(screen.getByText("ephemeral")).toBeInTheDocument();
  });
});
