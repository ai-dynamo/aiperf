/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { XrayPage } from "./XrayPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <XrayPage />
    </ReactFlowProvider>,
  );
}

describe("XrayPage", () => {
  it("renders the envelope layers and the first trace step", () => {
    renderPage();
    expect(screen.getByText("handler / aiperf.cell.register")).toBeInTheDocument();
    expect(screen.getByText("cell_id / u32")).toBeInTheDocument();
    expect(screen.getByText("1 / decode CellRegister")).toBeInTheDocument();
    expect(screen.getByText("raw payload → CellRegister")).toBeInTheDocument();
  });

  it("advances the trace to encode RegisterReply", () => {
    renderPage();
    for (let i = 0; i < 3; i++) {
      fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    }
    expect(screen.getByText("4 / encode RegisterReply")).toBeInTheDocument();
    expect(screen.getByText("envelope bytes + EventHandle")).toBeInTheDocument();
  });

  it("shows the RegisterReply fields", () => {
    renderPage();
    expect(screen.getByText("envelope: protocol-v2 bytes")).toBeInTheDocument();
    expect(screen.getByText("start_event: EventHandle")).toBeInTheDocument();
  });

  it("toggles the envelope open and closed", () => {
    renderPage();
    const toggle = screen.getByRole("button", { name: "Dissect envelope" });
    fireEvent.click(toggle);
    expect(screen.getByRole("button", { name: "Close envelope" })).toBeInTheDocument();
  });
});
