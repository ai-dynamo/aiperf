/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { CourierPage } from "./CourierPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <CourierPage />
    </ReactFlowProvider>,
  );
}

describe("CourierPage", () => {
  it("renders the three route stops at origin", () => {
    renderPage();
    expect(screen.getByText("fresh ship Velo")).toBeInTheDocument();
    expect(screen.getByText("controller handler")).toBeInTheDocument();
    expect(screen.getByText("await payload")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Deliver" })).toBeDisabled();
  });

  it("delivers the packet and registers the shipper before ACKing", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Send toward controller" }));
    fireEvent.click(screen.getByRole("button", { name: "Deliver" }));
    expect(screen.getByText("register_peer(shipper)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Return CellAck" }));
    expect(screen.getByRole("button", { name: "ACK returned" })).toBeInTheDocument();
  });

  it("counts retry attempts", () => {
    renderPage();
    expect(screen.getByRole("button", { name: "Retry · attempt 1" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry · attempt 1" }));
    expect(screen.getByRole("button", { name: "Retry · attempt 2" })).toBeInTheDocument();
  });
});
