/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { PhaserPage } from "./PhaserPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <PhaserPage />
    </ReactFlowProvider>,
  );
}

describe("PhaserPage", () => {
  it("starts at generation 2, not yet attached", () => {
    renderPage();
    expect(screen.getByText("generation 2")).toBeInTheDocument();
    expect(screen.getByText("not attached")).toBeInTheDocument();
  });

  it("advances the generation counter", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Advance" }));
    expect(screen.getByText("generation 3")).toBeInTheDocument();
  });

  it("classifies generations as replay before and live push after the attach boundary", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Advance" })); // generation 3
    fireEvent.click(screen.getByRole("button", { name: "Attach subscriber now" }));
    fireEvent.click(screen.getByRole("button", { name: "Advance" })); // generation 4
    expect(screen.getByText("attached @ generation 3")).toBeInTheDocument();
    expect(screen.getAllByText("reply replay").length).toBeGreaterThan(0);
    expect(screen.getAllByText("live push").length).toBeGreaterThan(0);
  });
});
