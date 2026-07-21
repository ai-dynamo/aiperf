/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GatePage } from "./GatePage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <GatePage />
    </ReactFlowProvider>,
  );
}

describe("GatePage", () => {
  it("starts with zero registered and START disabled", () => {
    renderPage();
    expect(screen.getByText("0 / 4 registered")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Trigger START" })).toBeDisabled();
  });

  it("enables START only once all four cells register", () => {
    renderPage();
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByRole("button", { name: `Register c${i}` }));
    }
    const start = screen.getByRole("button", { name: "Trigger START" });
    expect(start).not.toBeDisabled();
    fireEvent.click(start);
    expect(screen.getByText("all awaiters → Ready")).toBeInTheDocument();
  });

  it("resets the apparatus", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Register c0" }));
    expect(screen.getByText("1 / 4 registered")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset apparatus" }));
    expect(screen.getByText("0 / 4 registered")).toBeInTheDocument();
  });
});
