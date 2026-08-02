/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { MockServerArchitectureDeck } from "./MockServerArchitectureDeck.js";

describe("MockServerArchitectureDeck", () => {
  it("renders the top bar, all ten chapter tabs, and the default orientation page", () => {
    render(<MockServerArchitectureDeck />);
    expect(screen.getByText("Mock Foundry — mock-server architecture")).toBeInTheDocument();
    // Chapter tab labels (chapter.short) from the source canvas.
    for (const label of [
      "Foundry map",
      "Ingress manifold",
      "Glassworks",
      "Endpoint works",
      "Switching yard",
      "Escapement",
      "Foundry floor",
      "Fault lab",
      "Telemetry deck",
      "Proof machine",
    ]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByText("One request end to end")).toBeInTheDocument();
  });

  it("switches to another chapter when its tab is clicked", () => {
    render(<MockServerArchitectureDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Escapement" }));
    expect(screen.getByText("TTFT and ITL pacing")).toBeInTheDocument();
    expect(screen.getByText("Timing and generation")).toBeInTheDocument();
  });
});
