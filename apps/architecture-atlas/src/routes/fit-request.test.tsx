// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { FlavorOverlay } from "../domain/graph-derivation";
import type { GraphFitViewCommand } from "../features/graph/types";

interface FitAwareCanvasProps {
  fitViewCommand?: GraphFitViewCommand;
  overlay?: FlavorOverlay;
}

vi.mock("../features/graph/graph-canvas", () => ({
  GraphCanvas: ({ fitViewCommand, overlay }: FitAwareCanvasProps) => (
    <>
      <output aria-label="Observed graph fit request">
        {fitViewCommand?.requestId ?? "none"}
      </output>
      <output aria-label="Observed graph flavor overlay">
        {JSON.stringify(overlay ?? null)}
      </output>
    </>
  ),
}));

import { createAppRouter } from "./router";

describe("graph fit command integration", () => {
  it("passes each command-bar fit request through GraphScene to GraphCanvas", async () => {
    const user = userEvent.setup();
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: ["/?audience=developer"],
      }),
    });
    render(<RouterProvider router={router} />);

    expect(
      await screen.findByRole("status", {
        name: "Observed graph fit request",
      }),
    ).toHaveTextContent("none");

    await user.click(screen.getByRole("button", { name: "Fit graph" }));
    expect(
      screen.getByRole("status", { name: "Observed graph fit request" }),
    ).toHaveTextContent("1");

    await user.click(screen.getByRole("button", { name: "Fit graph" }));
    expect(
      screen.getByRole("status", { name: "Observed graph fit request" }),
    ).toHaveTextContent("2");
  });

  it("passes the derived comparison overlay through GraphScene", async () => {
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: [
          "/?audience=developer&primary=native_http&compare=dynamo_offline",
        ],
      }),
    });
    render(<RouterProvider router={router} />);

    expect(
      await screen.findByRole("status", {
        name: "Observed graph flavor overlay",
      }),
    ).toHaveTextContent('"sharedNodeIds"');
    expect(
      screen.getByRole("status", {
        name: "Observed graph flavor overlay",
      }),
    ).toHaveTextContent("node.runtime-composition");
  });
});
