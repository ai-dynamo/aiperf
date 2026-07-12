// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createAppRouter } from "./router";

function renderAtlas(path: string) {
  const router = createAppRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(<RouterProvider router={router} />);
  return router;
}

describe("guided architecture routes", () => {
  it("navigates between canonical scene routes from the collapsible scene rail", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/?audience=developer");
    const rail = await screen.findByRole("navigation", { name: "Runtime scenes" });

    await user.click(
      within(rail).getByRole("link", { name: "Metrics and telemetry" }),
    );

    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/scenes/metrics-telemetry");
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
      });
    });
  });

  it("supports compact scene rail collapse and restore interactions", async () => {
    const user = userEvent.setup();
    renderAtlas("/");

    await user.click(await screen.findByRole("button", { name: "Collapse scene rail" }));
    expect(
      screen.getByRole("button", { name: "Expand scene rail" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Expand scene rail" }));
    expect(
      screen.getByRole("button", { name: "Collapse scene rail" }),
    ).toBeInTheDocument();
  });

  it("writes command-bar search and flavor state without losing audience", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/?audience=executive");

    await user.type(
      await screen.findByRole("searchbox", { name: "Graph search" }),
      "clock seam",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Primary flavor" }),
      "native_grpc",
    );

    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "executive",
        primary: "native_grpc",
        q: "clock seam",
      });
    });
  });

  it("recovers invalid shared graph state with a visible non-blocking notice", async () => {
    const router = renderAtlas("/?audience=developer&s=not-a-valid-state");

    expect(
      await screen.findByRole("status", { name: "Graph state recovery notice" }),
    ).toHaveTextContent(/restored canonical scene/i);
    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
      });
    });
  });

  it.each([
    ["/journey?audience=developer", "/"],
    ["/execution?audience=developer", "/scenes/endpoint-bindings-transports"],
    ["/data-plane?audience=developer", "/scenes/dataset-segment-pipeline"],
    ["/observability?audience=developer", "/scenes/metrics-telemetry"],
    ["/parity?audience=developer", "/scenes/crate-dependency-topology"],
    ["/atlas?audience=developer", "/"],
  ])("redirects legacy guided route %s to %s", async (legacyRoute, expectedPath) => {
    const router = renderAtlas(legacyRoute);
    await screen.findAllByRole("heading", { level: 1 });
    await waitFor(() => {
      expect(router.state.location.pathname).toBe(expectedPath);
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
      });
    });
  });
});
