// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
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
  it("frames ownership for an executive without implementation jargon", async () => {
    renderAtlas("/?audience=executive");

    expect(
      await screen.findByRole("heading", { name: "Who owns what", level: 1 }),
    ).toBeInTheDocument();
    expect(screen.getByText("Product control room")).toBeInTheDocument();
    expect(screen.queryByText("RunnerApplication")).not.toBeInTheDocument();
    expect(
      screen.getByRole("list", { name: "Product handoff sequence" }),
    ).toBeInTheDocument();
  });

  it("changes execution labels, density, and evidence for maintainers", async () => {
    renderAtlas("/execution?audience=maintainer");

    expect(
      await screen.findByRole("heading", {
        name: "Clock, scheduling, transport, and placement matrix",
        level: 1,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Clock with RealClock and SimClock"),
    ).toBeInTheDocument();
    expect(screen.getByText("RequestSink<HttpRequest>")).toBeInTheDocument();
    expect(
      screen.getAllByRole("link", { name: /crates\/aiperf-clock/u }).length,
    ).toBeGreaterThan(0);
  });

  it("preserves mode and status filters in URL state and announces counts", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/parity?audience=developer");

    const results = await screen.findByRole("status", {
      name: "Filtered result count",
    });
    const initialCount = Number.parseInt(results.textContent ?? "", 10);

    await user.click(
      screen.getByRole("checkbox", { name: "Native gRPC" }),
    );
    await user.click(screen.getByRole("checkbox", { name: "Unbuilt" }));

    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
        modes: "online_grpc",
        statuses: "unbuilt",
      });
    });
    expect(Number.parseInt(results.textContent ?? "", 10)).toBeLessThan(
      initialCount,
    );
    expect(
      screen.getByRole("heading", { name: "gRPC lifecycle limits" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Offline mode constraints" }),
    ).not.toBeInTheDocument();
  });

  it("distinguishes every parity status in the legend", async () => {
    renderAtlas("/parity?audience=developer");

    const legend = await screen.findByRole("list", {
      name: "Architecture status legend",
    });
    for (const label of [
      "Built",
      "Feature-gated",
      "Runtime-conditional",
      "Compatibility-only",
      "Legacy-parallel",
      "Unbuilt",
    ]) {
      expect(within(legend).getByText(label)).toBeInTheDocument();
    }
  });

  it("enters, navigates, and exits presentation mode from the keyboard", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/journey?audience=executive");

    await user.click(
      await screen.findByRole("button", { name: "Present this view" }),
    );
    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "executive",
        present: true,
      });
    });
    expect(
      screen.queryByRole("navigation", { name: "Architecture views" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Presentation routes" }),
    ).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/execution");
      expect(router.state.location.search).toMatchObject({
        audience: "executive",
        present: true,
      });
    });

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/execution");
      expect(router.state.location.search.present).toBeUndefined();
    });
    expect(
      screen.getByRole("navigation", { name: "Architecture views" }),
    ).toBeInTheDocument();
  });

  it("uses semantic landmarks and non-trapping controls on mobile-ready views", async () => {
    renderAtlas("/data-plane?audience=developer");

    expect(await screen.findByRole("main")).toHaveAttribute(
      "id",
      "atlas-content",
    );
    expect(
      screen.getByRole("region", { name: "Dataset and endpoint data plane" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("list", { name: "Request shaping flow" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Present this view" }),
    ).toHaveAttribute("type", "button");
  });
});
