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

describe("unified atlas route", () => {
  it("renders a textual graph inventory with no dangling connections", async () => {
    renderAtlas(
      "/atlas?audience=developer&modes=online_grpc&ownership=rust&statuses=built",
    );

    const inventory = await screen.findByRole("list", {
      name: "Visible architecture components",
    });
    const summary = screen.getByRole("status", {
      name: "Atlas graph summary",
    });
    expect(within(inventory).getAllByRole("listitem").length).toBeGreaterThan(0);
    expect(summary).toHaveTextContent(/components, \d+ connections/u);
    expect(screen.getByLabelText("Architecture graph")).toBeInTheDocument();
  });

  it("shows selected evidence and changes copy with the audience lens", async () => {
    const router = renderAtlas(
      "/atlas?audience=executive&selected=component.clock-seam",
    );

    const drawer = await screen.findByRole("dialog", {
      name: "Consistent time model",
    });
    expect(drawer).toHaveTextContent(
      "Keeps real service tests and deterministic simulations comparable",
    );
    expect(drawer).toHaveTextContent(/upstream/u);
    expect(drawer).toHaveTextContent(/downstream/u);

    await router.navigate({
      to: "/atlas",
      search: {
        audience: "maintainer",
        selected: "component.clock-seam",
      },
    });

    expect(
      await screen.findByRole("dialog", {
        name: "Clock with RealClock and SimClock",
      }),
    ).toHaveTextContent("crates/aiperf-clock/src/clock.rs");
  });

  it("writes search, layout, and selected-node interaction state to the URL", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/atlas?audience=developer");

    await user.type(
      await screen.findByRole("searchbox", { name: "Search atlas" }),
      "clock",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Layout perspective" }),
      "lifecycle",
    );
    await user.click(
      screen.getByRole("button", { name: /^Injected execution clock/u }),
    );

    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
        layout: "lifecycle",
        query: "clock",
        selected: "component.clock-seam",
      });
    });
  });
});

describe("crate reference routes", () => {
  it("renders a known crate and its complete relationships", async () => {
    renderAtlas("/crates/aiperf-clock?audience=maintainer");

    expect(
      await screen.findByRole("heading", {
        name: "aiperf-clock package",
        level: 1,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Clock")).toBeInTheDocument();
    expect(screen.getAllByText("aiperf-timing").length).toBeGreaterThan(0);
    expect(screen.getByText("Virtual advance controls remain inherent on SimClock")).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Crate directory" }),
    ).toBeInTheDocument();
  });

  it("renders a useful typed not-found view", async () => {
    renderAtlas("/crates/not-real?audience=developer");

    expect(
      await screen.findByRole("heading", { name: "Crate not found" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Browse crate directory" })).toHaveAttribute(
      "href",
      expect.stringContaining("/crates/aiperf-clock"),
    );
    expect(screen.getByRole("link", { name: "Open unified atlas" })).toBeInTheDocument();
  });
});

describe("shell search", () => {
  it("navigates exact crate names to references without stale guided filters", async () => {
    const user = userEvent.setup();
    const router = renderAtlas(
      "/execution?audience=executive&modes=online_grpc&statuses=built",
    );

    await user.type(
      await screen.findByRole("searchbox", { name: "Search architecture" }),
      "aiperf-clock",
    );
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/crates/aiperf-clock");
      expect(router.state.location.search).toEqual({ audience: "executive" });
    });
  });

  it("sends general terms to the atlas and preserves only the audience", async () => {
    const user = userEvent.setup();
    const router = renderAtlas(
      "/parity?audience=maintainer&modes=dynamo_offline&statuses=unbuilt",
    );

    await user.type(
      await screen.findByRole("searchbox", { name: "Search architecture" }),
      "virtual clock",
    );
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/atlas");
      expect(router.state.location.search).toEqual({
        audience: "maintainer",
        query: "virtual clock",
      });
    });
  });
});
