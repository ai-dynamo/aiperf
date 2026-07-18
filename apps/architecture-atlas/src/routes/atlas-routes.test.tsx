// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { decompressFromEncodedURIComponent } from "lz-string";

import {
  canonicalGraphState,
  encodeGraphStateForUrl,
} from "../domain/graph-state";
import { createAppRouter } from "./router";

function renderAtlas(path: string) {
  const router = createAppRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(<RouterProvider router={router} />);
  return router;
}

describe("graph-first route shell", () => {
  it("surfaces all canonical scenes in compact rail navigation", async () => {
    renderAtlas("/?audience=developer");

    const rail = await screen.findByRole("navigation", { name: "Runtime scenes" });
    expect(
      within(rail).getByRole("link", { name: "Runtime composition" }),
    ).toBeInTheDocument();
    expect(
      within(rail).getByRole("link", {
        name: "Runner protocol and registries",
      }),
    ).toBeInTheDocument();
    expect(
      within(rail).getByRole("link", { name: "Crate dependency topology" }),
    ).toBeInTheDocument();
  });

  it("uses the only visible graph search input as the drawer focus fallback", async () => {
    const user = userEvent.setup();
    const state = encodeGraphStateForUrl(
      canonicalGraphState({
        audience: "developer",
        focusedEntityId: "node.clock-seam",
        primaryFlavor: "native_http",
        sceneId: "scene.runtime-composition",
      }),
    );
    renderAtlas(`/?audience=developer&s=${state}`);

    const graphSearch = await screen.findByRole("searchbox", {
      name: "Graph search",
    });
    expect(
      screen.queryByRole("textbox", { name: "Graph search focus target" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByRole("searchbox")).toEqual([graphSearch]);

    const close = await screen.findByRole("button", {
      name: "Close evidence panel",
    });
    const originalClosest = HTMLElement.prototype.closest;
    const closest = vi
      .spyOn(HTMLElement.prototype, "closest")
      .mockImplementation(function (this: HTMLElement, selector) {
        if (
          selector === "details:not([open])" &&
          this.dataset.graphEntityId === "node.clock-seam"
        ) {
          return document.createElement("details");
        }
        return originalClosest.call(this, selector);
      });
    try {
      await user.click(close);
      await waitFor(() => {
        expect(graphSearch).toHaveFocus();
      });
    } finally {
      closest.mockRestore();
    }
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

  it("drills related crate components into focused graph-first state", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/crates/aiperf-clock?audience=maintainer");

    const relatedComponent = await screen.findByRole("link", {
      name: "Clock with RealClock and SimClock",
    });
    const target = new URL(relatedComponent.getAttribute("href")!, window.location.origin);
    expect(
      JSON.parse(decompressFromEncodedURIComponent(target.searchParams.get("s")!) ?? "{}"),
    ).toMatchObject({ focusedEntityId: "node.clock-seam" });
    await user.click(relatedComponent);

    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/");
      expect(router.state.location.search).toMatchObject({
        audience: "maintainer",
        primary: "native_http",
        q: "Clock with RealClock and SimClock",
      });
      expect(router.state.location.search.s).toEqual(expect.any(String));
      expect(
        JSON.parse(
          decompressFromEncodedURIComponent(
            String(router.state.location.search.s),
          ) ?? "{}",
        ),
      ).toMatchObject({ focusedEntityId: "node.clock-seam" });
    });
    expect(
      await screen.findByTestId("graph-node-node.clock-seam"),
    ).toHaveAttribute("data-path-state", "focused");
    expect(
      await screen.findByRole("heading", {
        name: "Clock with RealClock and SimClock",
      }),
    ).toBeInTheDocument();
  });

  it("labels normal and development Cargo dependencies separately", async () => {
    renderAtlas("/crates/aiperf-extensions?audience=maintainer");

    expect(
      await screen.findByRole("heading", { name: "Normal dependencies" }),
    ).toBeInTheDocument();
    const development = screen.getByRole("region", {
      name: "Development dependencies",
    });
    expect(within(development).getByText("aiperf-rng")).toBeInTheDocument();
    expect(
      within(
        screen.getByRole("region", { name: "Normal dependencies" }),
      ).queryByText("aiperf-rng"),
    ).not.toBeInTheDocument();
  });

  it("preserves Cargo kind when displaying reverse dependents", async () => {
    renderAtlas("/crates/aiperf-rng?audience=maintainer");

    const normal = await screen.findByRole("region", {
      name: "Normal dependents",
    });
    const development = screen.getByRole("region", {
      name: "Development dependents",
    });
    expect(within(normal).getByText("aiperf-mock-rs")).toBeInTheDocument();
    expect(
      within(development).getByText("aiperf-extensions"),
    ).toBeInTheDocument();
    expect(within(normal).queryByText("aiperf-extensions")).not.toBeInTheDocument();
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
  it("navigates exact crate names to references from graph-first routes", async () => {
    const user = userEvent.setup();
    const router = renderAtlas(
      "/scenes/endpoint-bindings-transports?audience=executive",
    );

    await user.type(
      await screen.findByRole("searchbox", { name: "Graph search" }),
      "aiperf-clock",
    );
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/crates/aiperf-clock");
      expect(router.state.location.search).toEqual({ audience: "executive" });
    });
  });

  it("writes general graph terms to shared search state on the active scene", async () => {
    const user = userEvent.setup();
    const router = renderAtlas(
      "/scenes/metrics-telemetry?audience=maintainer",
    );

    await user.type(
      await screen.findByRole("searchbox", { name: "Graph search" }),
      "virtual clock",
    );

    await waitFor(() => {
      expect(router.state.location.pathname).toBe("/scenes/metrics-telemetry");
      expect(router.state.location.search).toMatchObject({
        audience: "maintainer",
        q: "virtual clock",
      });
    });
  });
});
