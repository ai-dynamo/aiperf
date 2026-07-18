// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AUDIENCE_STORAGE_KEY } from "../domain/audience";
import { createAppRouter } from "../routes/router";

function renderAtlas(path = "/") {
  const router = createAppRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(<RouterProvider router={router} />);
  return router;
}

describe("application shell", () => {
  it("defaults to the runtime graph route with the scene rail visible", async () => {
    renderAtlas("/");

    expect(
      await screen.findByRole("heading", {
        name: "Runtime composition",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Runtime scenes" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Collapse scene rail" })).toBeEnabled();
  });

  it("exposes the audience lenses as labeled icon buttons", async () => {
    renderAtlas("/");

    const group = await screen.findByRole("radiogroup", { name: "Audience" });
    expect(
      within(group)
        .getAllByRole("radio")
        .map((option) => option.getAttribute("aria-label")),
    ).toEqual(["Executive", "Developer", "Maintainer"]);
  });

  it("uses the URL audience before local storage", async () => {
    window.localStorage.setItem(AUDIENCE_STORAGE_KEY, "maintainer");
    renderAtlas("/journey?audience=executive");

    expect(
      await screen.findByRole("radio", { name: "Executive" }),
    ).toBeChecked();
    await waitFor(() => {
      expect(window.localStorage.getItem(AUDIENCE_STORAGE_KEY)).toBe(
        "executive",
      );
    });
  });

  it("hydrates a valid persisted audience into the URL", async () => {
    window.localStorage.setItem(AUDIENCE_STORAGE_KEY, "maintainer");
    const router = renderAtlas("/");

    expect(
      await screen.findByRole("radio", { name: "Maintainer" }),
    ).toBeChecked();
    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "maintainer",
      });
    });
  });

  it("recovers from invalid persisted state with the safe default", async () => {
    window.localStorage.setItem(AUDIENCE_STORAGE_KEY, "operator");
    const router = renderAtlas("/");

    expect(
      await screen.findByRole("radio", { name: "Developer" }),
    ).toBeChecked();
    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
      });
    });
  });

  it("uses the safe default when the localStorage getter is unavailable", async () => {
    const storageGetter = vi
      .spyOn(window, "localStorage", "get")
      .mockImplementation(() => {
        throw new DOMException("denied", "SecurityError");
      });

    try {
      renderAtlas("/");

      expect(
        await screen.findByRole("radio", { name: "Developer" }),
      ).toBeChecked();
    } finally {
      storageGetter.mockRestore();
    }
  });

  it("persists audience changes in search state and local storage", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/?audience=developer");

    await user.click(await screen.findByRole("radio", { name: "Maintainer" }));

    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "maintainer",
      });
      expect(window.localStorage.getItem(AUDIENCE_STORAGE_KEY)).toBe(
        "maintainer",
      );
    });
  });

  it("wires primary and comparison flavor selectors into shared URL state", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/?audience=developer");

    await user.selectOptions(
      await screen.findByRole("combobox", { name: "Primary flavor" }),
      "native_grpc",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Compare flavor" }),
      "dynamo_online",
    );

    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        primary: "native_grpc",
        compare: "dynamo_online",
      });
    });
  });
});
