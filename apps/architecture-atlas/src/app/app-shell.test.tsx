// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

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
  it("renders primary navigation and the active route", async () => {
    renderAtlas("/execution?audience=developer");

    expect(
      await screen.findByRole("heading", { name: "Execution modes" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Architecture views" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Presentation controls" }),
    ).toBeDisabled();
  });

  it("uses the URL audience before local storage", async () => {
    window.localStorage.setItem(AUDIENCE_STORAGE_KEY, "maintainer");
    renderAtlas("/journey?audience=executive");

    expect(
      await screen.findByRole("combobox", { name: "Audience" }),
    ).toHaveValue("executive");
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
      await screen.findByRole("combobox", { name: "Audience" }),
    ).toHaveValue("maintainer");
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
      await screen.findByRole("combobox", { name: "Audience" }),
    ).toHaveValue("developer");
    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "developer",
      });
    });
  });

  it("persists audience changes in search state and local storage", async () => {
    const user = userEvent.setup();
    const router = renderAtlas("/atlas?audience=developer");

    await user.selectOptions(
      await screen.findByRole("combobox", { name: "Audience" }),
      "maintainer",
    );

    await waitFor(() => {
      expect(router.state.location.search).toMatchObject({
        audience: "maintainer",
      });
      expect(window.localStorage.getItem(AUDIENCE_STORAGE_KEY)).toBe(
        "maintainer",
      );
    });
  });
});
