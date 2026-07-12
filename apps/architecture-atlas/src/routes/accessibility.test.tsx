// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it } from "vitest";

import { createAppRouter } from "./router";

async function renderAndAudit(path: string) {
  const router = createAppRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  const rendered = render(<RouterProvider router={router} />);
  await screen.findByRole("heading", { level: 1 });
  const results = await axe(rendered.container);
  expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual(
    [],
  );
}

describe("automated accessibility", () => {
  it.each([
    ["/?audience=developer", "ownership"],
    ["/journey?audience=developer", "journey"],
    ["/execution?audience=developer", "execution"],
    ["/data-plane?audience=developer", "data plane"],
    ["/observability?audience=developer", "observability"],
    ["/parity?audience=developer", "parity"],
    ["/?audience=executive", "executive lens"],
    ["/execution?audience=maintainer", "maintainer lens"],
  ])("has no detectable violations on the %s %s route", async (path) => {
    await renderAndAudit(path);
  });

  it("audits the atlas controls, inventory, and evidence drawer", async () => {
    await renderAndAudit(
      "/atlas?audience=maintainer&selected=component.clock-seam",
    );
    expect(
      screen.getByRole("status", { name: "Atlas graph summary" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Clear selected component" }),
    ).toHaveFocus();
  });

  it.each([
    "/crates/aiperf-clock?audience=maintainer",
    "/crates/not-real?audience=developer",
  ])("has no detectable violations on crate route %s", async (path) => {
    await renderAndAudit(path);
  });
});
