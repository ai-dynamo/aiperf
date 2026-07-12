// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const guidedRoutes = [
  ["/", "System ownership"],
  ["/journey", "Protocol-v2 run journey"],
  ["/execution", "Execution modes and controls"],
  ["/data-plane", "Dataset and endpoint data plane"],
  ["/observability", "Observability and evaluation"],
  ["/parity", "Parity and migration ledger"],
] as const;

async function open(page: Page, path: string) {
  await page.goto(path);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
}

test.describe("Architecture Atlas production build", () => {
  const runtimeErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    runtimeErrors.length = 0;
    page.on("pageerror", (error) => runtimeErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") {
        runtimeErrors.push(message.text());
      }
    });
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  for (const [path, heading] of guidedRoutes) {
    test(`navigates guided route ${path}`, async ({ page }) => {
      await open(page, `${path}?audience=developer`);
      await expect(
        page.getByRole("heading", { level: 1, name: heading }),
      ).toBeVisible();
      await expect(page.getByRole("main")).toHaveAttribute(
        "id",
        "atlas-content",
      );
    });
  }

  test("switches all audience lenses and changes framing", async ({ page }) => {
    await open(page, "/execution?audience=developer");
    const audience = page.getByRole("combobox", { name: "Audience" });

    await audience.selectOption("executive");
    await expect(page).toHaveURL(/audience=executive/u);
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: "Ways to run",
      }),
    ).toBeVisible();

    await audience.selectOption("maintainer");
    await expect(page).toHaveURL(/audience=maintainer/u);
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: "Clock, scheduling, transport, and placement matrix",
      }),
    ).toBeVisible();

    await audience.selectOption("developer");
    await expect(page).toHaveURL(/audience=developer/u);
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: "Execution modes and controls",
      }),
    ).toBeVisible();
  });

  test("applies and clears guided filters in URL state", async ({ page }) => {
    await open(page, "/parity?audience=developer");
    const mode = page.getByRole("checkbox", { name: "Native gRPC" });
    const status = page.getByRole("checkbox", { name: "Unbuilt" });

    await mode.check();
    await status.check();
    await expect(page).toHaveURL(/modes=online_grpc/u);
    await expect(page).toHaveURL(/statuses=unbuilt/u);
    await expect(
      page.getByRole("status", { name: "Filtered result count" }),
    ).toContainText("result");

    await mode.uncheck();
    await status.uncheck();
    await expect(page).not.toHaveURL(/modes=/u);
    await expect(page).not.toHaveURL(/statuses=/u);
  });

  test("enters, navigates, and exits presentation mode", async ({ page }) => {
    await open(page, "/journey?audience=executive");
    await page.getByRole("button", { name: "Present this view" }).click();
    await expect(page).toHaveURL(/present=true/u);
    await expect(
      page.getByRole("navigation", { name: "Presentation routes" }),
    ).toBeVisible();

    await page.keyboard.press("ArrowRight");
    await expect(page).toHaveURL(/\/execution\?/u);
    await page.keyboard.press("Escape");
    await expect(page).not.toHaveURL(/present=true/u);
    await expect(
      page.getByRole("button", { name: "Present this view" }),
    ).toBeFocused();
  });

  test("searches, selects, and closes the unified atlas", async ({ page }) => {
    await open(page, "/atlas?audience=developer");
    const search = page.getByRole("searchbox", { name: "Search atlas" });
    await search.fill("clock");
    await expect(page).toHaveURL(/query=clock/u);
    await page.getByText("Text inventory").click();
    const trigger = page.getByRole("button", {
      name: /^Injected execution clock/u,
    });
    await trigger.click();
    await expect(page).toHaveURL(/selected=component.clock-seam/u);
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByRole("button", { name: "Clear selected component" }).click();
    await expect(page).not.toHaveURL(/selected=/u);
    await expect(trigger).toBeFocused();
  });

  test("supports node and crate deep links including unknown crates", async ({
    page,
  }) => {
    await open(
      page,
      "/atlas?audience=maintainer&selected=component.clock-seam",
    );
    await expect(
      page.getByRole("dialog", {
        name: "Clock with RealClock and SimClock",
      }),
    ).toBeVisible();

    await open(page, "/crates/aiperf-clock?audience=maintainer");
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: "aiperf-clock package",
      }),
    ).toBeVisible();
    const evidence = page.locator(".evidence-citations a").last();
    await expect(evidence).toHaveAttribute(
      "href",
      /^https:\/\/github\.com\/ai-dynamo\/aiperf\//u,
    );

    await open(page, "/crates/not-real?audience=developer");
    await expect(
      page.getByRole("heading", { level: 1, name: "Crate not found" }),
    ).toBeVisible();
  });

  test("passes axe on core guided and atlas routes", async ({ page }) => {
    for (const path of [
      "/?audience=developer",
      "/atlas?audience=maintainer&selected=component.clock-seam",
    ]) {
      await open(page, path);
      const results = await new AxeBuilder({ page }).analyze();
      expect(results.violations).toEqual([]);
    }
  });
});
