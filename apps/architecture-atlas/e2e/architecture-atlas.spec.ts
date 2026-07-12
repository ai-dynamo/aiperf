// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

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

async function leadingCount(locator: Locator) {
  await expect(locator).toContainText(/\d+/u);
  return Number.parseInt(await locator.innerText(), 10);
}

async function graphCounts(locator: Locator) {
  await expect(locator).toContainText(/\d+ components, \d+ connections/u);
  const match = (await locator.innerText()).match(
    /(\d+) components, (\d+) connections/u,
  );
  if (!match) {
    throw new Error("Atlas graph summary did not contain component counts.");
  }
  return { components: Number(match[1]), connections: Number(match[2]) };
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
    const results = page.getByRole("status", {
      name: "Filtered result count",
    });
    const initialCount = await leadingCount(results);

    await mode.check();
    await status.check();
    await expect(page).toHaveURL(/modes=online_grpc/u);
    await expect(page).toHaveURL(/statuses=unbuilt/u);
    await expect.poll(() => leadingCount(results)).toBeLessThan(initialCount);
    await expect(
      page.getByRole("heading", { name: "gRPC lifecycle limits" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Offline mode constraints" }),
    ).toHaveCount(0);
    const filteredRows = page.locator(".parity-ledger > li");
    await expect(filteredRows).toHaveCount(await leadingCount(results));
    for (const row of await filteredRows.allTextContents()) {
      expect(row).toContain("Native gRPC");
      expect(row).toContain("Unbuilt");
    }

    await mode.uncheck();
    await status.uncheck();
    await expect(page).not.toHaveURL(/modes=/u);
    await expect(page).not.toHaveURL(/statuses=/u);
    await expect.poll(() => leadingCount(results)).toBe(initialCount);
  });

  test("filters the atlas without stale or dangling inventory", async ({
    page,
    isMobile,
  }) => {
    test.skip(isMobile, "Detailed filter assertions run once on desktop.");
    await open(page, "/atlas?audience=developer");
    await page.getByText("Text inventory").click();
    const summary = page.getByRole("status", { name: "Atlas graph summary" });
    const inventory = page.getByRole("list", {
      name: "Visible architecture components",
    });
    const initial = await graphCounts(summary);

    await page.getByRole("checkbox", { name: "Native gRPC" }).check();
    await expect(page).toHaveURL(/modes=online_grpc/u);
    const grpc = await graphCounts(summary);
    expect(grpc.components).toBeLessThan(initial.components);
    expect(grpc.connections).toBeLessThan(initial.connections);
    await expect(
      inventory.getByRole("button", {
        name: /^KServe and Riva gRPC transport/u,
      }),
    ).toBeVisible();
    await expect(
      inventory.getByRole("button", {
        name: /^Native HTTP and SSE transport/u,
      }),
    ).toHaveCount(0);

    await page.getByRole("checkbox", { name: "Native gRPC" }).uncheck();
    await page.getByRole("checkbox", { name: "Feature-gated" }).check();
    await expect(page).toHaveURL(/statuses=feature-gated/u);
    const featureGated = await graphCounts(summary);
    expect(featureGated.components).toBeLessThan(initial.components);
    for (const row of await inventory.getByRole("button").allTextContents()) {
      expect(row).toContain("Feature-gated");
    }
    await expect(
      inventory.getByRole("button", {
        name: /^In-process Dynamo offline backend/u,
      }),
    ).toBeVisible();

    await page.getByRole("checkbox", { name: "Feature-gated" }).uncheck();
    await page.getByRole("checkbox", { name: "Rust execution" }).check();
    await expect(page).toHaveURL(/ownership=rust/u);
    const rust = await graphCounts(summary);
    expect(rust.components).toBeLessThan(initial.components);
    for (const row of await inventory.getByRole("button").allTextContents()) {
      expect(row).toContain("Rust execution");
    }
    await expect(
      inventory.getByRole("button", {
        name: /^HTTP, gRPC, or mock inference target/u,
      }),
    ).toHaveCount(0);
    await expect(inventory.getByRole("listitem")).toHaveCount(rust.components);
    await expect(page.locator(".react-flow__node-component")).toHaveCount(
      rust.components,
    );
    await expect(page.locator(".react-flow__edge")).toHaveCount(
      rust.connections,
    );
  });

  test("keeps combined atlas filters accurate on mobile", async ({
    page,
    isMobile,
  }) => {
    test.skip(!isMobile, "Mobile-specific filter coverage.");
    await open(page, "/atlas?audience=developer");
    const summary = page.getByRole("status", { name: "Atlas graph summary" });
    const initial = await graphCounts(summary);

    await page.getByRole("checkbox", { name: "Native gRPC" }).check();
    await page
      .getByRole("checkbox", { name: "Built", exact: true })
      .check();
    await page.getByRole("checkbox", { name: "Rust execution" }).check();
    await expect(page).toHaveURL(/modes=online_grpc/u);
    await expect(page).toHaveURL(/statuses=built/u);
    await expect(page).toHaveURL(/ownership=rust/u);
    const filtered = await graphCounts(summary);
    expect(filtered.components).toBeLessThan(initial.components);
    await page.getByText("Text inventory").click();
    const rows = page
      .getByRole("list", { name: "Visible architecture components" })
      .getByRole("button");
    await expect(rows).toHaveCount(filtered.components);
    for (const row of await rows.allTextContents()) {
      expect(row).toContain("Rust execution");
      expect(row).toContain("Built");
    }
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
