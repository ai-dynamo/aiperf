// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

const canonicalScenes = [
  ["/", "Runtime composition"],
  ["/scenes/runner-protocol-registries", "Runner protocol and registries"],
  ["/scenes/scheduling-phase-lifecycle", "Scheduling and phase lifecycle"],
  ["/scenes/dataset-segment-pipeline", "Dataset and segment pipeline"],
  ["/scenes/endpoint-bindings-transports", "Endpoint bindings and HTTP/gRPC transports"],
  ["/scenes/graph-ir-execution", "Graph-IR execution"],
  ["/scenes/metrics-telemetry", "Metrics and telemetry"],
  ["/scenes/accuracy-evaluator-hosting", "Accuracy and evaluator hosting"],
  ["/scenes/crate-dependency-topology", "Crate dependency topology"],
] as const;

const legacyRedirects = [
  ["/journey", "/"],
  ["/execution", "/scenes/endpoint-bindings-transports"],
  ["/data-plane", "/scenes/dataset-segment-pipeline"],
  ["/observability", "/scenes/metrics-telemetry"],
  ["/parity", "/scenes/crate-dependency-topology"],
  ["/atlas", "/"],
] as const;

const seededMetricsWaypointState =
  "N4IgbgpgTgzglgewHYgFwEYA0IYGMJIQCSAJmjvoQHQC2EALlHLjALT0QA2EdjAniGwBDAK4k4BfORIRInBAAdogkAqY0hUPgDFOQsAijkkQ+nEgB9ABb16ClbgQ0FmiLv2G0SEZ07YIAB4uSDIkAHIIMqQwaADaALrYAGYIuCIwECQAokhm9Hyk5JkA5hBUUCK5cHRU4jAu9LhWtAxMLCqMQvgAspEQxsj92Eh9AAoI8GbIMagJ-iSlAOpCfAoIcLkzsaAlxGSoILvllWY1dQ1NLYzMMdhrG-RboAFoAMwATO-YAqivABwAFgAvvEQdhTlwNhBxpNEChUAAGIFAA";

function encodedGraphStateFromUrl(page: Page): string {
  const current = new URL(page.url());
  return current.searchParams.get("s") ?? "";
}

async function openScene(page: Page, path: string, search = "audience=developer") {
  await page.goto(`${path}?${search}`);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByRole("status", { name: "Graph layout status" })).toContainText(
    /Graph layout ready|Graph layout degraded/u,
  );
}

async function focusRuntimeDispatchEdge(page: Page) {
  await page.getByRole("button", { name: "Show graph accessibility outline" }).click();
  await page
    .getByRole("button", { name: /Select edge .*RequestObserver callbacks/u })
    .first()
    .click();
}

test.describe("Architecture Atlas functional journey", () => {
  const runtimeErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    runtimeErrors.length = 0;
    page.on("pageerror", (error) => runtimeErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") {
        const text = message.text();
        if (/net::ERR_INTERNET_DISCONNECTED/u.test(text)) {
          return;
        }
        runtimeErrors.push(text);
      }
    });
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  test("renders default runtime journey on desktop and mobile", async ({ page }) => {
    await openScene(page, "/");
    await expect(page.getByRole("heading", { level: 1, name: "Runtime composition" })).toBeVisible();
    await expect(page.getByRole("searchbox", { name: "Graph search" })).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Audience" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Fit graph" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Share graph state" })).toBeVisible();
    await expect(page.getByRole("main")).toHaveAttribute("id", "atlas-content");
  });

  test("navigates all nine canonical graph scenes", async ({ page }) => {
    for (const [path, heading] of canonicalScenes) {
      await openScene(page, path);
      await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
      await expect(page).toHaveURL(new RegExp(`${path.replace("/", "\\/")}\\?`, "u"));
    }
  });

  test("switches audience and flavor overlays including Dynamo states", async ({ page }) => {
    await openScene(page, "/scenes/runner-protocol-registries", "audience=developer&primary=native_http");

    const audience = page.getByRole("combobox", { name: "Audience" });
    await audience.selectOption("executive");
    await expect(page).toHaveURL(/audience=executive/u);
    await expect(
      page
        .getByTestId("graph-node-node.runner-protocol-registries")
        .getByRole("button", { name: "Runner protocol and registries", exact: true }),
    ).toBeVisible();

    await audience.selectOption("maintainer");
    await expect(page).toHaveURL(/audience=maintainer/u);
    await expect(
      page
        .getByTestId("graph-node-node.runner-protocol-registries")
        .getByRole("button", {
          name: "Strict protocol-v2 RunnerApplication registries",
          exact: true,
        }),
    ).toBeVisible();

    await audience.selectOption("developer");
    await page.getByRole("combobox", { name: "Compare flavor" }).selectOption("dynamo_online");
    await expect(page).toHaveURL(/compare=dynamo_online/u);

    const plannedDynamoPair = page.getByTestId("graph-node-node.dynamo-online-runner-pair");
    await expect(plannedDynamoPair).toBeVisible();
    await expect(plannedDynamoPair).toHaveAttribute("data-flavor-class", "compare-only");
    await expect(plannedDynamoPair).toHaveAttribute("data-implementation-state", "planned");
  });

  test("persists dragged node layout state after reload", async ({ page }) => {
    await openScene(
      page,
      "/",
      "audience=developer&s=N4IgbgpgTgzglgewHYgFwEYA0IYGMJIQCSAJmjvoQHRQCuSALnALYQC0uCzADgvE8hDYAhrRJwC+ciQiQANgm7QhIblBbCoATwBic4WARRySYU0gB9ABYMG3FZx6aIeg0bRJacudggAPbmEkGRIAOQQZUhg0AG0AXWwAMwRcWhgIEgBRRjgGLVIPLx8QBihhfABZCIgTZBrsJGqABT5cxCRo1HjfEgBzCAB1YS1eOEZO7pKWCDkxiBb+drQABgBfIA",
    );
    const draggedState = encodedGraphStateFromUrl(page);
    expect(draggedState).not.toBe("");
    await page.reload();
    await expect.poll(() => encodedGraphStateFromUrl(page)).toBe(draggedState);
    await expect(page.getByTestId("graph-node-node.runtime-composition")).toBeVisible();
  });

  test("supports expansion, drilldown, evidence drawer, and focus restoration", async ({ page }) => {
    await openScene(page, "/", "audience=executive");

    const runtimeNode = page.getByTestId("graph-node-node.runtime-composition");
    await runtimeNode.getByRole("button", { name: "Expand Runtime composition" }).click();
    await expect(page.getByTestId("graph-node-node.clock-seam")).toBeVisible();

    const clockNodeTrigger = page
      .getByTestId("graph-node-node.clock-seam")
      .getByRole("button", { name: "Clock seam", exact: true });
    await clockNodeTrigger.click();
    await expect(page.getByRole("dialog", { name: "Clock seam evidence" })).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "Clock seam evidence" })).toHaveCount(0);
    await expect
      .poll(async () =>
        page.evaluate(() => {
          const activeElement = (
            globalThis as {
              document?: {
                activeElement?: {
                  id?: string;
                  getAttribute?: (name: string) => string | null;
                } | null;
              };
            }
          ).document?.activeElement;
          return (
            activeElement?.id ??
            activeElement?.getAttribute?.("data-graph-entity-id") ??
            ""
          );
        }),
      )
      .toMatch(/atlas-graph-search|node\.clock-seam/u);

    await runtimeNode.getByRole("button", { name: "Collapse Runtime composition" }).click();
    await expect(page.getByTestId("graph-node-node.clock-seam")).toHaveCount(0);
  });

  test("supports edge waypoint pointer and keyboard editing", async ({ page }) => {
    await openScene(page, "/scenes/metrics-telemetry");
    await focusRuntimeDispatchEdge(page);
    const addWaypoint = page.getByTestId("graph-edge-waypoint-add-edge.runtime.dispatch.metrics");
    await expect(addWaypoint).toBeVisible();
    await addWaypoint.evaluate((element) => {
      (element as { click: () => void }).click();
    });

    const waypointHandle = page.getByTestId("graph-edge-waypoint-handle-edge.runtime.dispatch.metrics-0");
    await expect(waypointHandle).toHaveCount(1);
    await expect(waypointHandle).toBeVisible();
    const initialState = encodedGraphStateFromUrl(page);

    await waypointHandle.focus();
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowDown");
    await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(initialState);
    const keyboardState = encodedGraphStateFromUrl(page);

    const removeWaypoint = page.getByTestId(
      "graph-edge-waypoint-remove-edge.runtime.dispatch.metrics-0",
    );
    await removeWaypoint.click({ force: true });
    await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(keyboardState);
    await expect(page.getByTestId("graph-edge-waypoint-handle-edge.runtime.dispatch.metrics-0")).toHaveCount(0);
  });

  test("supports trace controls and pulse playback/scrub", async ({ page }) => {
    await openScene(page, "/scenes/metrics-telemetry");

    const metricsNode = page.getByTestId("graph-node-node.metrics-telemetry");
    await metricsNode
      .getByRole("button", { name: "Trace upstream from Metrics accumulator and telemetry producers" })
      .click();
    await expect(page.getByTestId("graph-node-node.runtime-composition")).toHaveAttribute(
      "data-path-state",
      "upstream",
    );
    await expect(page.getByTestId("graph-edge-edge.runtime.dispatch.metrics")).toHaveAttribute(
      "data-path-state",
      "upstream",
    );

    const narration = page.getByRole("status", { name: "Active pulse narration" });
    const firstNarration = await narration.innerText();
    await page.getByRole("button", { name: "Play pulse timeline" }).click();
    await expect(page.getByRole("button", { name: "Pause pulse timeline" })).toBeVisible();
    await expect.poll(async () => narration.innerText()).not.toBe(firstNarration);

    const scrubber = page.getByRole("slider", { name: "Pulse timeline scrubber" });
    await scrubber.fill("1");
    await expect(page.getByTestId("pulse-active-particle")).toHaveAttribute("data-motion", "animated");
  });

  test("uses reduced-motion pulse semantics without animated particles", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await openScene(page, "/scenes/metrics-telemetry");
    await expect(page.getByText("Motion reduced: semantic playback only.")).toBeVisible();
    await expect(page.getByTestId("pulse-active-particle")).toHaveAttribute("data-motion", "reduced");
  });

  test("shares, resets, and recovers from invalid graph state", async ({ page, context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.addInitScript(() => {
      const scope = globalThis as {
        __copiedGraphStateUrl?: string;
        navigator?: unknown;
      };
      Object.defineProperty(scope, "__copiedGraphStateUrl", {
        configurable: true,
        value: "",
        writable: true,
      });
      const clipboard = {
        writeText: async (value: string) => {
          scope.__copiedGraphStateUrl = value;
        },
      };
      const navigatorObject = scope.navigator as
        | { clipboard?: { writeText: (value: string) => Promise<void> } }
        | undefined;
      if (navigatorObject) {
        Object.defineProperty(navigatorObject, "clipboard", {
          configurable: true,
          value: clipboard,
        });
      }
    });

    await openScene(
      page,
      "/scenes/metrics-telemetry",
      `audience=developer&primary=native_http&s=${seededMetricsWaypointState}`,
    );

    await page.getByRole("button", { name: "Share graph state" }).click();
    const sharedState = encodedGraphStateFromUrl(page);
    expect(sharedState).not.toBe("");

    const copiedUrl = await page.evaluate(
      () =>
        (globalThis as { __copiedGraphStateUrl?: string }).__copiedGraphStateUrl ?? "",
    );
    expect(new URL(copiedUrl).searchParams.get("s")).toBe(sharedState);

    await page.getByRole("button", { name: "Reset graph" }).click();
    await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(sharedState);
    await expect(page.getByTestId("graph-edge-waypoint-handle-edge.runtime.dispatch.metrics-0")).toHaveCount(0);

    await page.goto("/scenes/metrics-telemetry?audience=developer&s=not-valid-state");
    await expect(
      page.getByRole("status", { name: "Graph state recovery notice" }),
    ).toContainText("Shared graph state was invalid; restored canonical scene.");
    await expect(page).not.toHaveURL(/s=not-valid-state/u);
  });

  test("redirects legacy guided routes to canonical scenes", async ({ page }) => {
    for (const [legacyPath, canonicalPath] of legacyRedirects) {
      await page.goto(`${legacyPath}?audience=maintainer`);
      await expect(page).toHaveURL(new RegExp(`${canonicalPath.replace("/", "\\/")}\\?`, "u"));
      await expect(page).toHaveURL(/audience=maintainer/u);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    }
  });
});
