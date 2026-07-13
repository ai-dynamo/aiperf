// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Locator, type Page } from "@playwright/test";

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
const graphStateStorageKey = "aiperf-atlas:graph-state:v1";

function encodedGraphStateFromUrl(page: Page): string {
  const current = new URL(page.url());
  return current.searchParams.get("s") ?? "";
}

async function storedGraphState(page: Page): Promise<string> {
  return page.evaluate((key) => {
    const storage = (
      globalThis as {
        localStorage?: { getItem: (storageKey: string) => string | null };
      }
    ).localStorage;
    return storage?.getItem(key) ?? "";
  }, graphStateStorageKey);
}

async function storedNodePosition(
  page: Page,
  nodeId: string,
): Promise<{ x: number; y: number } | null> {
  const raw = await storedGraphState(page);
  if (!raw) {
    return null;
  }
  const parsed = JSON.parse(raw) as {
    nodePositions?: Array<{ nodeId: string; x: number; y: number }>;
  };
  const position = parsed.nodePositions?.find(
    (candidate) => candidate.nodeId === nodeId,
  );
  return position ? { x: position.x, y: position.y } : null;
}

async function dragNodeBy(
  page: Page,
  nodeId: string,
  delta: { x: number; y: number },
): Promise<void> {
  const box = await page
    .getByTestId(`graph-node-drag-handle-${nodeId}`)
    .boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box!.x + box!.width / 2 + delta.x,
    box!.y + box!.height / 2 + delta.y,
    { steps: 8 },
  );
  await page.mouse.up();
}

async function dragNodeByTouch(
  page: Page,
  nodeId: string,
  delta: { x: number; y: number },
): Promise<void> {
  const box = await page
    .getByTestId(`graph-node-drag-handle-${nodeId}`)
    .boundingBox();
  expect(box).not.toBeNull();
  const session = await page.context().newCDPSession(page);
  const start = {
    x: box!.x + box!.width / 2,
    y: box!.y + box!.height / 2,
  };
  try {
    await session.send("Input.dispatchTouchEvent", {
      touchPoints: [start],
      type: "touchStart",
    });
    for (let step = 1; step <= 8; step += 1) {
      await session.send("Input.dispatchTouchEvent", {
        touchPoints: [
          {
            x: start.x + (delta.x * step) / 8,
            y: start.y + (delta.y * step) / 8,
          },
        ],
        type: "touchMove",
      });
    }
    await session.send("Input.dispatchTouchEvent", {
      touchPoints: [],
      type: "touchEnd",
    });
  } finally {
    await session.detach();
  }
}

async function moveWaypointUntilActionable(
  page: Page,
  waypointHandle: Locator,
  actionTarget: Locator,
): Promise<void> {
  const directions = [
    "ArrowRight",
    "ArrowDown",
    "ArrowLeft",
    "ArrowUp",
  ] as const;
  let legLength = 8;
  for (let leg = 0; leg < 16; leg += 1) {
    try {
      await actionTarget.click({ timeout: 100, trial: true });
      return;
    } catch {
      await waypointHandle.focus();
      const direction = directions[leg % directions.length];
      for (let step = 0; step < legLength; step += 1) {
        await page.keyboard.press(direction);
      }
      if (leg % 2 === 1) {
        legLength += 8;
      }
    }
  }
  await expect(actionTarget).toBeVisible();
  await actionTarget.click({ timeout: 100, trial: true });
}

async function openScene(page: Page, path: string, search = "audience=developer") {
  await page.goto(`${path}?${search}`);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expectProductionLayoutReady(page);
}

async function expectProductionLayoutReady(page: Page) {
  await expect(page.getByRole("status", { name: "Graph layout status" })).toHaveText(
    "Graph layout ready.",
  );
}

async function focusRuntimeDispatchEdge(page: Page) {
  await page.getByRole("button", { name: "Show graph accessibility outline" }).click();
  await page
    .getByRole("button", { name: /Select edge .*RequestObserver callbacks/u })
    .first()
    .click();
  await page
    .getByRole("button", { name: "Hide graph accessibility outline" })
    .click();
}

test.describe("Architecture Atlas functional journey", () => {
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

  test("renders default runtime journey on desktop and mobile", async ({ page }) => {
    await openScene(page, "/");
    await expect(page.getByRole("heading", { level: 1, name: "Runtime composition" })).toBeVisible();
    await expect(page.getByRole("searchbox", { name: "Graph search" })).toBeVisible();
    await expect(page.getByRole("radiogroup", { name: "Audience" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Fit graph" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Share graph state" })).toBeVisible();
    await expect(page.getByRole("main")).toHaveAttribute("id", "atlas-content");
  });

  test("keeps ELK layout healthy in production preview", async ({ page }) => {
    await openScene(page, "/");
    await expectProductionLayoutReady(page);
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

    const audience = page.getByRole("radiogroup", { name: "Audience" });
    await audience.getByRole("radio", { name: "Executive" }).click();
    await expect(page).toHaveURL(/audience=executive/u);
    await expect(
      page
        .getByTestId("graph-node-node.runner-protocol-registries")
        .getByRole("button", { name: "Runner protocol and registries", exact: true }),
    ).toBeVisible();

    await audience.getByRole("radio", { name: "Maintainer" }).click();
    await expect(page).toHaveURL(/audience=maintainer/u);
    await expect(
      page
        .getByTestId("graph-node-node.runner-protocol-registries")
        .getByRole("button", {
          name: "Strict protocol-v2 RunnerApplication registries",
          exact: true,
        }),
    ).toBeVisible();

    await audience.getByRole("radio", { name: "Developer" }).click();
    await page.getByRole("combobox", { name: "Compare flavor" }).selectOption("dynamo_online");
    await expect(page).toHaveURL(/compare=dynamo_online/u);

    const plannedDynamoPair = page.getByTestId("graph-node-node.dynamo-online-runner-pair");
    await expect(plannedDynamoPair).toBeVisible();
    await expect(plannedDynamoPair).toHaveAttribute("data-flavor-class", "compare-only");
    await expect(plannedDynamoPair).toHaveAttribute("data-implementation-state", "planned");
  });

  test("persists dragged node layout state after reload", async ({
    page,
    isMobile,
  }) => {
    await openScene(page, "/scenes/metrics-telemetry");
    const node = page.getByTestId("graph-node-node.journey.metrics-and-reporting");
    const initialBox = await node.boundingBox();
    expect(initialBox).not.toBeNull();
    const initialUrlState = encodedGraphStateFromUrl(page);
    const initialStoredState = await storedGraphState(page);
    const initialStoredPosition = await storedNodePosition(
      page,
      "node.journey.metrics-and-reporting",
    );

    const drag = isMobile ? dragNodeByTouch : dragNodeBy;
    await drag(page, "node.journey.metrics-and-reporting", { x: 72, y: 56 });

    await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(initialUrlState);
    await expect.poll(() => storedGraphState(page)).not.toBe(initialStoredState);
    await expect
      .poll(() =>
        storedNodePosition(page, "node.journey.metrics-and-reporting"),
      )
      .not.toEqual(initialStoredPosition);
    const draggedPosition = await storedNodePosition(
      page,
      "node.journey.metrics-and-reporting",
    );
    expect(draggedPosition).not.toBeNull();
    const draggedState = encodedGraphStateFromUrl(page);
    const draggedBox = await node.boundingBox();
    expect(draggedBox).not.toBeNull();
    expect(Math.abs(draggedBox!.x - initialBox!.x)).toBeGreaterThan(20);
    expect(Math.abs(draggedBox!.y - initialBox!.y)).toBeGreaterThan(20);

    await page.reload();
    await expectProductionLayoutReady(page);
    await expect.poll(() => encodedGraphStateFromUrl(page)).toBe(draggedState);
    await expect(node).toBeVisible();
    await expect
      .poll(() =>
        storedNodePosition(page, "node.journey.metrics-and-reporting"),
      )
      .toEqual(draggedPosition);
    await expect
      .poll(async () => {
        const reloadedBox = await node.boundingBox();
        return reloadedBox
          ? {
              x: Math.round(reloadedBox.x),
              y: Math.round(reloadedBox.y),
            }
          : null;
      })
      .toEqual({
        x: Math.round(draggedBox!.x),
        y: Math.round(draggedBox!.y),
      });
  });

  test("supports expansion, drilldown, evidence drawer, and focus restoration", async ({ page }) => {
    await openScene(page, "/", "audience=executive");

    await page
      .getByRole("button", { name: "Show graph accessibility outline" })
      .click();
    const runtimeOutlineItem = page.getByRole("treeitem", {
      name: "Node Runtime composition",
    });
    await runtimeOutlineItem
      .getByRole("button", { name: "Expand", exact: true })
      .click();
    await expectProductionLayoutReady(page);
    await expect(
      page.getByRole("treeitem", { name: "Node Clock seam" }),
    ).toBeVisible();

    const outlineClockTrigger = page.getByRole("button", {
      name: "Select node Clock seam",
      exact: true,
    });
    await outlineClockTrigger.click();
    await expect(page.getByRole("dialog", { name: "Clock seam evidence" })).toBeVisible();
    await page
      .getByRole("button", { name: "Hide graph accessibility outline" })
      .click();

    await page.getByRole("button", { name: "Close evidence panel" }).click();
    await expect(page.getByRole("dialog", { name: "Clock seam evidence" })).toHaveCount(0);
    await expect(
      page.locator('[data-graph-entity-id="node.clock-seam"]:focus'),
    ).toHaveCount(1);

    await page
      .getByRole("button", { name: "Show graph accessibility outline" })
      .click();
    const runtimeOutlineItemAfterClose = page.getByRole("treeitem", {
      name: "Node Runtime composition",
    });
    await runtimeOutlineItemAfterClose
      .getByRole("button", { name: "Collapse", exact: true })
      .click();
    await expect(
      page.getByRole("treeitem", { name: "Node Clock seam" }),
    ).toHaveCount(0);
  });

  test("supports edge waypoint pointer and keyboard editing", async ({
    page,
    isMobile,
  }) => {
    await openScene(page, "/scenes/metrics-telemetry");
    await dragNodeBy(page, "node.journey.metrics-and-reporting", {
      x: -180,
      y: 100,
    });
    await dragNodeBy(page, "node.runtime-composition", {
      x: 180,
      y: -100,
    });
    await focusRuntimeDispatchEdge(page);
    const addWaypoint = page.getByTestId("graph-edge-waypoint-add-edge.runtime.dispatch.metrics");
    await expect(addWaypoint).toBeVisible();
    await addWaypoint.click();

    const waypointHandle = page.getByTestId("graph-edge-waypoint-handle-edge.runtime.dispatch.metrics-0");
    await expect(waypointHandle).toHaveCount(1);
    await expect(waypointHandle).toBeVisible();
    const initialState = encodedGraphStateFromUrl(page);

    await waypointHandle.focus();
    await page.keyboard.press("ArrowRight");
    if (!isMobile) {
      await moveWaypointUntilActionable(page, waypointHandle, waypointHandle);
    }
    await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(initialState);
    const keyboardPreparationState = encodedGraphStateFromUrl(page);

    let pointerState = keyboardPreparationState;
    if (!isMobile) {
      await waypointHandle.dragTo(addWaypoint);
      await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(keyboardPreparationState);
      pointerState = encodedGraphStateFromUrl(page);
    }

    await waypointHandle.focus();
    for (let step = 0; step < 8; step += 1) {
      await page.keyboard.press("ArrowRight");
      await page.keyboard.press("ArrowDown");
    }
    await expect.poll(() => encodedGraphStateFromUrl(page)).not.toBe(pointerState);
    const keyboardState = encodedGraphStateFromUrl(page);

    const removeWaypoint = page.getByTestId(
      "graph-edge-waypoint-remove-edge.runtime.dispatch.metrics-0",
    );
    await moveWaypointUntilActionable(page, waypointHandle, removeWaypoint);
    await removeWaypoint.click();
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
    await page.getByRole("button", { name: "Close evidence panel" }).click();
    await expect(page.getByRole("dialog")).toHaveCount(0);

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
