/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { createSdkRegistry } from "../registry.js";
import type { SceneFragment, SdkExpansionContext } from "../types.js";

const GENERIC_CATALOG_IDS = [
  "sdk.shape",
  "sdk.text",
  "sdk.richText",
  "sdk.icon",
  "sdk.image",
  "sdk.line",
  "sdk.arrow",
  "sdk.spacer",
  "sdk.inset",
  "sdk.title",
  "sdk.paragraph",
  "sdk.caption",
  "sdk.codeBlock",
  "sdk.quote",
  "sdk.list",
  "sdk.keyValue",
  "sdk.propertyList",
  "sdk.badge",
  "sdk.statusDot",
  "sdk.avatar",
  "sdk.iconLabel",
  "sdk.alert",
  "sdk.statusCard",
  "sdk.emptyState",
  "sdk.stat",
  "sdk.metric",
  "sdk.table",
  "sdk.tableRow",
  "sdk.tableCell",
  "sdk.tagList",
  "sdk.breadcrumb",
  "sdk.tabs",
  "sdk.pagination",
  "sdk.timeline",
  "sdk.timelineItem",
  "sdk.progress",
  "sdk.meter",
  "sdk.gauge",
  "sdk.sparkline",
  "sdk.rating",
  "sdk.semaphore",
  "sdk.section",
  "sdk.toolbar",
  "sdk.splitPane",
  "sdk.mediaObject",
] as const;

const SOURCE_MAP = {
  source: "catalog.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

const CHILD: SceneFragment = {
  roots: [
    {
      kind: "rect",
      id: "child",
      capabilityId: "core.rect",
      geometry: { x: 0, y: 0, width: 40, height: 20 },
      style: {},
      accessibility: { label: "child" },
      fallback: "child",
      sourceMap: SOURCE_MAP,
    },
  ],
  ports: { self: { nodeId: "child" } },
  actions: { enter: ["child"] },
};

const BASE_PROPS = {
  id: "example",
  text: "Example",
  title: "Example",
  label: "Example",
  detail: "Detail",
  value: 0.6,
  min: 0,
  max: 1,
  icon: "check",
  src: "/example.svg",
  from: { x: 0, y: 0 },
  to: { x: 100, y: 0 },
  items: ["One", "Two"],
  entries: [
    { key: "one", label: "One", value: "1" },
    { key: "two", label: "Two", value: "2" },
  ],
  values: [1, 3, 2],
  columns: ["Name", "Value"],
} as const;

describe("generic SDK catalog", () => {
  it("registers every approved generic primitive", () => {
    const registry = createSdkRegistry();

    expect(GENERIC_CATALOG_IDS.filter((id) => registry.lookup(id) === undefined)).toEqual([]);
  });

  it("expands every approved generic primitive into usable Scene IR", () => {
    const registry = createSdkRegistry();
    const failures: string[] = [];

    for (const componentId of GENERIC_CATALOG_IDS) {
      const definition = registry.lookup(componentId)!;
      const context: SdkExpansionContext = {
        instanceId: componentId.replace(".", "-"),
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      };
      const result = definition.factory(
        BASE_PROPS,
        {
          children: [CHILD],
          content: [CHILD],
          rows: [CHILD],
          cells: [CHILD],
          items: [CHILD],
          media: [CHILD],
          body: [CHILD],
          leading: [CHILD],
          trailing: [CHILD],
        },
        context,
      );
      if (!result.ok || result.value.roots.length === 0) {
        failures.push(componentId);
      }
    }

    expect(failures).toEqual([]);
  });

  it("keeps sparkline path semantics on its generated series child", () => {
    const definition = createSdkRegistry().lookup("sdk.sparkline")!;
    const result = definition.factory(
      { id: "spark", values: [1, 3, 2] },
      {},
      { instanceId: "spark", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.roots[0]?.capabilityId).toBe("core.group");
      expect(result.value.roots[0]?.kind).toBe("group");
      expect(result.value.roots[0]?.kind === "group" && result.value.roots[0].children[0]?.path)
        .toMatch(/^M/);
    }
  });

  it("maps shape variants to renderer capabilities and rejects unknown variants", () => {
    const definition = createSdkRegistry().lookup("sdk.shape")!;
    const context = {
      instanceId: "shape",
      sourceMap: SOURCE_MAP,
      themeTokens: new Map(),
    };

    const circle = definition.factory({ id: "shape", variant: "circle" }, {}, context);
    const invalid = definition.factory({ id: "shape", variant: "hexagon" }, {}, context);

    expect(circle.ok && circle.value.roots[0]?.capabilityId).toBe("core.circle");
    expect(invalid.ok).toBe(false);
    expect(!invalid.ok && invalid.diagnostics[0]?.code).toBe("SDK_SHAPE_VARIANT_INVALID");
  });

  it("preserves media-object slot order and exposes semantic child ports", () => {
    const definition = createSdkRegistry().lookup("sdk.mediaObject")!;
    const fragment = (id: string): SceneFragment => ({
      ...CHILD,
      roots: [{ ...CHILD.roots[0]!, id }],
      ports: { self: { nodeId: id } },
    });
    const result = definition.factory(
      { id: "media", label: "Media object" },
      {
        leading: [fragment("leading")],
        media: [fragment("media")],
        body: [fragment("body")],
        trailing: [fragment("trailing")],
      },
      { instanceId: "media", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      expect(root?.kind === "group" && root.children.map((child) => child.id)).toEqual([
        "leading",
        "media",
        "body",
        "trailing",
      ]);
      expect(Object.keys(result.value.ports)).toEqual(
        expect.arrayContaining(["leading", "media", "body", "trailing"]),
      );
    }
  });

  it("exposes semantic table row and cell ports", () => {
    const registry = createSdkRegistry();
    const rowDefinition = registry.lookup("sdk.tableRow")!;
    const context = (instanceId: string) => ({
      instanceId,
      sourceMap: SOURCE_MAP,
      themeTokens: new Map(),
    });
    const row = rowDefinition.factory(
      { id: "row" },
      { cells: [CHILD] },
      context("row"),
    );
    expect(row.ok && row.value.ports["cell[0]"]).toEqual({ nodeId: "child" });

    if (row.ok) {
      const table = registry.lookup("sdk.table")!.factory(
        { id: "table", columns: ["Value"] },
        { rows: [row.value] },
        context("table"),
      );
      expect(table.ok && table.value.ports["row[0]"]).toEqual({ nodeId: "row" });
      expect(table.ok && table.value.ports["cell[0][0]"]).toEqual({ nodeId: "child" });
    }
  });

  it("computes deterministic equal-width table columns", () => {
    const fragment = (id: string): SceneFragment => ({
      ...CHILD,
      roots: [{ ...CHILD.roots[0]!, id }],
      ports: { self: { nodeId: id } },
    });
    const row = createSdkRegistry().lookup("sdk.tableRow")!.factory(
      { id: "row", width: 200, height: 40 },
      { cells: [fragment("left"), fragment("right")] },
      { instanceId: "row", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(row.ok).toBe(true);
    if (row.ok) {
      const root = row.value.roots[0];
      expect(
        root?.kind === "group"
          ? root.children.map((cell) => [cell.geometry.x, cell.geometry.width])
          : [],
      ).toEqual([
        [0, 100],
        [100, 100],
      ]);
    }
  });

  it("synthesizes visible table content from concise column props", () => {
    const registry = createSdkRegistry();
    const context = {
      instanceId: "concise-table",
      sourceMap: SOURCE_MAP,
      themeTokens: new Map(),
    };
    const table = registry.lookup("sdk.table")!.factory(
      { id: "concise-table", columns: ["Name", "Value"], width: 200, height: 40 },
      {},
      context,
    );
    const row = registry.lookup("sdk.tableRow")!.factory(
      { id: "concise-row", label: "Row" },
      {},
      { ...context, instanceId: "concise-row" },
    );

    expect(table.ok).toBe(true);
    expect(row.ok).toBe(true);
    if (table.ok && row.ok) {
      expect(table.value.roots[0]?.kind === "group" && table.value.roots[0].children.length)
        .toBeGreaterThan(0);
      expect(table.value.ports["row[0]"]).toBeDefined();
      expect(table.value.ports["cell[0][0]"]).toBeDefined();
      expect(row.value.roots[0]?.kind === "group" && row.value.roots[0].children[0]?.kind)
        .toBe("text");
    }
  });

  it("renders a labeled placeholder for empty visible containers", () => {
    const definition = createSdkRegistry().lookup("sdk.toolbar")!;
    const result = definition.factory(
      { id: "toolbar", label: "Toolbar" },
      {},
      { instanceId: "toolbar", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      expect(root?.kind === "group" && root.children[0]?.kind).toBe("text");
    }
  });

  it("emits native semantic chrome and layout roots", () => {
    const registry = createSdkRegistry();
    const context = {
      instanceId: "semantic",
      sourceMap: SOURCE_MAP,
      themeTokens: new Map(),
    };
    const panel = registry.lookup("sdk.panel")!.factory(
      { id: "semantic", title: "Profile", detail: "source" },
      {},
      context,
    );
    const stepper = registry.lookup("sdk.stepper")!.factory(
      {
        id: "semantic",
        steps: ["layout", "slots", "timeline"],
        linked: true,
      },
      {},
      context,
    );

    expect(panel.ok).toBe(true);
    expect(stepper.ok).toBe(true);
    if (panel.ok && stepper.ok) {
      expect(panel.value.roots[0]).toMatchObject({
        capabilityId: "core.panel",
        props: { title: "Profile", detail: "source" },
      });
      expect(
        panel.value.roots[0]?.kind === "group"
          ? panel.value.roots[0].children.map((child) => child.capabilityId)
          : [],
      ).not.toEqual(expect.arrayContaining(["core.rect", "core.text"]));
      expect(stepper.value.roots[0]).toMatchObject({
        capabilityId: "core.stepper",
        props: {
          steps: ["layout", "slots", "timeline"],
          linked: true,
        },
      });
      expect(
        stepper.value.roots[0]?.kind === "group"
          ? stepper.value.roots[0].children.map((child) => child.id)
          : [],
      ).toEqual([
        "semantic-step-0",
        "semantic-step-1",
        "semantic-step-2",
      ]);
    }
  });

  it("publishes strict family-specific descriptors", () => {
    const registry = createSdkRegistry();
    const titleProps = registry.lookup("sdk.title")!.descriptor.props;
    const sparklineProps = registry.lookup("sdk.sparkline")!.descriptor.props;

    expect(titleProps.text).toBeDefined();
    expect(titleProps.values).toBeUndefined();
    expect(sparklineProps.values).toBeDefined();
    expect(sparklineProps.src).toBeUndefined();
  });

  it("keeps emitted action bindings within each public action contract", () => {
    const registry = createSdkRegistry();
    const mismatches: string[] = [];
    for (const componentId of GENERIC_CATALOG_IDS) {
      const definition = registry.lookup(componentId)!;
      const result = definition.factory(
        BASE_PROPS,
        { children: [CHILD], rows: [CHILD], cells: [CHILD], body: [CHILD] },
        {
          instanceId: componentId.replace(".", "-"),
          sourceMap: SOURCE_MAP,
          themeTokens: new Map(),
        },
      );
      if (
        result.ok &&
        Object.keys(result.value.actions).some(
          (action) => !definition.actions.includes(action as never),
        )
      ) {
        mismatches.push(componentId);
      }
    }
    expect(mismatches).toEqual([]);
  });

  it("keeps generated icon geometry nonnegative in compact components", () => {
    const result = createSdkRegistry().lookup("sdk.avatar")!.factory(
      { id: "tiny", icon: "user", width: 8, height: 8 },
      {},
      { instanceId: "tiny", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      const icon = root?.kind === "group" ? root.children.find((child) => child.id.endsWith("__icon")) : undefined;
      expect(icon?.geometry.width).toBeGreaterThanOrEqual(0);
      expect(icon?.geometry.height).toBeGreaterThanOrEqual(0);
    }
  });
});
