/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { JsonValue } from "../../schema/json-value.js";
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

const BASE_PROPS: Readonly<Record<string, JsonValue>> = {
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
};

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

  it("emits a resolvable accent.danger role for danger variants", () => {
    const definition = createSdkRegistry().lookup("sdk.badge")!;
    const result = definition.factory(
      { id: "danger-badge", label: "BLOCKED", variant: "danger" },
      {},
      { instanceId: "danger-badge", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.roots[0]?.style?.fill).toBe("@theme.accent.danger");
    }
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

  it("emits chrome factories directly as semantic roots with stable contracts", () => {
    const registry = createSdkRegistry();
    const cases = [
      {
        id: "sdk.header",
        props: { id: "semantic", title: "Profile", caption: "source" },
        expectedProps: { title: "Profile", caption: "source" },
        ports: ["title", "caption"],
      },
      {
        id: "sdk.panel",
        props: { id: "semantic", title: "Profile", detail: "source" },
        expectedProps: { title: "Profile", detail: "source" },
        ports: ["title", "detail"],
      },
      {
        id: "sdk.card",
        props: { id: "semantic", title: "Profile", detail: "source", subtitle: "live" },
        expectedProps: { title: "Profile", detail: "source", subtitle: "live" },
        ports: ["title", "detail", "subtitle"],
      },
      {
        id: "sdk.chip",
        props: { id: "semantic", label: "Profile" },
        expectedProps: { label: "Profile" },
        ports: ["label"],
      },
      {
        id: "sdk.note",
        props: { id: "semantic", text: "Profile" },
        expectedProps: { text: "Profile" },
        ports: ["caption"],
      },
    ] as const;

    for (const entry of cases) {
      const result = registry.lookup(entry.id)!.factory(entry.props, {}, {
        instanceId: "semantic",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      });

      expect(result.ok, entry.id).toBe(true);
      if (result.ok) {
        const root = result.value.roots[0];
        expect(root, entry.id).toMatchObject({
          id: "semantic",
          kind: "group",
          props: entry.expectedProps,
          sdkOrigin: {
            componentId: entry.id,
            instanceId: "semantic",
            generatedRole: "root",
          },
        });
        expect(
          root?.kind === "group"
            ? root.children.filter(
                (child) =>
                  child.sdkOrigin?.generatedRole !== undefined &&
                  (child.capabilityId === "core.rect" || child.capabilityId === "core.text"),
              )
            : [],
          entry.id,
        ).toEqual([]);
        expect(Object.keys(result.value.ports), entry.id).toEqual(entry.ports);
        expect(result.value.actions, entry.id).toEqual({
          enter: ["semantic"],
          emphasis: ["semantic"],
          exit: ["semantic"],
        });
      }
    }
  });

  it("emits notes as one semantic owner while preserving the generated caption port", () => {
    const result = createSdkRegistry().lookup("sdk.note")!.factory(
      { id: "note", text: "The worker only executes" },
      {},
      {
        instanceId: "note",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.roots[0]).toMatchObject({
        id: "note",
        capabilityId: "core.note",
        props: { text: "The worker only executes" },
        children: [],
      });
      expect(result.value.ports.caption).toEqual({
        nodeId: "note__caption",
      });
    }
  });

  it.each([
    ["sdk.header", { id: "header", title: "Title", caption: "Caption" }],
    ["sdk.panel", { id: "panel", title: "Title", detail: "Detail" }],
    ["sdk.card", { id: "card", title: "Title", detail: "Detail", subtitle: "Sub" }],
    ["sdk.chip", { id: "chip", label: "Ready" }],
  ] as const)("%s keeps generated chrome out of authored children", (componentId, props) => {
    const result = createSdkRegistry().lookup(componentId)!.factory(
      props,
      {},
      {
        instanceId: props.id,
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(
        result.value.roots[0]?.kind === "group"
          ? result.value.roots[0].children
          : undefined,
      ).toEqual([]);
    }
  });

  it("uses native semantic roots for catalog badge and panel-like chrome", () => {
    const registry = createSdkRegistry();
    const cases = [
      ["sdk.badge", { id: "badge", label: "Ready" }, "core.chip", { label: "Ready" }],
      [
        "sdk.alert",
        { id: "alert", title: "Warning", detail: "Retry" },
        "core.panel",
        { title: "Warning", detail: "Retry" },
      ],
      [
        "sdk.statusCard",
        { id: "status", title: "Healthy", detail: "12 workers" },
        "core.panel",
        { title: "Healthy", detail: "12 workers" },
      ],
      [
        "sdk.emptyState",
        { id: "empty", title: "No runs", detail: "Start a profile" },
        "core.panel",
        { title: "No runs", detail: "Start a profile" },
      ],
      [
        "sdk.codeBlock",
        { id: "code", text: "const n = 1;" },
        "core.group",
        {
          text: "const n = 1;",
          presentation: "code-block",
          inkRole: "@theme.ink.primary",
        },
      ],
      [
        "sdk.quote",
        { id: "quote", text: "Measure twice." },
        "core.group",
        {
          text: "Measure twice.",
          presentation: "quote",
          inkRole: "@theme.ink.primary",
        },
      ],
      [
        "sdk.avatar",
        { id: "avatar", icon: "user" },
        "core.group",
        {
          presentation: "avatar",
          icon: "user",
          inkRole: "@theme.ink.primary",
        },
      ],
      [
        "sdk.iconLabel",
        { id: "labeled", icon: "check", label: "Ready" },
        "core.group",
        {
          presentation: "icon-label",
          icon: "check",
          label: "Ready",
          inkRole: "@theme.ink.primary",
        },
      ],
    ] as const;

    for (const [componentId, props, capabilityId, expectedProps] of cases) {
      const instanceId = props.id;
      const result = registry.lookup(componentId)!.factory(props, {}, {
        instanceId,
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      });

      expect(result.ok, componentId).toBe(true);
      if (result.ok) {
        expect(result.value.roots[0], componentId).toMatchObject({
          id: instanceId,
          kind: "group",
          capabilityId,
          props: expectedProps,
        });
        const root = result.value.roots[0];
        expect(
          root?.kind === "group"
            ? root.children.filter(
                (child) =>
                  child.sdkOrigin?.generatedRole !== undefined &&
                  (child.capabilityId === "core.rect" ||
                    child.capabilityId === "core.text"),
              )
            : [],
          componentId,
        ).toEqual([]);
        expect(result.value.ports.self, componentId).toEqual({ nodeId: instanceId });
        expect(result.value.actions, componentId).toEqual({
          enter: [instanceId],
          emphasis: [instanceId],
          exit: [instanceId],
        });
      }
    }
  });

  it("registers managed overlay and frame factories with shared layout props", () => {
    const registry = createSdkRegistry();
    const context = {
      instanceId: "managed",
      sourceMap: SOURCE_MAP,
      themeTokens: new Map(),
    };
    const overlayDefinition = registry.lookup("sdk.overlay");
    const frameDefinition = registry.lookup("sdk.frame");
    const managedDefinitions = [
      registry.lookup("sdk.stack"),
      registry.lookup("sdk.grid"),
      registry.lookup("sdk.rail"),
      overlayDefinition,
      frameDefinition,
    ];

    expect(overlayDefinition?.descriptor.capabilityId).toBe("layout.overlay");
    expect(frameDefinition?.descriptor.capabilityId).toBe("layout.frame");
    for (const definition of managedDefinitions) {
      expect(definition?.descriptor.props).toMatchObject({
        padding: { type: "number", required: false },
        align: { type: "string", required: false },
        justify: { type: "string", required: false },
        fixedWidth: { type: "boolean", required: false },
        fixedHeight: { type: "boolean", required: false },
      });
    }

    const stack = registry.lookup("sdk.stack")!.factory(
      {
        id: "managed",
        padding: 6,
        align: "stretch",
        justify: "space-between",
        fixedHeight: true,
      },
      { children: [CHILD] },
      context,
    );

    const overlay = overlayDefinition!.factory(
      {
        id: "managed",
        padding: 8,
        align: "center",
        justify: "end",
        fixedWidth: true,
      },
      { children: [CHILD] },
      context,
    );
    const frame = frameDefinition!.factory(
      {
        id: "managed",
        title: "One worker process",
        detail: "One event loop",
        padding: 14,
        gap: 12,
        fixedWidth: true,
        width: 640,
      },
      { children: [CHILD] },
      context,
    );

    expect(overlay.ok).toBe(true);
    expect(frame.ok).toBe(true);
    expect(stack.ok).toBe(true);
    if (overlay.ok && frame.ok && stack.ok) {
      expect(stack.value.roots[0]?.style).toMatchObject({
        coordinateSpace: "local",
        padding: 6,
        align: "stretch",
        justify: "space-between",
        fixedWidth: false,
        fixedHeight: true,
      });
      expect(overlay.value.roots[0]).toMatchObject({
        capabilityId: "layout.overlay",
        style: {
          coordinateSpace: "local",
          padding: 8,
          align: "center",
          justify: "end",
          fixedWidth: true,
          fixedHeight: false,
        },
      });
      expect(frame.value.roots[0]).toMatchObject({
        capabilityId: "layout.frame",
        geometry: { width: 640 },
        props: {
          title: "One worker process",
          detail: "One event loop",
        },
        style: {
          coordinateSpace: "local",
          padding: 14,
          gap: 12,
          fixedWidth: true,
        },
      });
      expect(overlay.value.ports["child[0]"]).toEqual({ nodeId: "child" });
      expect(frame.value.ports["child[0]"]).toEqual({ nodeId: "child" });
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

  it("emits directed edges by default and preserves explicit opt-out", () => {
    const edge = createSdkRegistry().lookup("sdk.edge")!;
    const expand = (props: Parameters<typeof edge.factory>[0]) =>
      edge.factory(props, {}, {
        instanceId: "edge",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      });

    const path = expand({
      id: "edge",
      mode: "path",
      from: { x: 0, y: 0 },
      to: { x: 100, y: 0 },
      path: "M0 0 L100 0",
    });
    const line = expand({
      id: "edge",
      mode: "line",
      from: { x: 0, y: 0 },
      to: { x: 100, y: 0 },
    });
    const undirected = expand({
      id: "edge",
      from: { x: 0, y: 0 },
      to: { x: 100, y: 0 },
      arrowhead: false,
    });

    expect(path.ok && path.value.roots[0]?.style).toMatchObject({
      markerEnd: "arrow",
      arrowhead: true,
    });
    expect(line.ok && line.value.roots[0]?.style).toMatchObject({
      markerEnd: "arrow",
      arrowhead: true,
    });
    expect(undirected.ok && undirected.value.roots[0]?.style).toMatchObject({
      markerEnd: "none",
      arrowhead: false,
    });
  });

  it("binds signals to edges and rejects mixed motion modes", () => {
    const signal = createSdkRegistry().lookup("sdk.signal")!;
    const expand = (props: Parameters<typeof signal.factory>[0]) =>
      signal.factory(props, {}, {
        instanceId: "motion",
        sourceMap: SOURCE_MAP,
        themeTokens: new Map(),
      });

    const edgeBound = expand({ id: "motion", edge: "request-credit" });
    expect(edgeBound.ok).toBe(true);
    if (edgeBound.ok) {
      expect(edgeBound.value.roots[0]).toMatchObject({
        capabilityId: "motion.signal",
        edgeRef: "request-credit",
      });
      expect(edgeBound.value.roots[0]).not.toHaveProperty("from");
      expect(edgeBound.value.roots[0]).not.toHaveProperty("to");
    }

    const conflictingCases: ReadonlyArray<Readonly<Record<string, JsonValue>>> = [
      { id: "motion", edge: "request-credit", from: { x: 0, y: 0 }, to: { x: 1, y: 1 } },
      { id: "motion", edge: "request-credit", path: "M0 0 L1 1" },
      { id: "motion", edge: "request-credit", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
      { id: "motion", from: { x: 0, y: 0 } },
    ];
    for (const conflicting of conflictingCases) {
      const result = expand(conflicting);
      expect(result.ok).toBe(false);
      expect(!result.ok && result.diagnostics[0]?.code).toBe("SDK_SIGNAL_MODE_CONFLICT");
    }
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

  it("lays rail collection items in a content-sized row", () => {
    const registry = createSdkRegistry();
    const railIds = ["sdk.tagList", "sdk.breadcrumb", "sdk.tabs", "sdk.pagination"] as const;
    const items = ["Alpha", "Beta", "Gamma"];
    const containerWidth = 260;

    for (const componentId of railIds) {
      const result = registry.lookup(componentId)!.factory(
        { id: "rail", items, width: containerWidth, height: 32 },
        {},
        {
          instanceId: componentId.replace(".", "-"),
          sourceMap: SOURCE_MAP,
          themeTokens: new Map(),
        },
      );

      expect(result.ok, componentId).toBe(true);
      if (!result.ok) {
        continue;
      }
      const root = result.value.roots[0];
      expect(root?.capabilityId, componentId).toBe("layout.rail");
      expect(root?.kind === "group" && root.children.length, componentId).toBe(items.length);
      if (root?.kind !== "group") {
        continue;
      }
      for (const child of root.children) {
        expect(child.geometry.y, `${componentId}:${child.id}`).toBe(0);
        // Rail layout sums child widths; full-bleed items overflow 2–3×.
        expect(child.geometry.width, `${componentId}:${child.id}`).toBeLessThan(
          containerWidth / items.length,
        );
      }
      const totalItemWidth = root.children.reduce(
        (sum, child) => sum + child.geometry.width,
        0,
      );
      expect(totalItemWidth, componentId).toBeLessThanOrEqual(containerWidth);
    }
  });

  it("maps sdk.inset gap onto the pad style key layout.pad reads", () => {
    const result = createSdkRegistry().lookup("sdk.inset")!.factory(
      { id: "inset", gap: 18, label: "Inset" },
      { children: [CHILD] },
      { instanceId: "inset", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const style = result.value.roots[0]?.style ?? {};
      const inset =
        typeof style.inset === "number"
          ? style.inset
          : typeof style.pad === "number"
            ? style.pad
            : undefined;
      expect(inset).toBe(18);
    }
  });

  it("defaults sdk.iconLabel to a height floor that fits icon-label presentation chrome", () => {
    const result = createSdkRegistry().lookup("sdk.iconLabel")!.factory(
      { id: "labeled", icon: "check", label: "Ready" },
      {},
      { instanceId: "labeled", sourceMap: SOURCE_MAP, themeTokens: new Map() },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.roots[0]?.geometry.height).toBeGreaterThanOrEqual(40);
    }
  });
});
