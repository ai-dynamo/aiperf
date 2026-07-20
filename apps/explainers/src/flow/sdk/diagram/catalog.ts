/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Systems-diagram SDK component catalog.
//!
//! The catalog shares one labeled-node grammar while preserving category
//! semantics through stable capabilities, icon glyphs, ports, and actions.

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import type {
  ConnectorEndpointIr,
  GeometryIr,
  RenderNodeIr,
} from "../../schema/ir.js";
import type { JsonValue } from "../../schema/json-value.js";
import type { StyleValueIr } from "../../schema/theme.js";
import { attachSdkOrigin } from "../provenance.js";
import type {
  SceneFragment,
  SdkActionName,
  SdkComponentDefinition,
  SdkComponentFactory,
  SdkExpansionContext,
} from "../types.js";

type Props = Readonly<Record<string, JsonValue>>;
type Slots = Readonly<Record<string, readonly SceneFragment[]>>;
type Category = "actor" | "compute" | "storage" | "messaging" | "network" | "control" | "boundary" | "symbol";

type DiagramSpec = Readonly<{
  id: string;
  category: Category;
  glyph: string;
  width?: number;
  height?: number;
}>;

const CHROME_ACTIONS = ["enter", "emphasis", "exit"] as const satisfies readonly SdkActionName[];
const TOPOLOGY_ACTIONS = ["enter", "draw", "trace"] as const satisfies readonly SdkActionName[];
const LAYOUT_ACTIONS = ["enter", "stagger"] as const satisfies readonly SdkActionName[];

const SPECS: readonly DiagramSpec[] = [
  { id: "sdk.user", category: "actor", glyph: "●", width: 112, height: 76 },
  { id: "sdk.client", category: "compute", glyph: "⌘" },
  { id: "sdk.service", category: "compute", glyph: "◆" },
  { id: "sdk.server", category: "compute", glyph: "▤" },
  { id: "sdk.process", category: "compute", glyph: "◌" },
  { id: "sdk.worker", category: "compute", glyph: "⚙" },
  { id: "sdk.function", category: "compute", glyph: "ƒ" },
  { id: "sdk.container", category: "compute", glyph: "⬡" },
  { id: "sdk.cloud", category: "compute", glyph: "☁", width: 132, height: 76 },
  { id: "sdk.database", category: "storage", glyph: "◒" },
  { id: "sdk.dataStore", category: "storage", glyph: "▥" },
  { id: "sdk.cache", category: "storage", glyph: "ϟ" },
  { id: "sdk.file", category: "storage", glyph: "▧" },
  { id: "sdk.objectStore", category: "storage", glyph: "◫" },
  { id: "sdk.volume", category: "storage", glyph: "▱" },
  { id: "sdk.queue", category: "messaging", glyph: "≡" },
  { id: "sdk.topic", category: "messaging", glyph: "◎" },
  { id: "sdk.stream", category: "messaging", glyph: "≈" },
  { id: "sdk.eventBus", category: "messaging", glyph: "↠" },
  { id: "sdk.gateway", category: "network", glyph: "⇥" },
  { id: "sdk.endpoint", category: "network", glyph: "◉" },
  { id: "sdk.loadBalancer", category: "network", glyph: "⑂" },
  { id: "sdk.firewall", category: "network", glyph: "▦" },
  { id: "sdk.start", category: "control", glyph: "▶", width: 84, height: 48 },
  { id: "sdk.end", category: "control", glyph: "■", width: 84, height: 48 },
  { id: "sdk.processStep", category: "control", glyph: "→" },
  { id: "sdk.decision", category: "control", glyph: "◇", width: 112, height: 86 },
  { id: "sdk.merge", category: "control", glyph: "⋈" },
  { id: "sdk.delay", category: "control", glyph: "◷" },
  { id: "sdk.retry", category: "control", glyph: "↻" },
  { id: "sdk.loop", category: "control", glyph: "⟳" },
  { id: "sdk.boundary", category: "boundary", glyph: "□", width: 320, height: 200 },
  { id: "sdk.zone", category: "boundary", glyph: "▢", width: 320, height: 200 },
  { id: "sdk.cluster", category: "boundary", glyph: "⬚", width: 320, height: 200 },
  { id: "sdk.trustBoundary", category: "boundary", glyph: "⛨", width: 320, height: 200 },
  { id: "sdk.document", category: "symbol", glyph: "▧", width: 96, height: 72 },
  { id: "sdk.terminal", category: "symbol", glyph: ">_", width: 112, height: 72 },
  { id: "sdk.clock", category: "symbol", glyph: "◷", width: 96, height: 72 },
  { id: "sdk.lock", category: "symbol", glyph: "▣", width: 96, height: 72 },
  { id: "sdk.key", category: "symbol", glyph: "⚿", width: 96, height: 72 },
  { id: "sdk.warning", category: "symbol", glyph: "⚠", width: 96, height: 72 },
];

const COMMON_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  id: { type: "string", required: true },
  title: { type: "string", required: false },
  label: { type: "string", required: false },
  detail: { type: "string", required: false },
  description: { type: "string", required: false },
  variant: { type: "string", required: false },
  branches: { type: "array", required: false },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false },
  height: { type: "number", required: false },
  surfaceRole: { type: "string", required: false },
  strokeRole: { type: "string", required: false },
  inkRole: { type: "string", required: false },
  position: { type: "object", required: false },
};

const CHILDREN_SLOT: Readonly<Record<string, ComponentSlotDescriptor>> = {
  children: { accepts: "sdk.*", required: false },
};

function numberProp(props: Props, key: string, fallback: number): number {
  const value = props[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringProp(props: Props, key: string): string | undefined {
  const value = props[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function geometry(props: Props, spec: DiagramSpec): GeometryIr {
  return {
    x: numberProp(props, "x", 0),
    y: numberProp(props, "y", 0),
    width: numberProp(props, "width", spec.width ?? 144),
    height: numberProp(props, "height", spec.height ?? 82),
  };
}

function withOrigin<T extends RenderNodeIr>(
  node: T,
  context: SdkExpansionContext,
  componentId: string,
  role: string,
): T {
  return attachSdkOrigin(node, {
    componentId,
    instanceId: context.instanceId,
    sourceMap: context.sourceMap,
    generatedRole: role,
  });
}

function rect(
  id: string,
  geometryValue: GeometryIr,
  style: Readonly<Record<string, StyleValueIr>>,
  label: string,
  context: SdkExpansionContext,
  spec: DiagramSpec,
  role: string,
): RenderNodeIr {
  return withOrigin(
    {
      kind: "rect",
      id,
      capabilityId: "core.rect",
      geometry: geometryValue,
      style,
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
    },
    context,
    spec.id,
    role,
  );
}

function text(
  id: string,
  content: string,
  geometryValue: GeometryIr,
  style: Readonly<Record<string, StyleValueIr>>,
  context: SdkExpansionContext,
  spec: DiagramSpec,
  role: string,
): RenderNodeIr {
  return withOrigin(
    {
      kind: "text",
      id,
      capabilityId: "core.text",
      geometry: geometryValue,
      style,
      accessibility: { label: content },
      fallback: content,
      sourceMap: context.sourceMap,
      text: content,
    },
    context,
    spec.id,
    role,
  );
}

function group(
  spec: DiagramSpec,
  context: SdkExpansionContext,
  geometryValue: GeometryIr,
  label: string,
  children: readonly RenderNodeIr[],
): RenderNodeIr {
  return withOrigin(
    {
      kind: "group",
      id: context.instanceId,
      capabilityId: `diagram.${spec.category}`,
      geometry: geometryValue,
      style: { coordinateSpace: "local" },
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      children,
    },
    context,
    spec.id,
    "root",
  );
}

function themeStyle(props: Props, spec: DiagramSpec): Record<string, StyleValueIr> {
  const surfaceByCategory: Readonly<Record<Category, string>> = {
    actor: "@theme.surface.elevated",
    compute: "@theme.surface.elevated",
    storage: "@theme.accent.secondary",
    messaging: "@theme.accent.tertiary",
    network: "@theme.accent.primary",
    control: "@theme.surface.elevated",
    boundary: "none",
    symbol: "@theme.surface.elevated",
  };
  return {
    fill: stringProp(props, "surfaceRole") ?? surfaceByCategory[spec.category],
    stroke: stringProp(props, "strokeRole") ?? "@theme.ink.secondary",
    strokeWidth: spec.category === "boundary" ? 1.5 : 1.25,
    radius: spec.category === "storage" ? 12 : spec.category === "boundary" ? 8 : 7,
    ...(spec.id === "sdk.trustBoundary" ? { strokeDasharray: "6 4" } : {}),
  };
}

function labelFor(props: Props, spec: DiagramSpec): string {
  return stringProp(props, "title") ?? stringProp(props, "label") ?? spec.id.slice(4);
}

function branches(props: Props): readonly string[] {
  return Array.isArray(props.branches)
    ? props.branches.filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
    : [];
}

function childFragments(slots: Slots): readonly SceneFragment[] {
  return slots.children ?? [];
}

function semanticPorts(
  spec: DiagramSpec,
  context: SdkExpansionContext,
  box: GeometryIr,
  props: Props,
): Record<string, ConnectorEndpointIr> {
  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: context.instanceId },
    input: { nodeId: context.instanceId, anchor: "w" },
    output: { nodeId: context.instanceId, anchor: "e" },
  };
  if (spec.category === "storage") {
    ports.read = { nodeId: context.instanceId, anchor: "w" };
    ports.write = { nodeId: context.instanceId, anchor: "e" };
  }
  if (spec.category === "messaging") {
    ports.producer = { nodeId: context.instanceId, anchor: "w" };
    ports.consumer = { nodeId: context.instanceId, anchor: "e" };
  }
  if (spec.category === "network") {
    ports.inbound = { nodeId: context.instanceId, anchor: "w" };
    ports.outbound = { nodeId: context.instanceId, anchor: "e" };
  }
  if (spec.category === "boundary") {
    ports.entry = { nodeId: context.instanceId, anchor: "w" };
    ports.exit = { nodeId: context.instanceId, anchor: "e" };
  }
  if (spec.id === "sdk.decision") {
    const authoredBranches = branches(props);
    const names = authoredBranches.length > 0 ? authoredBranches : ["yes", "no"];
    names.forEach((name, index) => {
      const anchor = index % 2 === 0 ? "s" : "e";
      ports[`branch[${index}]`] = { nodeId: context.instanceId, anchor };
      ports[`branch.${name}`] = { nodeId: context.instanceId, anchor };
    });
  }
  if (spec.id === "sdk.retry" || spec.id === "sdk.loop") {
    ports.back = {
      nodeId: context.instanceId,
      x: box.x,
      y: box.y + box.height,
      anchor: "s",
    };
  }
  return ports;
}

function standardFactory(spec: DiagramSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const label = labelFor(props, spec);
    const detail = stringProp(props, "detail");
    const chromeId = `${context.instanceId}__chrome`;
    const glyphId = `${context.instanceId}__glyph`;
    const titleId = `${context.instanceId}__title`;
    const children: RenderNodeIr[] = [
      rect(
        chromeId,
        { x: 0, y: 0, width: box.width, height: box.height },
        themeStyle(props, spec),
        label,
        context,
        spec,
        "chrome",
      ),
      text(
        glyphId,
        spec.glyph,
        { x: 10, y: 10, width: 30, height: 28 },
        {
          fill: stringProp(props, "inkRole") ?? "@theme.ink.primary",
          fontSize: spec.glyph.length > 1 ? 14 : 20,
          fontWeight: "bold",
          textAnchor: "middle",
        },
        context,
        spec,
        "glyph",
      ),
      text(
        titleId,
        label,
        { x: 46, y: detail === undefined ? 20 : 12, width: Math.max(box.width - 56, 0), height: 22 },
        {
          fill: stringProp(props, "inkRole") ?? "@theme.ink.primary",
          fontSize: 13,
          fontWeight: "bold",
          textAnchor: "start",
        },
        context,
        spec,
        "title",
      ),
    ];
    if (detail !== undefined) {
      children.push(
        text(
          `${context.instanceId}__detail`,
          detail,
          { x: 46, y: 38, width: Math.max(box.width - 56, 0), height: 18 },
          { fill: "@theme.ink.secondary", fontSize: 10, textAnchor: "start" },
          context,
          spec,
          "detail",
        ),
      );
    }
    if (spec.id === "sdk.retry" || spec.id === "sdk.loop") {
      children.push(
        withOrigin(
          {
            kind: "connector",
            id: `${context.instanceId}__back-edge`,
            capabilityId: "core.connector",
            geometry: { x: 8, y: box.height - 20, width: box.width - 16, height: 12 },
            style: { fill: "none", markerEnd: "arrow", route: "curve" },
            accessibility: { label: `${label} back edge` },
            fallback: `${label} back edge`,
            sourceMap: context.sourceMap,
            from: { x: box.width - 12, y: box.height - 14 },
            to: { x: 12, y: box.height - 14 },
          },
          context,
          spec.id,
          "back-edge",
        ),
      );
    }
    const childRoots = childFragments(slots).flatMap((fragment) => fragment.roots);
    children.push(...childRoots);
    const root = group(spec, context, box, label, children);
    const ports = semanticPorts(spec, context, box, props);
    ports.icon = { nodeId: glyphId };
    ports.title = { nodeId: titleId };
    return {
      ok: true,
      value: {
        roots: [root],
        ports,
        actions:
          spec.category === "control" || spec.category === "messaging" || spec.category === "network"
            ? { enter: [root.id], draw: [chromeId], trace: [root.id] }
            : { enter: [root.id], emphasis: [root.id], exit: [root.id] },
      },
      diagnostics: [],
    };
  };
}

function boundaryFactory(spec: DiagramSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const label = labelFor(props, spec);
    const fragments = childFragments(slots);
    const content = fragments.flatMap((fragment) => fragment.roots);
    const chrome = rect(
      `${context.instanceId}__chrome`,
      { x: 0, y: 0, width: box.width, height: box.height },
      themeStyle(props, spec),
      label,
      context,
      spec,
      "chrome",
    );
    const title = text(
      `${context.instanceId}__title`,
      label,
      { x: 12, y: 8, width: Math.max(box.width - 24, 0), height: 20 },
      { fill: stringProp(props, "inkRole") ?? "@theme.ink.primary", fontSize: 12, fontWeight: "bold", textAnchor: "start" },
      context,
      spec,
      "title",
    );
    const root = group(spec, context, box, label, [chrome, title, ...content]);
    const ports = semanticPorts(spec, context, box, props);
    fragments.forEach((fragment, index) => {
      const child = fragment.roots[0];
      if (child !== undefined) {
        ports[`child[${index}]`] = { nodeId: child.id };
      }
      for (const [portName, endpoint] of Object.entries(fragment.ports)) {
        ports[`child[${index}].${portName}`] = endpoint;
      }
    });
    return {
      ok: true,
      value: {
        roots: [root],
        ports,
        actions: { enter: [root.id], stagger: content.map((child) => child.id) },
      },
      diagnostics: [],
    };
  };
}

function actionsFor(category: Category): readonly SdkActionName[] {
  if (category === "boundary") {
    return LAYOUT_ACTIONS;
  }
  if (category === "control" || category === "messaging" || category === "network") {
    return TOPOLOGY_ACTIONS;
  }
  return CHROME_ACTIONS;
}

function descriptor(spec: DiagramSpec): ComponentDescriptor {
  const segment = spec.id.slice(4);
  const propKeys = [
    "id",
    "title",
    "label",
    "detail",
    "description",
    "x",
    "y",
    "width",
    "height",
    "surfaceRole",
    "strokeRole",
    "inkRole",
    "position",
    ...(spec.id === "sdk.decision" ? ["branches"] : []),
  ] as const;
  return {
    id: spec.id,
    symbolExport: segment.charAt(0).toUpperCase() + segment.slice(1),
    version: "1.0.0",
    classification: "flow-only",
    props: Object.fromEntries(propKeys.map((key) => [key, COMMON_PROPS[key]!])) as Readonly<
      Record<string, ComponentPropDescriptor>
    >,
    slots: spec.category === "boundary" ? CHILDREN_SLOT : {},
    events: [],
    capabilityId: `diagram.${spec.category}`,
    deterministic: true,
  };
}

/** Exhaustive generic systems-diagram SDK catalog. */
export const DIAGRAM_SDK_COMPONENTS: readonly SdkComponentDefinition[] = SPECS.map((spec) => ({
  descriptor: descriptor(spec),
  factory: spec.category === "boundary" ? boundaryFactory(spec) : standardFactory(spec),
  actions: actionsFor(spec.category),
}));
