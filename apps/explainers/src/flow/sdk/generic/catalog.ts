/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Declarative generic SDK component catalog.
//!
//! These factories intentionally lower into the small existing Scene IR
//! vocabulary. Repeated visual grammar lives in family factories rather than
//! one renderer branch per authored component.

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
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
type Family =
  | "shape"
  | "text"
  | "icon"
  | "image"
  | "connector"
  | "container"
  | "collection"
  | "indicator";

type CatalogSpec = Readonly<{
  id: string;
  capabilityId: string;
  family: Family;
  width: number;
  height: number;
  actions: readonly SdkActionName[];
  slots?: Readonly<Record<string, ComponentSlotDescriptor>>;
}>;

const CHROME_ACTIONS = ["enter", "emphasis", "exit"] as const satisfies readonly SdkActionName[];
const LAYOUT_ACTIONS = ["enter", "stagger"] as const satisfies readonly SdkActionName[];
const TOPOLOGY_ACTIONS = ["enter", "draw", "trace"] as const satisfies readonly SdkActionName[];
const INDICATOR_ACTIONS = [
  "enter",
  "emphasis",
  "pulse",
  "exit",
] as const satisfies readonly SdkActionName[];

const CHILDREN_SLOT = { children: { accepts: "sdk.*", required: false } } as const;
const COLLECTION_SLOTS = {
  children: { accepts: "sdk.*", required: false },
  items: { accepts: "sdk.*", required: false },
  rows: { accepts: "sdk.*", required: false },
  cells: { accepts: "sdk.*", required: false },
} as const;

const COMMON_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  id: { type: "string", required: true },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false },
  height: { type: "number", required: false },
  text: { type: "string", required: false },
  title: { type: "string", required: false },
  label: { type: "string", required: false },
  detail: { type: "string", required: false },
  description: { type: "string", required: false },
  variant: { type: "string", required: false },
  icon: { type: "string", required: false },
  src: { type: "string", required: false },
  value: { type: "number", required: false },
  min: { type: "number", required: false, default: 0 },
  max: { type: "number", required: false, default: 1 },
  clamp: { type: "boolean", required: false, default: false },
  items: { type: "array", required: false },
  entries: { type: "array", required: false },
  values: { type: "array", required: false },
  columns: { type: "array", required: false },
  path: { type: "string", required: false },
  from: { type: "endpoint", required: false },
  to: { type: "endpoint", required: false },
  surfaceRole: { type: "string", required: false },
  strokeRole: { type: "string", required: false },
  inkRole: { type: "string", required: false },
  direction: { type: "string", required: false },
  gap: { type: "number", required: false },
  position: { type: "object", required: false },
};

const CATALOG: readonly CatalogSpec[] = [
  { id: "sdk.shape", capabilityId: "core.rect", family: "shape", width: 120, height: 64, actions: CHROME_ACTIONS },
  { id: "sdk.text", capabilityId: "core.text", family: "text", width: 180, height: 24, actions: CHROME_ACTIONS },
  { id: "sdk.richText", capabilityId: "core.text", family: "text", width: 240, height: 72, actions: CHROME_ACTIONS },
  { id: "sdk.icon", capabilityId: "core.path", family: "icon", width: 24, height: 24, actions: CHROME_ACTIONS },
  { id: "sdk.image", capabilityId: "core.image", family: "image", width: 160, height: 90, actions: CHROME_ACTIONS },
  { id: "sdk.line", capabilityId: "core.line", family: "connector", width: 120, height: 0, actions: TOPOLOGY_ACTIONS },
  { id: "sdk.arrow", capabilityId: "core.arrow", family: "connector", width: 120, height: 0, actions: TOPOLOGY_ACTIONS },
  { id: "sdk.spacer", capabilityId: "core.group", family: "container", width: 24, height: 24, actions: LAYOUT_ACTIONS },
  { id: "sdk.inset", capabilityId: "layout.pad", family: "container", width: 180, height: 80, actions: LAYOUT_ACTIONS, slots: CHILDREN_SLOT },
  { id: "sdk.title", capabilityId: "core.text", family: "text", width: 280, height: 32, actions: CHROME_ACTIONS },
  { id: "sdk.paragraph", capabilityId: "core.text", family: "text", width: 280, height: 72, actions: CHROME_ACTIONS },
  { id: "sdk.caption", capabilityId: "core.text", family: "text", width: 180, height: 18, actions: CHROME_ACTIONS },
  { id: "sdk.codeBlock", capabilityId: "core.group", family: "text", width: 320, height: 140, actions: CHROME_ACTIONS },
  { id: "sdk.quote", capabilityId: "core.group", family: "text", width: 280, height: 88, actions: CHROME_ACTIONS },
  { id: "sdk.list", capabilityId: "core.group", family: "collection", width: 240, height: 96, actions: LAYOUT_ACTIONS, slots: COLLECTION_SLOTS },
  { id: "sdk.keyValue", capabilityId: "core.panel", family: "collection", width: 220, height: 32, actions: CHROME_ACTIONS },
  { id: "sdk.propertyList", capabilityId: "core.panel", family: "collection", width: 260, height: 120, actions: LAYOUT_ACTIONS, slots: COLLECTION_SLOTS },
  { id: "sdk.badge", capabilityId: "core.chip", family: "shape", width: 88, height: 26, actions: CHROME_ACTIONS },
  { id: "sdk.statusDot", capabilityId: "core.circle", family: "shape", width: 14, height: 14, actions: CHROME_ACTIONS },
  { id: "sdk.avatar", capabilityId: "core.group", family: "icon", width: 48, height: 48, actions: CHROME_ACTIONS },
  { id: "sdk.iconLabel", capabilityId: "core.group", family: "icon", width: 160, height: 40, actions: CHROME_ACTIONS },
  { id: "sdk.alert", capabilityId: "core.panel", family: "shape", width: 280, height: 72, actions: CHROME_ACTIONS },
  { id: "sdk.statusCard", capabilityId: "core.panel", family: "shape", width: 220, height: 88, actions: CHROME_ACTIONS },
  { id: "sdk.emptyState", capabilityId: "core.panel", family: "icon", width: 260, height: 140, actions: CHROME_ACTIONS },
  { id: "sdk.stat", capabilityId: "core.panel", family: "collection", width: 150, height: 72, actions: CHROME_ACTIONS },
  { id: "sdk.metric", capabilityId: "core.panel", family: "collection", width: 190, height: 80, actions: CHROME_ACTIONS },
  { id: "sdk.table", capabilityId: "core.panel", family: "collection", width: 420, height: 180, actions: LAYOUT_ACTIONS, slots: COLLECTION_SLOTS },
  { id: "sdk.tableRow", capabilityId: "layout.rail", family: "container", width: 420, height: 32, actions: LAYOUT_ACTIONS, slots: COLLECTION_SLOTS },
  { id: "sdk.tableCell", capabilityId: "core.group", family: "container", width: 120, height: 32, actions: CHROME_ACTIONS, slots: CHILDREN_SLOT },
  { id: "sdk.tagList", capabilityId: "core.lane", family: "collection", width: 260, height: 32, actions: LAYOUT_ACTIONS, slots: COLLECTION_SLOTS },
  { id: "sdk.breadcrumb", capabilityId: "core.lane", family: "collection", width: 300, height: 28, actions: LAYOUT_ACTIONS },
  { id: "sdk.tabs", capabilityId: "core.lane", family: "collection", width: 300, height: 34, actions: LAYOUT_ACTIONS },
  { id: "sdk.pagination", capabilityId: "core.lane", family: "collection", width: 220, height: 30, actions: LAYOUT_ACTIONS },
  { id: "sdk.timeline", capabilityId: "core.stepper", family: "collection", width: 300, height: 160, actions: LAYOUT_ACTIONS, slots: COLLECTION_SLOTS },
  { id: "sdk.timelineItem", capabilityId: "core.panel", family: "collection", width: 240, height: 48, actions: CHROME_ACTIONS, slots: CHILDREN_SLOT },
  { id: "sdk.progress", capabilityId: "core.group", family: "indicator", width: 220, height: 18, actions: INDICATOR_ACTIONS },
  { id: "sdk.meter", capabilityId: "core.group", family: "indicator", width: 220, height: 22, actions: INDICATOR_ACTIONS },
  { id: "sdk.gauge", capabilityId: "core.group", family: "indicator", width: 120, height: 72, actions: INDICATOR_ACTIONS },
  { id: "sdk.sparkline", capabilityId: "core.group", family: "indicator", width: 160, height: 48, actions: INDICATOR_ACTIONS },
  { id: "sdk.rating", capabilityId: "layout.rail", family: "indicator", width: 120, height: 24, actions: CHROME_ACTIONS },
  { id: "sdk.semaphore", capabilityId: "layout.rail", family: "indicator", width: 72, height: 24, actions: INDICATOR_ACTIONS },
  { id: "sdk.section", capabilityId: "core.panel", family: "container", width: 360, height: 220, actions: LAYOUT_ACTIONS, slots: CHILDREN_SLOT },
  { id: "sdk.toolbar", capabilityId: "layout.rail", family: "container", width: 360, height: 44, actions: LAYOUT_ACTIONS, slots: CHILDREN_SLOT },
  { id: "sdk.splitPane", capabilityId: "layout.stack", family: "container", width: 480, height: 240, actions: LAYOUT_ACTIONS, slots: CHILDREN_SLOT },
  { id: "sdk.mediaObject", capabilityId: "core.group", family: "container", width: 320, height: 96, actions: LAYOUT_ACTIONS, slots: {
    media: { accepts: "sdk.*", required: false },
    body: { accepts: "sdk.*", required: false },
    leading: { accepts: "sdk.*", required: false },
    trailing: { accepts: "sdk.*", required: false },
    children: { accepts: "sdk.*", required: false },
  } },
];

const ICON_PATHS: Readonly<Record<string, string>> = {
  check: "M4 12 L9 17 L20 5",
  warning: "M12 2 L22 21 H2 Z M12 8 V14 M12 18 V18.5",
  user: "M12 3 A4 4 0 1 1 12 11 A4 4 0 1 1 12 3 M4 22 C4 16 8 13 12 13 C16 13 20 16 20 22",
  server: "M3 4 H21 V10 H3 Z M3 14 H21 V20 H3 Z M6 7 H7 M6 17 H7",
  database: "M4 6 C4 2 20 2 20 6 V18 C20 22 4 22 4 18 Z M4 6 C4 10 20 10 20 6 M4 12 C4 16 20 16 20 12",
  cloud: "M7 19 H18 A4 4 0 0 0 18 11 A6 6 0 0 0 6.5 9 A5 5 0 0 0 7 19",
  arrow: "M3 12 H20 M14 6 L20 12 L14 18",
  code: "M9 6 L3 12 L9 18 M15 6 L21 12 L15 18",
  file: "M5 2 H15 L20 7 V22 H5 Z M15 2 V7 H20",
  lock: "M6 10 H18 V22 H6 Z M8 10 V7 A4 4 0 0 1 16 7 V10",
};

function numberProp(props: Props, key: string, fallback: number): number {
  const value = props[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringProp(props: Props, key: string): string | undefined {
  const value = props[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function geometry(props: Props, spec: CatalogSpec): GeometryIr {
  return {
    x: numberProp(props, "x", 0),
    y: numberProp(props, "y", 0),
    width: numberProp(props, "width", spec.width),
    height: numberProp(props, "height", spec.height),
  };
}

function origin<T extends RenderNodeIr>(
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

function textNode(
  id: string,
  text: string,
  geometryValue: GeometryIr,
  context: SdkExpansionContext,
  componentId: string,
  role: string,
  style: Readonly<Record<string, StyleValueIr>> = {},
): RenderNodeIr {
  return origin(
    {
      kind: "text",
      id,
      capabilityId: "core.text",
      geometry: geometryValue,
      style,
      accessibility: { label: text },
      fallback: text,
      sourceMap: context.sourceMap,
      text,
    },
    context,
    componentId,
    role,
  );
}

function rectNode(
  id: string,
  geometryValue: GeometryIr,
  context: SdkExpansionContext,
  componentId: string,
  role: string,
  label: string,
  style: Readonly<Record<string, StyleValueIr>>,
  capabilityId = "core.rect",
): RenderNodeIr {
  return origin(
    {
      kind: "rect",
      id,
      capabilityId,
      geometry: geometryValue,
      style,
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
    },
    context,
    componentId,
    role,
  );
}

function groupNode(
  spec: CatalogSpec,
  context: SdkExpansionContext,
  geometryValue: GeometryIr,
  label: string,
  children: readonly RenderNodeIr[],
  style: Readonly<Record<string, StyleValueIr>> = { coordinateSpace: "local" },
  props?: Readonly<Record<string, JsonValue>>,
): RenderNodeIr {
  return origin(
    {
      kind: "group",
      id: context.instanceId,
      capabilityId: spec.capabilityId,
      geometry: geometryValue,
      style,
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      ...(props !== undefined ? { props } : {}),
      children,
    },
    context,
    spec.id,
    "root",
  );
}

function success(
  roots: readonly RenderNodeIr[],
  ports: Readonly<Record<string, ConnectorEndpointIr>>,
  actions: Readonly<Partial<Record<SdkActionName, readonly string[]>>>,
): Result<SceneFragment> {
  return { ok: true, value: { roots, ports, actions }, diagnostics: [] };
}

function actionsFor(
  spec: CatalogSpec,
  rootId: string,
  childIds: readonly string[] = [],
): Partial<Record<SdkActionName, readonly string[]>> {
  const actions: Partial<Record<SdkActionName, readonly string[]>> = {};
  for (const action of spec.actions) {
    actions[action] =
      action === "stagger"
        ? childIds
        : action === "pulse"
          ? [childIds.at(-1) ?? rootId]
          : [rootId];
  }
  return actions;
}

function fail(
  context: SdkExpansionContext,
  code: string,
  message: string,
  repair: string,
): Result<SceneFragment> {
  return {
    ok: false,
    diagnostics: [diagnostic(code, "error", message, context.sourceMap, repair)],
  };
}

function childFragments(slots: Slots): readonly SceneFragment[] {
  for (const key of ["children", "items", "rows", "cells", "body"]) {
    const entries = slots[key];
    if (entries !== undefined && entries.length > 0) {
      return entries;
    }
  }
  return [];
}

function rootsOf(fragments: readonly SceneFragment[]): readonly RenderNodeIr[] {
  return fragments.flatMap((fragment) => fragment.roots);
}

function variantStyle(props: Props): Record<string, StyleValueIr> {
  const variant = stringProp(props, "variant") ?? "neutral";
  const variantRole: Readonly<Record<string, string>> = {
    neutral: "@theme.surface.elevated",
    info: "@theme.accent.primary",
    success: "@theme.accent.green",
    warning: "@theme.accent.tertiary",
    danger: "@theme.accent.danger",
  };
  return {
    fill: stringProp(props, "surfaceRole") ?? variantRole[variant] ?? variantRole.neutral!,
    stroke: stringProp(props, "strokeRole") ?? "@theme.ink.secondary",
  };
}

function visibleText(props: Props, fallback: string): string {
  return (
    stringProp(props, "text") ??
    stringProp(props, "title") ??
    stringProp(props, "label") ??
    fallback
  );
}

function stringItems(props: Props): readonly string[] {
  const value = props.items;
  if (Array.isArray(value)) {
    return value.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
  }
  const entries = props.entries;
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.flatMap((entry) => {
    if (typeof entry === "string") {
      return [entry];
    }
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      return [];
    }
    const record = entry as Record<string, JsonValue>;
    const label = typeof record.label === "string" ? record.label : undefined;
    const key = typeof record.key === "string" ? record.key : undefined;
    const valueText =
      typeof record.value === "string" || typeof record.value === "number"
        ? String(record.value)
        : undefined;
    return [label ?? [key, valueText].filter(Boolean).join(": ")].filter(Boolean) as string[];
  });
}

function endpoint(value: JsonValue | undefined, fallback: ConnectorEndpointIr): ConnectorEndpointIr {
  if (typeof value === "string" && value.length > 0) {
    return { nodeId: value };
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return fallback;
  }
  const record = value as Record<string, JsonValue>;
  const nodeId = typeof record.nodeId === "string" ? record.nodeId : undefined;
  const x = typeof record.x === "number" && Number.isFinite(record.x) ? record.x : undefined;
  const y = typeof record.y === "number" && Number.isFinite(record.y) ? record.y : undefined;
  const anchor = typeof record.anchor === "string" ? record.anchor : undefined;
  return {
    ...(nodeId !== undefined ? { nodeId } : {}),
    ...(x !== undefined ? { x } : {}),
    ...(y !== undefined ? { y } : {}),
    ...(anchor !== undefined ? { anchor } : {}),
  };
}

function shapeFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, _slots, context) => {
    const box = geometry(props, spec);
    const label = visibleText(props, spec.id.slice(4));
    const shapeVariant = stringProp(props, "variant") ?? "rect";
    if (
      spec.id === "sdk.shape" &&
      !["rect", "rounded", "circle", "ellipse", "path"].includes(shapeVariant)
    ) {
      return fail(
        context,
        "SDK_SHAPE_VARIANT_INVALID",
        `sdk.shape "${context.instanceId}" has unknown variant "${shapeVariant}".`,
        "Use rect, rounded, circle, ellipse, or path.",
      );
    }
    if (spec.id === "sdk.shape" && shapeVariant === "path") {
      const path = stringProp(props, "path");
      if (path === undefined) {
        return fail(
          context,
          "SDK_SHAPE_PATH_REQUIRED",
          `sdk.shape "${context.instanceId}" with variant "path" requires path data.`,
          'Provide path = "M…".',
        );
      }
      const root: RenderNodeIr = origin(
        {
          kind: "connector",
          id: context.instanceId,
          capabilityId: "core.path",
          geometry: box,
          style: { ...variantStyle(props), fill: "none", markerEnd: "none" },
          accessibility: { label },
          fallback: label,
          sourceMap: context.sourceMap,
          from: { x: box.x, y: box.y },
          to: { x: box.x + box.width, y: box.y + box.height },
          path,
        },
        context,
        spec.id,
        "root",
      );
      return success([root], { self: { nodeId: root.id } }, {
        enter: [root.id],
        emphasis: [root.id],
        exit: [root.id],
      });
    }
    const capabilityId =
      spec.id !== "sdk.shape"
        ? spec.capabilityId
        : shapeVariant === "circle"
          ? "core.circle"
          : shapeVariant === "ellipse"
            ? "core.ellipse"
            : "core.rect";
    const style = {
      ...variantStyle(props),
      radius:
        spec.id === "sdk.statusDot" || shapeVariant === "circle"
          ? Math.max(box.width, box.height)
          : spec.id === "sdk.badge" || shapeVariant === "rounded"
            ? 10
            : 0,
    };
    if (
      spec.id === "sdk.badge" ||
      spec.id === "sdk.alert" ||
      spec.id === "sdk.statusCard"
    ) {
      const semanticProps: Record<string, JsonValue> =
        spec.id === "sdk.badge"
          ? { label }
          : (() => {
              const record: Record<string, JsonValue> = { title: label };
              const detail = stringProp(props, "detail");
              if (detail !== undefined) {
                record.detail = detail;
              }
              return record;
            })();
      const root = groupNode(
        spec,
        context,
        box,
        label,
        [],
        { ...style, coordinateSpace: "local" },
        semanticProps,
      );
      return success([root], { self: { nodeId: root.id } }, {
        enter: [root.id],
        emphasis: [root.id],
        exit: [root.id],
      });
    }
    const chrome = rectNode(
      context.instanceId,
      box,
      context,
      spec.id,
      "root",
      label,
      style,
      capabilityId,
    );
    return success([chrome], { self: { nodeId: chrome.id } }, {
      enter: [chrome.id],
      emphasis: [chrome.id],
      exit: [chrome.id],
    });
  };
}

function textFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, _slots, context) => {
    const box = geometry(props, spec);
    const text = visibleText(props, spec.id.slice(4));
    const isCode = spec.id === "sdk.codeBlock";
    const isQuote = spec.id === "sdk.quote";
    const inkRole = stringProp(props, "inkRole") ?? "@theme.ink.primary";
    if (!isCode && !isQuote) {
      const node = textNode(context.instanceId, text, box, context, spec.id, "root", {
        fontSize: spec.id === "sdk.title" ? 22 : spec.id === "sdk.caption" ? 10 : 12,
        fontWeight: spec.id === "sdk.title" ? "bold" : "normal",
        fontFamily: "inherit",
        fontStyle: "normal",
        textAnchor: "start",
        fill: inkRole,
        whiteSpace: "normal",
      });
      return success([node], { self: { nodeId: node.id }, text: { nodeId: node.id } }, {
        enter: [node.id],
        emphasis: [node.id],
        exit: [node.id],
      });
    }
    const root = groupNode(
      spec,
      context,
      box,
      text,
      [],
      {
        ...variantStyle(props),
        radius: 6,
        coordinateSpace: "local",
        ...(isQuote ? { strokeWidth: 0, borderLeftWidth: 4 } : {}),
      },
      {
        text,
        presentation: isCode ? "code-block" : "quote",
        inkRole,
      },
    );
    return success([root], { self: { nodeId: root.id }, text: { nodeId: root.id } }, {
      enter: [root.id],
      emphasis: [root.id],
      exit: [root.id],
    });
  };
}

function iconFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, _slots, context) => {
    const box = geometry(props, spec);
    const iconName = stringProp(props, "icon") ?? (spec.id === "sdk.avatar" ? "user" : "check");
    const path = ICON_PATHS[iconName];
    if (path === undefined) {
      return fail(
        context,
        "SDK_ICON_UNKNOWN",
        `${spec.id} "${context.instanceId}" references unknown icon "${iconName}".`,
        `Use one of: ${Object.keys(ICON_PATHS).join(", ")}.`,
      );
    }
    const iconId = spec.id === "sdk.icon" ? context.instanceId : `${context.instanceId}__icon`;
    const icon: RenderNodeIr = origin(
      {
        kind: "connector",
        id: iconId,
        capabilityId: "core.path",
        geometry: spec.id === "sdk.icon"
          ? box
          : {
              x: 8,
              y: 8,
              width: Math.max(0, Math.min(24, box.width - 16)),
              height: Math.max(0, Math.min(24, box.height - 16)),
            },
        style: {
          fill: "none",
          stroke: stringProp(props, "inkRole") ?? "@theme.ink.primary",
          strokeWidth: 1.75,
          markerEnd: "none",
        },
        accessibility: { label: iconName },
        fallback: iconName,
        sourceMap: context.sourceMap,
        from: { x: 0, y: 0 },
        to: { x: 24, y: 24 },
        path,
      },
      context,
      spec.id,
      spec.id === "sdk.icon" ? "root" : "icon",
    );
    if (spec.id === "sdk.icon") {
      return success([icon], { self: { nodeId: icon.id }, icon: { nodeId: icon.id } }, {
        enter: [icon.id],
        emphasis: [icon.id],
        exit: [icon.id],
      });
    }
    const label = visibleText(props, spec.id.slice(4));
    const inkRole = stringProp(props, "inkRole") ?? "@theme.ink.primary";
    if (spec.id === "sdk.emptyState") {
      const detail = stringProp(props, "detail");
      const root = groupNode(
        spec,
        context,
        box,
        label,
        [icon],
        { ...variantStyle(props), radius: 8, coordinateSpace: "local" },
        {
          title: label,
          ...(detail !== undefined ? { detail } : {}),
          inkRole,
        },
      );
      return success([root], { self: { nodeId: root.id }, icon: { nodeId: iconId } }, {
        enter: [root.id],
        emphasis: [root.id],
        exit: [root.id],
      });
    }
    const isAvatar = spec.id === "sdk.avatar";
    const root = groupNode(
      spec,
      context,
      box,
      label,
      [icon],
      {
        ...variantStyle(props),
        radius: isAvatar ? box.width : 8,
        coordinateSpace: "local",
      },
      {
        presentation: isAvatar ? "avatar" : "icon-label",
        icon: iconName,
        inkRole,
        ...(isAvatar ? {} : { label }),
      },
    );
    return success([root], { self: { nodeId: root.id }, icon: { nodeId: iconId } }, {
      enter: [root.id],
      emphasis: [root.id],
      exit: [root.id],
    });
  };
}

function imageFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, _slots, context) => {
    const src = stringProp(props, "src");
    if (src === undefined) {
      return fail(
        context,
        "SDK_IMAGE_SRC_REQUIRED",
        `sdk.image "${context.instanceId}" requires a non-empty "src".`,
        "Provide a package-resolvable image source.",
      );
    }
    const box = geometry(props, spec);
    const root: RenderNodeIr = origin(
      {
        kind: "component",
        id: context.instanceId,
        capabilityId: "core.image",
        geometry: box,
        style: { overflow: "hidden" },
        accessibility: { label: stringProp(props, "label") ?? src },
        fallback: src,
        sourceMap: context.sourceMap,
        props: { src, fit: stringProp(props, "variant") ?? "contain" },
        children: [],
      },
      context,
      spec.id,
      "root",
    );
    return success([root], { self: { nodeId: root.id }, image: { nodeId: root.id } }, {
      enter: [root.id],
      emphasis: [root.id],
      exit: [root.id],
    });
  };
}

function connectorFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, _slots, context) => {
    const box = geometry(props, spec);
    const from = endpoint(props.from, { x: box.x, y: box.y });
    const to = endpoint(props.to, { x: box.x + box.width, y: box.y + box.height });
    const root: RenderNodeIr = origin(
      {
        kind: "connector",
        id: context.instanceId,
        capabilityId: spec.capabilityId,
        geometry: box,
        style: {
          fill: "none",
          stroke: stringProp(props, "strokeRole") ?? "@theme.ink.secondary",
          markerEnd: spec.id === "sdk.arrow" ? "arrow" : "none",
        },
        accessibility: { label: visibleText(props, spec.id.slice(4)) },
        fallback: visibleText(props, spec.id.slice(4)),
        sourceMap: context.sourceMap,
        from,
        to,
      },
      context,
      spec.id,
      "root",
    );
    return success([root], {
      self: { nodeId: root.id },
      start: from,
      end: to,
    }, { enter: [root.id], draw: [root.id], trace: [root.id] });
  };
}

function containerFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const fragments = childFragments(slots);
    const label = visibleText(props, spec.id.slice(4));
    const title = stringProp(props, "title") ?? label;
    const detail = stringProp(props, "detail");
    const suppliedRoots = rootsOf(fragments);
    const childRoots =
      suppliedRoots.length > 0 || spec.id === "sdk.spacer"
        ? suppliedRoots
        : spec.id === "sdk.section" || spec.id === "sdk.splitPane"
          ? []
          : [
              textNode(
                `${context.instanceId}__label`,
                label,
                { x: 6, y: 6, width: Math.max(box.width - 12, 0), height: Math.max(box.height - 12, 0) },
                context,
                spec.id,
                "label",
                { fontSize: 10, textAnchor: "middle", fill: "@theme.ink.secondary" },
              ),
            ];
    const direction =
      stringProp(props, "direction") ??
      (spec.id === "sdk.section"
        ? "column"
        : spec.id === "sdk.splitPane"
          ? "row"
          : "row");
    const semanticProps: Record<string, JsonValue> | undefined =
      spec.id === "sdk.section"
        ? {
            title,
            ...(detail !== undefined ? { detail } : {}),
          }
        : spec.id === "sdk.splitPane"
          ? {
              ...(stringProp(props, "title") !== undefined
                ? { title: stringProp(props, "title")! }
                : stringProp(props, "label") !== undefined
                  ? { title: stringProp(props, "label")! }
                  : {}),
              ...(detail !== undefined ? { detail } : {}),
            }
          : undefined;
    // Section keeps a managed column stack so slotted children lay out under
    // panel chrome; splitPane uses layout.stack with a row (or authored) stack.
    const managedChildren =
      spec.id === "sdk.section" && childRoots.length > 0
        ? [
            origin(
              {
                kind: "group",
                id: `${context.instanceId}__stack`,
                capabilityId: "layout.stack",
                geometry: {
                  x: 8,
                  y: 36,
                  width: Math.max(box.width - 16, 0),
                  height: Math.max(box.height - 44, 0),
                },
                style: {
                  coordinateSpace: "local",
                  direction: "column",
                  gap: numberProp(props, "gap", 8),
                },
                accessibility: { label: `${title} content` },
                fallback: title,
                sourceMap: context.sourceMap,
                children: childRoots,
              },
              context,
              spec.id,
              "stack",
            ),
          ]
        : childRoots;
    const root = groupNode(spec, context, box, title, managedChildren, {
      coordinateSpace: "local",
      direction,
      gap: numberProp(props, "gap", 8),
      // layout.pad reads style.inset / style.pad (never style.padding).
      ...(spec.id === "sdk.inset" ? { inset: numberProp(props, "gap", 12) } : {}),
      ...(spec.id === "sdk.section" || spec.id === "sdk.tableCell"
        ? { ...variantStyle(props), radius: 6 }
        : {}),
    }, semanticProps);
    const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: root.id } };
    fragments.forEach((fragment, index) => {
      const child = fragment.roots[0];
      if (child !== undefined) {
        ports[`child[${index}]`] = { nodeId: child.id };
      }
      for (const [name, endpointValue] of Object.entries(fragment.ports)) {
        ports[`child[${index}].${name}`] = endpointValue;
      }
    });
    return success(
      [root],
      ports,
      actionsFor(spec, root.id, childRoots.map((child) => child.id)),
    );
  };
}

function mediaObjectFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const roles = ["leading", "media", "body", "trailing", "children"] as const;
    const ordered = roles.flatMap((role) =>
      (slots[role] ?? []).map((fragment) => ({ role, fragment })),
    );
    const children = ordered.flatMap(({ fragment }) => fragment.roots);
    const label = visibleText(props, "media object");
    const visibleChildren =
      children.length > 0
        ? children
        : [
            textNode(
              `${context.instanceId}__label`,
              label,
              { x: 8, y: 8, width: Math.max(box.width - 16, 0), height: Math.max(box.height - 16, 0) },
              context,
              spec.id,
              "label",
              { fontSize: 11, textAnchor: "middle" },
            ),
          ];
    const root = groupNode(spec, context, box, label, visibleChildren, {
      coordinateSpace: "local",
      direction: "row",
      gap: numberProp(props, "gap", 10),
    });
    const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: root.id } };
    const roleCounts = new Map<string, number>();
    for (const { role, fragment } of ordered) {
      const index = roleCounts.get(role) ?? 0;
      roleCounts.set(role, index + 1);
      const primary = fragment.roots[0];
      if (primary !== undefined) {
        ports[index === 0 ? role : `${role}[${index}]`] = { nodeId: primary.id };
      }
      for (const [portName, endpointValue] of Object.entries(fragment.ports)) {
        ports[`${role}${index === 0 ? "" : `[${index}]`}.${portName}`] = endpointValue;
      }
    }
    return success(
      [root],
      ports,
      actionsFor(spec, root.id, visibleChildren.map((child) => child.id)),
    );
  };
}

function tableRowFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const cells = slots.cells ?? slots.children ?? [];
    const columnWidth = cells.length > 0 ? box.width / cells.length : box.width;
    const suppliedChildren = cells.flatMap((fragment, index) =>
      fragment.roots.map((root) => ({
        ...root,
        geometry: {
          ...root.geometry,
          x: index * columnWidth,
          y: 0,
          width: columnWidth,
          height: box.height,
        },
      })),
    );
    const children =
      suppliedChildren.length > 0
        ? suppliedChildren
        : [
            textNode(
              `${context.instanceId}__label`,
              visibleText(props, "table row"),
              { x: 6, y: 4, width: Math.max(box.width - 12, 0), height: Math.max(box.height - 8, 0) },
              context,
              spec.id,
              "label",
              { fontSize: 10, textAnchor: "start" },
            ),
          ];
    const root = groupNode(spec, context, box, visibleText(props, "table row"), children, {
      coordinateSpace: "local",
      direction: "row",
      gap: numberProp(props, "gap", 0),
    });
    const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: root.id } };
    cells.forEach((fragment, index) => {
      const cell = fragment.roots[0];
      if (cell !== undefined) {
        ports[`cell[${index}]`] = { nodeId: cell.id };
      }
      for (const [portName, endpointValue] of Object.entries(fragment.ports)) {
        ports[`cell[${index}].${portName}`] = endpointValue;
      }
    });
    return success(
      [root],
      ports,
      actionsFor(spec, root.id, children.map((child) => child.id)),
    );
  };
}

function tableFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const rows = slots.rows ?? slots.children ?? [];
    if (rows.length === 0) {
      const authoredColumns = Array.isArray(props.columns)
        ? props.columns.filter(
            (entry): entry is string => typeof entry === "string" && entry.length > 0,
          )
        : [];
      const labels = authoredColumns.length > 0 ? authoredColumns : stringItems(props);
      const cellLabels = labels.length > 0 ? labels : [visibleText(props, "table")];
      const title = stringProp(props, "title") ?? cellLabels[0] ?? "table";
      const detail =
        stringProp(props, "detail") ??
        (cellLabels.length > 1 ? cellLabels.slice(1).join(" · ") : undefined);
      const root = groupNode(
        spec,
        context,
        box,
        title,
        [],
        {
          coordinateSpace: "local",
          direction: "column",
          gap: 0,
          overflow: "hidden",
          ...variantStyle(props),
          radius: 6,
        },
        {
          title,
          ...(detail !== undefined ? { detail } : {}),
          ...(cellLabels.length > 0 ? { steps: [...cellLabels] } : {}),
        },
      );
      return success([root], { self: { nodeId: root.id } }, actionsFor(spec, root.id, []));
    }
    const children = rootsOf(rows);
    const title = stringProp(props, "title") ?? visibleText(props, "table");
    const root = groupNode(
      spec,
      context,
      box,
      title,
      children,
      {
        coordinateSpace: "local",
        direction: "column",
        gap: numberProp(props, "gap", 0),
        overflow: "hidden",
        ...variantStyle(props),
        radius: 6,
      },
      { title },
    );
    const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: root.id } };
    rows.forEach((fragment, rowIndex) => {
      const row = fragment.roots[0];
      if (row !== undefined) {
        ports[`row[${rowIndex}]`] = { nodeId: row.id };
      }
      for (const [portName, endpointValue] of Object.entries(fragment.ports)) {
        const cellMatch = /^cell\[(\d+)\]$/.exec(portName);
        if (cellMatch !== null) {
          ports[`cell[${rowIndex}][${cellMatch[1]}]`] = endpointValue;
        }
        ports[`row[${rowIndex}].${portName}`] = endpointValue;
      }
    });
    return success(
      [root],
      ports,
      actionsFor(spec, root.id, children.map((child) => child.id)),
    );
  };
}

function collectionSemanticProps(
  spec: CatalogSpec,
  props: Props,
  texts: readonly string[],
): Readonly<{ capabilityId: string; props: Record<string, JsonValue>; label: string }> {
  const label = visibleText(props, spec.id.slice(4));
  const authoredTitle = stringProp(props, "title");
  const title = authoredTitle ?? texts[0] ?? label;
  const authoredDetail = stringProp(props, "detail");
  const detail =
    authoredDetail ??
    (authoredTitle !== undefined && texts.length > 0
      ? texts.join(" · ")
      : texts.length > 1
        ? texts.slice(1).join(" · ")
        : undefined);
  if (spec.id === "sdk.timeline") {
    return {
      capabilityId: "core.stepper",
      label: title,
      props: { steps: [...texts] },
    };
  }
  if (spec.id === "sdk.timelineItem") {
    return {
      capabilityId: "core.panel",
      label: title,
      props: {
        title,
        ...(detail !== undefined ? { detail } : {}),
      },
    };
  }
  if (
    spec.id === "sdk.tagList" ||
    spec.id === "sdk.breadcrumb" ||
    spec.id === "sdk.tabs" ||
    spec.id === "sdk.pagination"
  ) {
    return {
      capabilityId: "core.lane",
      label: title,
      props: {
        title: texts.length > 0 ? texts.join(spec.id === "sdk.breadcrumb" ? " / " : " · ") : title,
        ...(detail !== undefined && texts.length <= 1 ? { detail } : {}),
        ...(texts.length > 0 ? { steps: [...texts] } : {}),
      },
    };
  }
  // stat / metric / keyValue / propertyList (and other panel-like collections)
  return {
    capabilityId: "core.panel",
    label: title,
    props: {
      title,
      ...(detail !== undefined ? { detail } : {}),
      ...(texts.length > 0 ? { steps: [...texts] } : {}),
    },
  };
}

function collectionFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, slots, context) => {
    const box = geometry(props, spec);
    const fragments = childFragments(slots);
    const suppliedRoots = rootsOf(fragments);
    const itemTexts = stringItems(props);
    const texts =
      itemTexts.length > 0
        ? itemTexts
        : (() => {
            const fallback = visibleText(props, spec.id.slice(4));
            const title = stringProp(props, "title");
            const detail = stringProp(props, "detail");
            if (title !== undefined && detail !== undefined) {
              return [title, detail];
            }
            if (title !== undefined) {
              return [title];
            }
            return [fallback];
          })();
    // Keep loose text generation only for sdk.list; other collections map to
    // semantic chrome capabilities with synthesized title/detail/steps.
    if (spec.id === "sdk.list") {
      const rail = spec.capabilityId === "layout.rail";
      const rowHeight = Math.max(20, Math.min(32, box.height / Math.max(texts.length, 1)));
      const generated = texts.map((text, index) =>
        textNode(
          `${context.instanceId}__item-${index}`,
          text,
          rail
            ? {
                x: 0,
                y: 0,
                width: 0,
                height: Math.max(box.height, rowHeight),
              }
            : {
                x: 8,
                y: index * rowHeight,
                width: Math.max(box.width - 16, 0),
                height: rowHeight,
              },
          context,
          spec.id,
          `item-${index}`,
          {
            fontSize: 11,
            fontWeight: "normal",
            textAnchor: "start",
          },
        ),
      );
      const children = suppliedRoots.length > 0 ? suppliedRoots : generated;
      const root = groupNode(spec, context, box, visibleText(props, "list"), children, {
        coordinateSpace: "local",
        direction: "column",
        gap: numberProp(props, "gap", 6),
      });
      const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: root.id } };
      children.forEach((child, index) => {
        ports[`item[${index}]`] = { nodeId: child.id };
      });
      return success(
        [root],
        ports,
        actionsFor(spec, root.id, children.map((child) => child.id)),
      );
    }

    const semantic = collectionSemanticProps(spec, props, texts);
    const children = suppliedRoots;
    const root = origin(
      {
        kind: "group",
        id: context.instanceId,
        capabilityId: semantic.capabilityId,
        geometry: box,
        style: {
          coordinateSpace: "local",
          direction: spec.id === "sdk.timeline" ? "row" : "column",
          gap: numberProp(props, "gap", 6),
          ...variantStyle(props),
          radius: 6,
        },
        props: semantic.props,
        accessibility: { label: semantic.label },
        fallback: semantic.label,
        sourceMap: context.sourceMap,
        children,
      },
      context,
      spec.id,
      "root",
    );
    const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: root.id } };
    children.forEach((child, index) => {
      ports[`item[${index}]`] = { nodeId: child.id };
    });
    return success(
      [root],
      ports,
      actionsFor(spec, root.id, children.map((child) => child.id)),
    );
  };
}

function numericValues(props: Props): readonly number[] {
  const raw = props.values;
  return Array.isArray(raw)
    ? raw.filter((entry): entry is number => typeof entry === "number" && Number.isFinite(entry))
    : [];
}

function indicatorFactory(spec: CatalogSpec): SdkComponentFactory {
  return (props, _slots, context) => {
    const box = geometry(props, spec);
    const min = numberProp(props, "min", 0);
    const max = numberProp(props, "max", 1);
    const authored = numberProp(props, "value", min);
    if (!(max > min)) {
      return fail(
        context,
        "SDK_INDICATOR_RANGE_INVALID",
        `${spec.id} "${context.instanceId}" requires max greater than min.`,
        "Increase max or decrease min.",
      );
    }
    const shouldClamp = props.clamp === true;
    if (!shouldClamp && (authored < min || authored > max)) {
      return fail(
        context,
        "SDK_INDICATOR_VALUE_OUT_OF_RANGE",
        `${spec.id} "${context.instanceId}" value ${authored} is outside [${min}, ${max}].`,
        "Provide an in-range value or set clamp = true.",
      );
    }
    const value = Math.min(max, Math.max(min, authored));
    const ratio = (value - min) / (max - min);
    const children: RenderNodeIr[] = [];
    let pulseTargetId: string | undefined;
    if (spec.id === "sdk.sparkline") {
      const values = numericValues(props);
      if (values.length < 2) {
        return fail(
          context,
          "SDK_SPARKLINE_VALUES_REQUIRED",
          `sdk.sparkline "${context.instanceId}" requires at least two finite values.`,
          "Provide values = [n1, n2, ...].",
        );
      }
      const low = Math.min(...values);
      const high = Math.max(...values);
      const span = high - low || 1;
      const points = values.map((entry, index) => ({
        x: (index / (values.length - 1)) * box.width,
        y: box.height - ((entry - low) / span) * box.height,
      }));
      const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x} ${point.y}`).join(" ");
      children.push(
        origin(
          {
            kind: "connector",
            id: `${context.instanceId}__series`,
            capabilityId: "core.path",
            geometry: { x: 0, y: 0, width: box.width, height: box.height },
            style: { fill: "none", stroke: "@theme.accent.primary", strokeWidth: 2, markerEnd: "none" },
            accessibility: { label: "sparkline series" },
            fallback: "sparkline",
            sourceMap: context.sourceMap,
            from: points[0]!,
            to: points[points.length - 1]!,
            points,
            path,
          },
          context,
          spec.id,
          "series",
        ),
      );
    } else if (spec.id === "sdk.rating") {
      for (let index = 0; index < 5; index += 1) {
        children.push(
          textNode(
            `${context.instanceId}__rating-${index}`,
            index < Math.round(ratio * 5) ? "★" : "☆",
            { x: index * 24, y: 0, width: 22, height: box.height },
            context,
            spec.id,
            `rating-${index}`,
            { fontSize: 18, textAnchor: "middle", fill: "@theme.accent.tertiary" },
          ),
        );
      }
    } else if (spec.id === "sdk.semaphore") {
      ["danger", "warning", "success"].forEach((variant, index) => {
        children.push(
          rectNode(
            `${context.instanceId}__light-${index}`,
            { x: index * 24, y: 2, width: 18, height: 18 },
            context,
            spec.id,
            `light-${index}`,
            variant,
            {
              ...variantStyle({ variant }),
              radius: 18,
              opacity: index === Math.round(ratio * 2) ? 1 : 0.25,
            },
            "core.circle",
          ),
        );
      });
    } else {
      const track = rectNode(
        `${context.instanceId}__track`,
        { x: 0, y: 0, width: box.width, height: box.height },
        context,
        spec.id,
        "track",
        "indicator track",
        { fill: "@theme.surface.elevated", radius: box.height / 2 },
      );
      const fill = rectNode(
        `${context.instanceId}__value`,
        { x: 0, y: 0, width: box.width * ratio, height: box.height },
        context,
        spec.id,
        "value",
        "indicator value",
        { fill: "@theme.accent.primary", radius: box.height / 2 },
      );
      pulseTargetId = fill.id;
      // Track and value intentionally share the same band: the value bar
      // paints over the full-width track to show progress. Nest both under
      // a layout.overlay band so the resolver treats the pairing as an
      // authored overlay rather than a sibling-spacing defect.
      children.push(
        origin(
          {
            kind: "group",
            id: `${context.instanceId}__band`,
            capabilityId: "layout.overlay",
            geometry: { x: 0, y: 0, width: box.width, height: box.height },
            style: {},
            accessibility: { label: "indicator band" },
            fallback: "indicator band",
            sourceMap: context.sourceMap,
            children: [track, fill],
          },
          context,
          spec.id,
          "band",
        ),
      );
    }
    const root = groupNode(
      spec,
      context,
      box,
      visibleText(props, spec.id.slice(4)),
      children,
      {
        coordinateSpace: "local",
        ...(spec.id === "sdk.rating" || spec.id === "sdk.semaphore"
          ? { direction: "row", gap: numberProp(props, "gap", 4) }
          : {}),
      },
    );
    return success(
      [root],
      {
        self: { nodeId: root.id },
        value: { nodeId: pulseTargetId ?? children.at(-1)?.id ?? root.id },
      },
      actionsFor(
        spec,
        root.id,
        pulseTargetId !== undefined ? [pulseTargetId] : children.map((child) => child.id),
      ),
    );
  };
}

function factoryFor(spec: CatalogSpec): SdkComponentFactory {
  if (spec.id === "sdk.mediaObject") {
    return mediaObjectFactory(spec);
  }
  if (spec.id === "sdk.table") {
    return tableFactory(spec);
  }
  if (spec.id === "sdk.tableRow") {
    return tableRowFactory(spec);
  }
  switch (spec.family) {
    case "shape":
      return shapeFactory(spec);
    case "text":
      return textFactory(spec);
    case "icon":
      return iconFactory(spec);
    case "image":
      return imageFactory(spec);
    case "connector":
      return connectorFactory(spec);
    case "container":
      return containerFactory(spec);
    case "collection":
      return collectionFactory(spec);
    case "indicator":
      return indicatorFactory(spec);
  }
}

const BASE_PROP_KEYS = [
  "id",
  "x",
  "y",
  "width",
  "height",
  "description",
  "position",
] as const;

const FAMILY_PROP_KEYS: Readonly<Record<Family, readonly (keyof typeof COMMON_PROPS)[]>> = {
  shape: ["text", "title", "label", "detail", "variant", "path", "surfaceRole", "strokeRole", "inkRole"],
  text: ["text", "title", "label", "detail", "variant", "surfaceRole", "strokeRole", "inkRole"],
  icon: ["text", "title", "label", "detail", "icon", "variant", "surfaceRole", "strokeRole", "inkRole"],
  image: ["src", "label", "variant"],
  connector: ["from", "to", "label", "strokeRole"],
  container: ["title", "label", "direction", "gap", "surfaceRole", "strokeRole", "inkRole"],
  collection: ["title", "label", "detail", "items", "entries", "columns", "direction", "gap", "surfaceRole", "strokeRole", "inkRole"],
  indicator: ["title", "label", "value", "min", "max", "clamp", "values", "variant", "surfaceRole", "strokeRole", "inkRole"],
};

function descriptorPropsFor(spec: CatalogSpec): Readonly<Record<string, ComponentPropDescriptor>> {
  const keys = [...BASE_PROP_KEYS, ...FAMILY_PROP_KEYS[spec.family]];
  return Object.fromEntries(keys.map((key) => [key, COMMON_PROPS[key]!]));
}

function descriptorFor(spec: CatalogSpec): ComponentDescriptor {
  const segment = spec.id.slice(4);
  return {
    id: spec.id,
    symbolExport: segment.charAt(0).toUpperCase() + segment.slice(1),
    version: "1.0.0",
    classification: "flow-only",
    props: descriptorPropsFor(spec),
    slots: spec.slots ?? {},
    events: [],
    capabilityId: spec.capabilityId,
    deterministic: true,
  };
}

/** Exhaustive generic UI/content SDK catalog introduced after the base pack. */
export const GENERIC_CATALOG_COMPONENTS: readonly SdkComponentDefinition[] = CATALOG.map(
  (spec) => ({
    descriptor: descriptorFor(spec),
    factory: factoryFor(spec),
    actions: spec.actions,
  }),
);
