/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! AIPerf domain SDK pack: metrics export composite.
//!
//! `aiperf.metricsExport` composes a deterministic Scene IR fragment
//! describing the metrics plane fanning out to configured exporters (JSON,
//! CSV, Parquet, console, timeslice, server-metrics, accuracy, OTLP, MLflow,
//! W&B, ...). The component carries no deck-specific prose or fixed slide
//! ids; callers supply labels and theme roles as props.
//!
//! The generic `sdk.*` factory pack (`sdk/generic/chrome.ts`, `topology.ts`,
//! ...) is being built concurrently and is not yet available. Until it lands,
//! this module composes ordinary Scene IR through a small local kit (below)
//! that mirrors the same primitives (`core.rect` + `core.text` chrome,
//! `core.fan-out`). Swapping the local kit for `sdk.card` / `sdk.fanOut`
//! calls is a self-contained follow-up once those factories are registered.

import type { ComponentPropDescriptor } from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
import type {
  ConnectorEndpointIr,
  ConnectorNodeIr,
  FanNodeIr,
  GeometryIr,
  GroupNodeIr,
  RectNodeIr,
  RenderNodeIr,
  TextNodeIr,
} from "../../schema/ir.js";
import type { JsonValue } from "../../schema/json-value.js";
import type { SourceRange } from "../../schema/source.js";
import {
  THEME_ROLES,
  type StyleValueIr,
  type ThemeRole,
  type ThemeRoleReferenceIr,
} from "../../schema/theme.js";
import type {
  SceneFragment,
  SdkActionName,
  SdkComponentDefinition,
  SdkComponentFactory,
  SdkExpansionContext,
} from "../types.js";

// --- Local composition kit (temporary; see module doc) ---------------------

function nodeId(instanceId: string, role: string): string {
  return `${instanceId}__${role}`;
}

function themeRole(role: ThemeRole): ThemeRoleReferenceIr {
  return { kind: "theme-role", role };
}

function readString(
  props: Readonly<Record<string, JsonValue>>,
  key: string,
  fallback: string,
): string {
  const value = props[key];
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function readStringArray(
  props: Readonly<Record<string, JsonValue>>,
  key: string,
  fallback: readonly string[],
): readonly string[] {
  const value = props[key];
  if (!Array.isArray(value)) {
    return fallback;
  }
  const strings = value.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0,
  );
  return strings.length > 0 ? strings : fallback;
}

function readThemeRole(
  props: Readonly<Record<string, JsonValue>>,
  key: string,
  fallback: ThemeRole,
): ThemeRole {
  const value = props[key];
  if (
    typeof value === "string" &&
    (THEME_ROLES as readonly string[]).includes(value)
  ) {
    return value as ThemeRole;
  }
  return fallback;
}

function requireLabels(
  labels: readonly string[],
  componentId: string,
  context: SdkExpansionContext,
): Result<SceneFragment> | undefined {
  if (labels.length > 0) {
    return undefined;
  }
  return {
    ok: false,
    diagnostics: [
      diagnostic(
        "SDK_FACTORY_FAILED",
        "error",
        `AIPerf component "${componentId}" instance "${context.instanceId}" requires at least one label.`,
        context.sourceMap,
        "Provide at least one entry in the labels array, or omit the prop to use the component default.",
      ),
    ],
  };
}

function textNode(
  id: string,
  text: string,
  geometry: GeometryIr,
  sourceMap: SourceRange,
  style: Readonly<Record<string, StyleValueIr>>,
): TextNodeIr {
  return {
    kind: "text",
    id,
    capabilityId: "core.text",
    geometry,
    style,
    accessibility: { label: text },
    fallback: text,
    sourceMap,
    text,
  };
}

function rectNode(
  id: string,
  geometry: GeometryIr,
  sourceMap: SourceRange,
  style: Readonly<Record<string, StyleValueIr>>,
  label: string,
): RectNodeIr {
  return {
    kind: "rect",
    id,
    capabilityId: "core.rect",
    geometry,
    style,
    accessibility: { label },
    fallback: label,
    sourceMap,
  };
}

function groupNode(
  id: string,
  geometry: GeometryIr,
  sourceMap: SourceRange,
  label: string,
  children: readonly RenderNodeIr[],
): GroupNodeIr {
  return {
    kind: "group",
    id,
    capabilityId: "core.group",
    geometry,
    style: { coordinateSpace: "local" },
    accessibility: { label },
    fallback: label,
    sourceMap,
    children,
  };
}

/** Optional authored origin for AIPerf composites (`x` / `y` scene placement). */
function authoredOrigin(
  props: Readonly<Record<string, JsonValue>>,
): Readonly<{ x: number; y: number }> {
  const x = props.x;
  const y = props.y;
  return {
    x: typeof x === "number" && Number.isFinite(x) ? x : 0,
    y: typeof y === "number" && Number.isFinite(y) ? y : 0,
  };
}

const GEOMETRY_ORIGIN_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
};

function connectorNode(
  id: string,
  from: ConnectorEndpointIr,
  to: ConnectorEndpointIr,
  sourceMap: SourceRange,
  label: string,
): ConnectorNodeIr {
  return {
    kind: "connector",
    id,
    capabilityId: "core.connector",
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: {
      stroke: themeRole("line.structural"),
      strokeWidth: 1.5,
      fill: "none",
      markerEnd: "arrow",
    },
    accessibility: { label },
    fallback: label,
    sourceMap,
    from,
    to,
  };
}

function fanOutNode(
  id: string,
  from: ConnectorEndpointIr,
  to: readonly ConnectorEndpointIr[],
  sourceMap: SourceRange,
  label: string,
): FanNodeIr {
  return {
    kind: "fan",
    id,
    capabilityId: "core.fan-out",
    capability: "core.fan-out",
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: {
      stroke: themeRole("line.structural"),
      strokeWidth: 1.5,
      fill: "none",
      markerEnd: "arrow",
    },
    accessibility: { label },
    fallback: label,
    sourceMap,
    from,
    to,
    axis: "y",
  };
}

function labeledBox(args: {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  sourceMap: SourceRange;
  surfaceRole: ThemeRole;
  inkRole: ThemeRole;
  lineRole: ThemeRole;
}): GroupNodeIr {
  const { id, label, x, y, width, height, sourceMap, surfaceRole, inkRole, lineRole } =
    args;
  const chrome = rectNode(
    `${id}-chrome`,
    { x: 0, y: 0, width, height },
    sourceMap,
    {
      fill: themeRole(surfaceRole),
      stroke: themeRole(lineRole),
      strokeWidth: 1.2,
      radius: 8,
    },
    label,
  );
  const text = textNode(
    `${id}-label`,
    label,
    { x: 0, y: Math.max((height - 16) / 2, 0), width, height: 16 },
    sourceMap,
    {
      fontSize: 12,
      fontWeight: "bold",
      textAnchor: "middle",
      fill: themeRole(inkRole),
    },
  );
  return groupNode(id, { x, y, width, height }, sourceMap, label, [chrome, text]);
}

function symbolExportFromId(id: string): string {
  const segment = id.includes(".") ? id.split(".", 2)[1]! : id;
  return segment.charAt(0).toUpperCase() + segment.slice(1);
}

function stringProp(defaultValue: string): ComponentPropDescriptor {
  return { type: "string", required: false, default: defaultValue };
}

function stringArrayProp(defaultValue: readonly string[]): ComponentPropDescriptor {
  return { type: "string[]", required: false, default: [...defaultValue] };
}

function defineAiperfComponent(args: {
  id: string;
  props: Readonly<Record<string, ComponentPropDescriptor>>;
  actions: readonly SdkActionName[];
  factory: SdkComponentFactory;
}): SdkComponentDefinition {
  return {
    descriptor: {
      id: args.id,
      symbolExport: symbolExportFromId(args.id),
      version: "1.0.0",
      classification: "flow-only",
      props: { id: { type: "string", required: true }, ...args.props },
      slots: {},
      events: [],
      capabilityId: "core.group",
      deterministic: true,
    },
    factory: args.factory,
    actions: args.actions,
  };
}

// --- aiperf.metricsExport --------------------------------------------------

const CHIP_WIDTH = 128;
const CHIP_HEIGHT = 48;
const CHIP_GAP = 24;
const ROW_GAP = 64;

/**
 * Metrics plane (record / aggregate / derived / phase-window / sweep) fanning
 * out to the configured exporter plane sinks.
 */
function metricsExportFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.metricsExport";
  const exporterLabels = readStringArray(props, "exporterLabels", [
    "JSON",
    "CSV",
    "Console",
  ]);
  const failure = requireLabels(exporterLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const sourceLabel = readString(props, "sourceLabel", "Metrics");
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.panel");
  const exporterSurfaceRole = readThemeRole(props, "exporterSurfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");
  const origin = authoredOrigin(props);

  const { instanceId, sourceMap } = context;
  const rowWidth =
    exporterLabels.length * CHIP_WIDTH + (exporterLabels.length - 1) * CHIP_GAP;
  const width = Math.max(rowWidth, CHIP_WIDTH);
  const height = CHIP_HEIGHT * 2 + ROW_GAP;

  const sourceId = nodeId(instanceId, "source");
  const source = labeledBox({
    id: sourceId,
    label: sourceLabel,
    x: (width - CHIP_WIDTH) / 2,
    y: 0,
    width: CHIP_WIDTH,
    height: CHIP_HEIGHT,
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
  });

  const exportersY = CHIP_HEIGHT + ROW_GAP;
  const exporters = exporterLabels.map((label, index) =>
    labeledBox({
      id: nodeId(instanceId, `exporter-${index}`),
      label,
      x: index * (CHIP_WIDTH + CHIP_GAP),
      y: exportersY,
      width: CHIP_WIDTH,
      height: CHIP_HEIGHT,
      sourceMap,
      surfaceRole: exporterSurfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const fanId = nodeId(instanceId, "export");
  const sourceOut: ConnectorEndpointIr = { nodeId: sourceId, anchor: "s" };
  const exporterIns: readonly ConnectorEndpointIr[] = exporters.map((exporter) => ({
    nodeId: exporter.id,
    anchor: "n",
  }));
  const fan: RenderNodeIr =
    exporters.length >= 2
      ? fanOutNode(fanId, sourceOut, exporterIns, sourceMap, `${sourceLabel} export`)
      : connectorNode(
          fanId,
          sourceOut,
          exporterIns[0] ?? sourceOut,
          sourceMap,
          `${sourceLabel} export`,
        );

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    `${sourceLabel} export`,
    [source, fan, ...exporters],
  );

  const ports: Record<string, ConnectorEndpointIr> = { metrics: sourceOut };
  exporters.forEach((exporter, index) => {
    ports[`exporter[${index}]`] = { nodeId: exporter.id, anchor: "n" };
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: [sourceId, ...exporters.map((exporter) => exporter.id)],
        draw: [fanId],
        trace: [fanId],
        emphasis: [sourceId, ...exporters.map((exporter) => exporter.id)],
      },
    },
    diagnostics: [],
  };
}

/** AIPerf metrics pack: metrics-plane-to-exporter fan-out composite. */
export const AIPERF_METRICS_COMPONENTS: readonly SdkComponentDefinition[] = [
  defineAiperfComponent({
    id: "aiperf.metricsExport",
    props: {
      sourceLabel: stringProp("Metrics"),
      exporterLabels: stringArrayProp(["JSON", "CSV", "Console"]),
      exporterSurfaceRole: stringProp("surface.panel"),
      surfaceRole: stringProp("surface.panel"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
      ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: metricsExportFactory,
  }),
];
