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
//! Metrics and exporter nodes use native semantic `core.chip` Scene IR so
//! their chrome and text remain renderer-owned rather than serialized children.

import type { ComponentPropDescriptor } from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
import type {
  ConnectorEndpointIr,
  ConnectorNodeIr,
  FanNodeIr,
  GeometryIr,
  GroupNodeIr,
  RenderNodeIr,
} from "../../schema/ir.js";
import type { JsonValue } from "../../schema/json-value.js";
import type { SourceRange } from "../../schema/source.js";
import {
  THEME_ROLES,
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
  // Omit the prop to use defaults; an explicit array (even empty) is author intent
  // and must reach requireLabels rather than silently substituting the fallback.
  if (!Array.isArray(value)) {
    return fallback;
  }
  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0,
  );
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

function semanticChip(args: {
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
  return {
    kind: "group",
    id,
    capabilityId: "core.chip",
    geometry: { x, y, width, height },
    style: {
      fill: themeRole(surfaceRole),
      stroke: themeRole(lineRole),
      strokeWidth: 1.2,
      radius: 8,
    },
    props: { label, inkRole },
    accessibility: { label },
    fallback: label,
    sourceMap,
    children: [],
  };
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
  const source = semanticChip({
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
    semanticChip({
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
