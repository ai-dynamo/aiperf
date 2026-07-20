/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! AIPerf domain SDK pack: process topology composites.
//!
//! `aiperf.controllerCells`, `aiperf.workerMerge`, and `aiperf.registryBootstrap`
//! compose deterministic Scene IR fragments describing the controller/cell
//! partitioning, worker-local accumulation merge, and extension registration
//! shapes documented in `AGENTS.md`. Components carry no deck-specific prose
//! or fixed slide ids; callers supply labels and theme roles as props.
//!
//! The generic `sdk.*` factory pack (`sdk/generic/chrome.ts`, `topology.ts`,
//! ...) is being built concurrently and is not yet available. Until it lands,
//! this module composes ordinary Scene IR through a small local kit (below)
//! that mirrors the same primitives (`core.rect` + `core.text` chrome,
//! `core.connector`, `core.fan-out` / `core.fan-in`). Swapping the local kit
//! for `sdk.card` / `sdk.pipeline` / `sdk.fanOut` / `sdk.fanIn` calls is a
//! self-contained follow-up once those factories are registered.

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

function fanInNode(
  id: string,
  from: readonly ConnectorEndpointIr[],
  to: ConnectorEndpointIr,
  sourceMap: SourceRange,
  label: string,
): FanNodeIr {
  return {
    kind: "fan",
    id,
    capabilityId: "core.fan-in",
    capability: "core.fan-in",
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

// --- aiperf.controllerCells -------------------------------------------------

const CELL_WIDTH = 128;
const CELL_HEIGHT = 48;
const CELL_GAP = 24;
const CELL_ROW_GAP = 64;

/**
 * Controller partitioning work across `N` cell processes (`--cells N` /
 * `runtime.cells`), dispatching from a single controller down to each cell.
 */
function controllerCellsFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.controllerCells";
  const controllerLabel = readString(props, "controllerLabel", "Controller");
  const cellLabels = readStringArray(props, "cellLabels", ["Cell 1", "Cell 2"]);
  const failure = requireLabels(cellLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");
  const origin = authoredOrigin(props);

  const { instanceId, sourceMap } = context;
  const rowWidth = cellLabels.length * CELL_WIDTH + (cellLabels.length - 1) * CELL_GAP;
  const width = Math.max(rowWidth, CELL_WIDTH);
  const height = CELL_HEIGHT * 2 + CELL_ROW_GAP;

  const controllerId = nodeId(instanceId, "controller");
  const controller = labeledBox({
    id: controllerId,
    label: controllerLabel,
    x: (width - CELL_WIDTH) / 2,
    y: 0,
    width: CELL_WIDTH,
    height: CELL_HEIGHT,
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
  });

  const cellY = CELL_HEIGHT + CELL_ROW_GAP;
  const cells = cellLabels.map((label, index) =>
    labeledBox({
      id: nodeId(instanceId, `cell-${index}`),
      label,
      x: index * (CELL_WIDTH + CELL_GAP),
      y: cellY,
      width: CELL_WIDTH,
      height: CELL_HEIGHT,
      sourceMap,
      surfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const dispatchId = nodeId(instanceId, "dispatch");
  const controllerOut: ConnectorEndpointIr = { nodeId: controllerId, anchor: "s" };
  const cellIns: readonly ConnectorEndpointIr[] = cells.map((cell) => ({
    nodeId: cell.id,
    anchor: "n",
  }));
  const dispatch: RenderNodeIr =
    cells.length >= 2
      ? fanOutNode(dispatchId, controllerOut, cellIns, sourceMap, `${controllerLabel} dispatch`)
      : connectorNode(
          dispatchId,
          controllerOut,
          cellIns[0] ?? controllerOut,
          sourceMap,
          `${controllerLabel} dispatch`,
        );

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    `${controllerLabel} cell topology`,
    [controller, dispatch, ...cells],
  );

  const ports: Record<string, ConnectorEndpointIr> = { controller: controllerOut };
  cells.forEach((cell, index) => {
    ports[`cell[${index}]`] = { nodeId: cell.id, anchor: "n" };
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: [controllerId, ...cells.map((cell) => cell.id)],
        draw: [dispatchId],
        trace: [dispatchId],
        emphasis: [controllerId, ...cells.map((cell) => cell.id)],
      },
    },
    diagnostics: [],
  };
}

// --- aiperf.workerMerge ------------------------------------------------------

/**
 * `N` worker-local accumulators folding into a single merge boundary,
 * matching the "accumulate per worker, merge at a boundary" runtime rule.
 */
function workerMergeFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.workerMerge";
  const workerLabels = readStringArray(props, "workerLabels", ["Worker 1", "Worker 2"]);
  const failure = requireLabels(workerLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const mergeLabel = readString(props, "mergeLabel", "Merge");
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");
  const origin = authoredOrigin(props);

  const { instanceId, sourceMap } = context;
  const rowWidth =
    workerLabels.length * CELL_WIDTH + (workerLabels.length - 1) * CELL_GAP;
  const width = Math.max(rowWidth, CELL_WIDTH);
  const height = CELL_HEIGHT * 2 + CELL_ROW_GAP;

  const workers = workerLabels.map((label, index) =>
    labeledBox({
      id: nodeId(instanceId, `worker-${index}`),
      label,
      x: index * (CELL_WIDTH + CELL_GAP),
      y: 0,
      width: CELL_WIDTH,
      height: CELL_HEIGHT,
      sourceMap,
      surfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const mergeId = nodeId(instanceId, "merge");
  const merge = labeledBox({
    id: mergeId,
    label: mergeLabel,
    x: (width - CELL_WIDTH) / 2,
    y: CELL_HEIGHT + CELL_ROW_GAP,
    width: CELL_WIDTH,
    height: CELL_HEIGHT,
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
  });

  const foldId = nodeId(instanceId, "fold");
  const workerOuts: readonly ConnectorEndpointIr[] = workers.map((worker) => ({
    nodeId: worker.id,
    anchor: "s",
  }));
  const mergeIn: ConnectorEndpointIr = { nodeId: mergeId, anchor: "n" };
  const fold: RenderNodeIr =
    workers.length >= 2
      ? fanInNode(foldId, workerOuts, mergeIn, sourceMap, `${mergeLabel} fold`)
      : connectorNode(foldId, workerOuts[0] ?? mergeIn, mergeIn, sourceMap, `${mergeLabel} fold`);

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    `${mergeLabel} worker merge`,
    [...workers, fold, merge],
  );

  const ports: Record<string, ConnectorEndpointIr> = {
    result: { nodeId: mergeId, anchor: "s" },
  };
  workers.forEach((worker, index) => {
    ports[`worker[${index}]`] = { nodeId: worker.id, anchor: "s" };
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: [...workers.map((worker) => worker.id), mergeId],
        draw: [foldId],
        trace: [foldId],
        emphasis: [mergeId],
      },
    },
    diagnostics: [],
  };
}

// --- aiperf.registryBootstrap ------------------------------------------------

/**
 * `AIPerfExtension` categories (endpoints, datasets, transports, ...)
 * registering into the transactional `AIPerfRegistry`.
 */
function registryBootstrapFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.registryBootstrap";
  const categoryLabels = readStringArray(props, "categoryLabels", [
    "Endpoints",
    "Datasets",
    "Transports",
    "Exporters",
  ]);
  const failure = requireLabels(categoryLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const registryLabel = readString(props, "registryLabel", "AIPerfRegistry");
  const categorySurfaceRole = readThemeRole(props, "categorySurfaceRole", "surface.panel");
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.raised");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");
  const origin = authoredOrigin(props);

  const { instanceId, sourceMap } = context;
  const rowWidth =
    categoryLabels.length * CELL_WIDTH + (categoryLabels.length - 1) * CELL_GAP;
  const width = Math.max(rowWidth, CELL_WIDTH);
  const height = CELL_HEIGHT * 2 + CELL_ROW_GAP;

  const categories = categoryLabels.map((label, index) =>
    labeledBox({
      id: nodeId(instanceId, `category-${index}`),
      label,
      x: index * (CELL_WIDTH + CELL_GAP),
      y: 0,
      width: CELL_WIDTH,
      height: CELL_HEIGHT,
      sourceMap,
      surfaceRole: categorySurfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const registryId = nodeId(instanceId, "registry");
  const registry = labeledBox({
    id: registryId,
    label: registryLabel,
    x: (width - CELL_WIDTH) / 2,
    y: CELL_HEIGHT + CELL_ROW_GAP,
    width: CELL_WIDTH,
    height: CELL_HEIGHT,
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
  });

  const registerId = nodeId(instanceId, "register");
  const categoryOuts: readonly ConnectorEndpointIr[] = categories.map((category) => ({
    nodeId: category.id,
    anchor: "s",
  }));
  const registryIn: ConnectorEndpointIr = { nodeId: registryId, anchor: "n" };
  const register: RenderNodeIr =
    categories.length >= 2
      ? fanInNode(registerId, categoryOuts, registryIn, sourceMap, `${registryLabel} registration`)
      : connectorNode(
          registerId,
          categoryOuts[0] ?? registryIn,
          registryIn,
          sourceMap,
          `${registryLabel} registration`,
        );

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    `${registryLabel} bootstrap`,
    [...categories, register, registry],
  );

  const ports: Record<string, ConnectorEndpointIr> = { registry: registryIn };
  categories.forEach((category, index) => {
    ports[`category[${index}]`] = { nodeId: category.id, anchor: "s" };
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: [...categories.map((category) => category.id), registryId],
        draw: [registerId],
        trace: [registerId],
        emphasis: [registryId],
      },
    },
    diagnostics: [],
  };
}

/** AIPerf architecture pack: controller/cell, worker-merge, and registry topology. */
export const AIPERF_ARCHITECTURE_COMPONENTS: readonly SdkComponentDefinition[] = [
  defineAiperfComponent({
    id: "aiperf.controllerCells",
    props: {
      controllerLabel: stringProp("Controller"),
      cellLabels: stringArrayProp(["Cell 1", "Cell 2"]),
      surfaceRole: stringProp("surface.panel"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
      ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: controllerCellsFactory,
  }),
  defineAiperfComponent({
    id: "aiperf.workerMerge",
    props: {
      workerLabels: stringArrayProp(["Worker 1", "Worker 2"]),
      mergeLabel: stringProp("Merge"),
      surfaceRole: stringProp("surface.panel"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
      ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: workerMergeFactory,
  }),
  defineAiperfComponent({
    id: "aiperf.registryBootstrap",
    props: {
      registryLabel: stringProp("AIPerfRegistry"),
      categoryLabels: stringArrayProp([
        "Endpoints",
        "Datasets",
        "Transports",
        "Exporters",
      ]),
      categorySurfaceRole: stringProp("surface.panel"),
      surfaceRole: stringProp("surface.raised"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
      ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: registryBootstrapFactory,
  }),
];
