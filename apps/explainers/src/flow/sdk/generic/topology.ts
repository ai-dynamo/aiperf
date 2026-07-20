/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Generic SDK topology factories: `sdk.edge`, `sdk.route`, `sdk.pipeline`,
 * `sdk.fanOut`, `sdk.fanIn`.
 *
 * `sdk.edge` unifies `core.connector` / `core.route` / `core.path` /
 * `core.line` behind one `mode` prop; `sdk.route` is a thin alias that
 * forces `mode: "route"`. `mode: "curve"` (rendered as `core.connector` with
 * `style.route = "curve"`) selects deterministic anchor-aware cubic routing
 * between any of the nine perimeter anchors on connected nodes. Curved edges
 * leave and enter tangentially, arc around other scene nodes (obstacles),
 * separate into lanes when several share the same endpoints, and arc as a loop
 * when an edge returns to its own node. Six optional open-style controls tune
 * it, all falling back to defaults on absent/invalid values:
 * `style.clearance` (obstacle padding, default 12), `style.curvature`
 * (0.05–0.95 bow, default 0.45), `style.avoidObstacles` (default true),
 * `style.preferredSide` (`auto`/`n`/`s`/`e`/`w`, default `auto`),
 * `style.bundle` (merge parallel edges into one corridor, default false), and
 * `style.parallelGap` (lane spacing, default 8). `sdk.pipeline` wires auto
 * edges between consecutive slot fragments and places them left-to-right when
 * per-node x/y are omitted. `sdk.fanOut` /
 * `sdk.fanIn` wrap the existing first-class fan IR (`core.fan-out` /
 * `core.fan-in`) with semantic ports and actions.
 *
 * `from` / `to` endpoints accept a literal endpoint (`nodeId`/`anchor`/
 * `x`/`y`, or a bare string treated as `nodeId`) or a pending semantic
 * reference `{ ref: "instanceId.port" }`. Semantic refs are encoded as a
 * `nodeId` carrying the `SDK_PENDING_REF_PREFIX` sentinel so the Task 5/6
 * expansion pipeline can resolve them once every component instance's
 * ports are known; this module never resolves refs itself.
 */

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
import type {
  ConnectorAxisIr,
  ConnectorEndpointIr,
  FanNodeIr,
  GeometryIr,
  PointIr,
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

type PropRecord = Readonly<Record<string, JsonValue>>;
type SlotRecord = Readonly<Record<string, readonly SceneFragment[]>>;

/** Sentinel prefix marking a `ConnectorEndpointIr.nodeId` as an unresolved semantic ref. */
export const SDK_PENDING_REF_PREFIX = "sdk-ref::";

function isJsonRecord(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberProp(props: PropRecord, key: string, fallback: number): number {
  const value = props[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringProp(props: PropRecord, key: string): string | undefined {
  const value = props[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function boolPropOrUndefined(props: PropRecord, key: string): boolean | undefined {
  const value = props[key];
  return typeof value === "boolean" ? value : undefined;
}

function geometryFromProps(props: PropRecord): GeometryIr {
  const nested = props.geometry;
  const source = isJsonRecord(nested) ? nested : props;
  return {
    x: numberProp(source, "x", 0),
    y: numberProp(source, "y", 0),
    width: numberProp(source, "width", 0),
    height: numberProp(source, "height", 0),
  };
}

function nodeId(context: SdkExpansionContext, role?: string): string {
  return role === undefined ? context.instanceId : `${context.instanceId}__${role}`;
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

function ok(
  roots: readonly RenderNodeIr[],
  ports: Readonly<Record<string, ConnectorEndpointIr>>,
  actions: Readonly<Partial<Record<SdkActionName, readonly string[]>>>,
): Result<SceneFragment> {
  return { ok: true, value: { roots, ports, actions }, diagnostics: [] };
}

function fail(
  context: SdkExpansionContext,
  code: string,
  message: string,
  repair?: string,
): Result<SceneFragment> {
  return {
    ok: false,
    diagnostics: [diagnostic(code, "error", message, context.sourceMap, repair)],
  };
}

function slotFragments(slots: SlotRecord, key: string): readonly SceneFragment[] {
  return slots[key] ?? [];
}

function primaryRootId(fragment: SceneFragment): string | undefined {
  return fragment.roots[0]?.id;
}

function mergeChildPorts(
  fragments: readonly SceneFragment[],
  role: string,
): Record<string, ConnectorEndpointIr> {
  const ports: Record<string, ConnectorEndpointIr> = {};
  fragments.forEach((fragment, index) => {
    const rootId = primaryRootId(fragment);
    if (rootId !== undefined) {
      ports[`${role}[${index}]`] = { nodeId: rootId };
    }
    for (const [portName, endpoint] of Object.entries(fragment.ports)) {
      ports[`${role}[${index}].${portName}`] = endpoint;
    }
  });
  return ports;
}

function scalarStyleFromJson(record: Record<string, JsonValue>): Record<string, StyleValueIr> {
  const out: Record<string, StyleValueIr> = {};
  for (const [key, value] of Object.entries(record)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      out[key] = value;
    }
  }
  return out;
}

function createDescriptor(args: {
  id: string;
  capabilityId: string;
  props: Readonly<Record<string, ComponentPropDescriptor>>;
  slots?: Readonly<Record<string, ComponentSlotDescriptor>>;
}): ComponentDescriptor {
  const segment = args.id.includes(".") ? args.id.split(".", 2)[1]! : args.id;
  return {
    id: args.id,
    symbolExport: segment.charAt(0).toUpperCase() + segment.slice(1),
    version: "1.0.0",
    classification: "flow-only",
    props: args.props,
    slots: args.slots ?? {},
    events: [],
    capabilityId: args.capabilityId,
    deterministic: true,
  };
}

const TOPOLOGY_ACTIONS = ["enter", "draw", "trace"] as const satisfies readonly SdkActionName[];
const FAN_ACTIONS = [
  "enter",
  "draw",
  "trace",
  "emphasis",
] as const satisfies readonly SdkActionName[];

const GEOMETRY_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: 0 },
  height: { type: "number", required: false, default: 0 },
};

// --- endpoint resolution ----------------------------------------------------

/**
 * Resolves a `from` / `to` prop value into a `ConnectorEndpointIr`.
 *
 * Accepts a bare string (`nodeId`), a literal endpoint object
 * (`nodeId`/`anchor`/`x`/`y`), or `{ ref: "instanceId.port" }`, which is
 * encoded as a pending-ref `nodeId` for later semantic-port resolution.
 * Returns `undefined` when the value cannot be resolved to any of these
 * shapes.
 */
export function endpointFromJson(value: JsonValue | undefined): ConnectorEndpointIr | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? { nodeId: trimmed } : undefined;
  }
  if (!isJsonRecord(value)) {
    return undefined;
  }
  if (typeof value.ref === "string" && value.ref.length > 0) {
    return { nodeId: `${SDK_PENDING_REF_PREFIX}${value.ref}` };
  }
  const x = typeof value.x === "number" && Number.isFinite(value.x) ? value.x : undefined;
  const y = typeof value.y === "number" && Number.isFinite(value.y) ? value.y : undefined;
  const resolvedNodeId =
    typeof value.nodeId === "string" && value.nodeId.length > 0 ? value.nodeId : undefined;
  const anchor =
    typeof value.anchor === "string" && value.anchor.length > 0 ? value.anchor : undefined;
  if (x !== undefined && y !== undefined) {
    return {
      x,
      y,
      ...(resolvedNodeId !== undefined ? { nodeId: resolvedNodeId } : {}),
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  if (resolvedNodeId !== undefined) {
    return { nodeId: resolvedNodeId, ...(anchor !== undefined ? { anchor } : {}) };
  }
  return undefined;
}

/** True when an endpoint carries an unresolved semantic-ref `nodeId` sentinel. */
export function isPendingSdkRef(endpoint: ConnectorEndpointIr): boolean {
  return (
    typeof endpoint.nodeId === "string" && endpoint.nodeId.startsWith(SDK_PENDING_REF_PREFIX)
  );
}

function pointFromJson(value: JsonValue | undefined): PointIr | undefined {
  if (!isJsonRecord(value)) {
    return undefined;
  }
  const x = typeof value.x === "number" && Number.isFinite(value.x) ? value.x : undefined;
  const y = typeof value.y === "number" && Number.isFinite(value.y) ? value.y : undefined;
  return x !== undefined && y !== undefined ? { x, y } : undefined;
}

function axisFromJson(value: JsonValue | undefined): ConnectorAxisIr | undefined {
  return value === "x" || value === "y" ? value : undefined;
}

function pointsFromJson(
  value: JsonValue | undefined,
): ReadonlyArray<PointIr | ConnectorEndpointIr> | undefined {
  if (!Array.isArray(value) || value.length === 0) {
    return undefined;
  }
  const points: Array<PointIr | ConnectorEndpointIr> = [];
  for (const entry of value) {
    const endpoint = endpointFromJson(entry as JsonValue);
    if (endpoint !== undefined) {
      points.push(endpoint);
    }
  }
  return points.length > 0 ? points : undefined;
}

function endpointListFromJson(
  value: JsonValue | undefined,
): readonly ConnectorEndpointIr[] | undefined {
  if (Array.isArray(value)) {
    const list = value
      .map((entry) => endpointFromJson(entry as JsonValue))
      .filter((entry): entry is ConnectorEndpointIr => entry !== undefined);
    return list.length > 0 ? list : undefined;
  }
  const single = endpointFromJson(value);
  return single !== undefined ? [single] : undefined;
}

function geometryFromEndpointsOrProps(
  props: PropRecord,
  from: ConnectorEndpointIr,
  to: ConnectorEndpointIr,
): GeometryIr {
  const authored = geometryFromProps(props);
  if (authored.width !== 0 || authored.height !== 0 || authored.x !== 0 || authored.y !== 0) {
    return authored;
  }
  if (
    typeof from.x === "number" &&
    typeof from.y === "number" &&
    typeof to.x === "number" &&
    typeof to.y === "number"
  ) {
    return {
      x: Math.min(from.x, to.x),
      y: Math.min(from.y, to.y),
      width: Math.abs(to.x - from.x),
      height: Math.abs(to.y - from.y),
    };
  }
  return authored;
}

// --- sdk.edge / sdk.route ----------------------------------------------------

type EdgeMode = "connector" | "route" | "path" | "line" | "curve";

const EDGE_MODE_CAPABILITY: Readonly<Record<EdgeMode, string>> = {
  connector: "core.connector",
  route: "core.route",
  path: "core.path",
  line: "core.line",
  curve: "core.connector",
};

function edgeModeFromProp(value: string | undefined): EdgeMode | undefined {
  return value === "connector" ||
    value === "route" ||
    value === "path" ||
    value === "line" ||
    value === "curve"
    ? value
    : undefined;
}

function buildEdgeFragment(
  props: PropRecord,
  context: SdkExpansionContext,
  componentId: string,
  forcedMode?: EdgeMode,
): Result<SceneFragment> {
  const mode = forcedMode ?? edgeModeFromProp(stringProp(props, "mode")) ?? "connector";
  const capabilityId = EDGE_MODE_CAPABILITY[mode];

  const from = endpointFromJson(props.from);
  if (from === undefined) {
    return fail(
      context,
      "SDK_EDGE_FROM_REQUIRED",
      `${componentId} "${context.instanceId}" requires a resolvable "from" endpoint.`,
      'Provide {nodeId}, {x, y}, a bare string node id, or {ref: "instance.port"}.',
    );
  }
  const to = endpointFromJson(props.to);
  if (to === undefined) {
    return fail(
      context,
      "SDK_EDGE_TO_REQUIRED",
      `${componentId} "${context.instanceId}" requires a resolvable "to" endpoint.`,
      'Provide {nodeId}, {x, y}, a bare string node id, or {ref: "instance.port"}.',
    );
  }

  const id = nodeId(context);
  const via = pointFromJson(props.via);
  const axis = axisFromJson(props.axis);
  const path = stringProp(props, "path");
  const points = pointsFromJson(props.points);
  const arrowhead = boolPropOrUndefined(props, "arrowhead");
  const styleOverride = isJsonRecord(props.style) ? scalarStyleFromJson(props.style) : {};

  let style: Record<string, StyleValueIr> = { fill: "none" };
  if (mode === "route") {
    style.route = "elbow";
  }
  if (mode === "curve") {
    style.route = "curve";
  }
  if (arrowhead === false) {
    style.markerEnd = "none";
    style.arrowhead = false;
  } else {
    style.markerEnd = "arrow";
    style.arrowhead = true;
  }
  style = { ...style, ...styleOverride };

  const label = stringProp(props, "label") ?? `${componentId} edge`;
  const description = stringProp(props, "description");

  const node: RenderNodeIr = withOrigin(
    {
      kind: "connector",
      id,
      capabilityId,
      geometry: geometryFromEndpointsOrProps(props, from, to),
      style,
      accessibility: { label, ...(description !== undefined ? { description } : {}) },
      fallback: label,
      sourceMap: context.sourceMap,
      from,
      to,
      ...(via !== undefined ? { via } : {}),
      ...(axis !== undefined ? { axis } : {}),
      ...(path !== undefined ? { path } : {}),
      ...(points !== undefined ? { points } : {}),
    },
    context,
    componentId,
    "root",
  );

  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: id },
    source: from,
    target: to,
  };
  return ok([node], ports, { enter: [id], draw: [id], trace: [id] });
}

const edgeFactory: SdkComponentFactory = (props, _slots, context) =>
  buildEdgeFragment(props, context, "sdk.edge");

const routeFactory: SdkComponentFactory = (props, _slots, context) =>
  buildEdgeFragment(props, context, "sdk.route", "route");

const EDGE_ENDPOINT_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  from: { type: "endpoint", required: true },
  to: { type: "endpoint", required: true },
  via: { type: "object", required: false },
  axis: { type: "string", required: false },
  path: { type: "string", required: false },
  points: { type: "array", required: false },
  arrowhead: { type: "boolean", required: false },
  label: { type: "string", required: false },
  description: { type: "string", required: false },
  style: { type: "object", required: false },
  ...GEOMETRY_PROPS,
};

export const SDK_EDGE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.edge",
    capabilityId: "core.connector",
    props: {
      id: { type: "string", required: true },
      mode: { type: "string", required: false, default: "connector" },
      ...EDGE_ENDPOINT_PROPS,
    },
  }),
  factory: edgeFactory,
  actions: TOPOLOGY_ACTIONS,
};

export const SDK_ROUTE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.route",
    capabilityId: "core.route",
    props: {
      id: { type: "string", required: true },
      ...EDGE_ENDPOINT_PROPS,
    },
  }),
  factory: routeFactory,
  actions: TOPOLOGY_ACTIONS,
};

// --- sdk.pipeline -------------------------------------------------------------

function portOrRootEndpoint(
  fragment: SceneFragment,
  preferredPorts: readonly string[],
  anchor: string,
): ConnectorEndpointIr {
  for (const portName of preferredPorts) {
    const port = fragment.ports[portName];
    if (port !== undefined) {
      return port;
    }
  }
  const rootId = primaryRootId(fragment);
  return rootId !== undefined ? { nodeId: rootId, anchor } : { x: 0, y: 0 };
}

const PIPELINE_DEFAULT_GAP = 24;
const PIPELINE_DEFAULT_NODE_WIDTH = 96;
const PIPELINE_DEFAULT_NODE_HEIGHT = 56;

/** Place a pipeline stage root at a local (x, y), preserving size. */
function placePipelineNode(node: RenderNodeIr, x: number, y: number): RenderNodeIr {
  const geometry = node.geometry ?? {
    x: 0,
    y: 0,
    width: PIPELINE_DEFAULT_NODE_WIDTH,
    height: PIPELINE_DEFAULT_NODE_HEIGHT,
  };
  return {
    ...node,
    geometry: {
      ...geometry,
      x,
      y,
      width: geometry.width > 0 ? geometry.width : PIPELINE_DEFAULT_NODE_WIDTH,
      height: geometry.height > 0 ? geometry.height : PIPELINE_DEFAULT_NODE_HEIGHT,
    },
  };
}

/**
 * Ordered stages wired left-to-right. Slot children keep their ids/ports; the
 * factory assigns non-overlapping local geometry so stages do not stack at
 * the origin when authors omit per-node x/y (the common pipeline pattern).
 */
const pipelineFactory: SdkComponentFactory = (props, slots, context) => {
  const nodes = slotFragments(slots, "nodes");
  if (nodes.length < 2) {
    return fail(
      context,
      "SDK_PIPELINE_NODES_REQUIRED",
      `sdk.pipeline "${context.instanceId}" requires at least two "nodes" slot entries.`,
      "Provide two or more component invocations in the nodes slot.",
    );
  }
  const mode = edgeModeFromProp(stringProp(props, "edgeMode")) ?? "connector";
  const capabilityId = EDGE_MODE_CAPABILITY[mode];
  const gap = numberProp(props, "gap", PIPELINE_DEFAULT_GAP);
  const authored = geometryFromProps(props);
  const baseStyle: Record<string, StyleValueIr> =
    mode === "route"
      ? { route: "elbow", fill: "none" }
      : mode === "curve"
        ? { route: "curve", fill: "none" }
        : { fill: "none" };

  let cursorX = 0;
  let maxHeight = 0;
  const placedRoots: RenderNodeIr[] = [];
  const placedFragments: SceneFragment[] = [];
  for (const fragment of nodes) {
    const root = fragment.roots[0];
    if (root === undefined) {
      continue;
    }
    // Shift every stage root by the same delta so secondary roots keep their
    // offsets relative to the primary while the stage lands at cursorX.
    const originX = root.geometry?.x ?? 0;
    const originY = root.geometry?.y ?? 0;
    const placedStageRoots = fragment.roots.map((node, index) => {
      if (index === 0) {
        return placePipelineNode(node, cursorX, 0);
      }
      const geometry = node.geometry ?? {
        x: 0,
        y: 0,
        width: PIPELINE_DEFAULT_NODE_WIDTH,
        height: PIPELINE_DEFAULT_NODE_HEIGHT,
      };
      return placePipelineNode(
        node,
        cursorX + (geometry.x - originX),
        0 + (geometry.y - originY),
      );
    });
    const placed = placedStageRoots[0]!;
    const width = placed.geometry?.width ?? PIPELINE_DEFAULT_NODE_WIDTH;
    const height = placed.geometry?.height ?? PIPELINE_DEFAULT_NODE_HEIGHT;
    placedRoots.push(...placedStageRoots);
    placedFragments.push({
      roots: placedStageRoots,
      ports: fragment.ports,
      actions: fragment.actions,
    });
    cursorX += width + gap;
    maxHeight = Math.max(maxHeight, height);
  }
  if (placedRoots.length < 2) {
    return fail(
      context,
      "SDK_PIPELINE_NODES_REQUIRED",
      `sdk.pipeline "${context.instanceId}" requires at least two placeable "nodes" slot roots.`,
    );
  }
  const contentWidth = Math.max(0, cursorX - gap);

  const edges: RenderNodeIr[] = [];
  const edgePorts: Record<string, ConnectorEndpointIr> = {};
  for (let index = 0; index < placedFragments.length - 1; index += 1) {
    const sourceFragment = placedFragments[index]!;
    const targetFragment = placedFragments[index + 1]!;
    const source = portOrRootEndpoint(sourceFragment, ["output", "result", "next", "self"], "e");
    const target = portOrRootEndpoint(targetFragment, ["input", "control", "self"], "w");
    const edgeId = nodeId(context, `edge-${index}`);
    const label = `${context.instanceId} stage ${index + 1} to ${index + 2}`;
    const edge: RenderNodeIr = withOrigin(
      {
        kind: "connector",
        id: edgeId,
        capabilityId,
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: baseStyle,
        accessibility: { label },
        fallback: label,
        sourceMap: context.sourceMap,
        from: source,
        to: target,
      },
      context,
      "sdk.pipeline",
      `edge-${index}`,
    );
    edges.push(edge);
    edgePorts[`edge[${index}]`] = { nodeId: edgeId };
  }

  const edgeIds = edges.map((edge) => edge.id);
  const groupLabel = stringProp(props, "label") ?? "pipeline";
  const group: RenderNodeIr = withOrigin(
    {
      kind: "group",
      id: context.instanceId,
      capabilityId: "core.group",
      geometry: {
        x: authored.x,
        y: authored.y,
        width: authored.width > 0 ? Math.max(authored.width, contentWidth) : contentWidth,
        height: authored.height > 0 ? Math.max(authored.height, maxHeight) : maxHeight,
      },
      style: { coordinateSpace: "local" },
      accessibility: { label: groupLabel },
      fallback: groupLabel,
      sourceMap: context.sourceMap,
      children: [...placedRoots, ...edges],
    },
    context,
    "sdk.pipeline",
    "root",
  );
  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: context.instanceId },
    ...mergeChildPorts(placedFragments, "node"),
    ...edgePorts,
  };
  return ok([group], ports, {
    enter: [context.instanceId, ...placedRoots.map((root) => root.id), ...edgeIds],
    draw: edgeIds,
    trace: edgeIds,
  });
};

export const SDK_PIPELINE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.pipeline",
    capabilityId: "core.group",
    props: {
      id: { type: "string", required: true },
      edgeMode: { type: "string", required: false, default: "connector" },
      gap: { type: "number", required: false, default: PIPELINE_DEFAULT_GAP },
      label: { type: "string", required: false },
      x: { type: "number", required: false, default: 0 },
      y: { type: "number", required: false, default: 0 },
      width: { type: "number", required: false, default: 0 },
      height: { type: "number", required: false, default: 0 },
    },
    slots: {
      nodes: { accepts: "sdk.*", required: true },
    },
  }),
  factory: pipelineFactory,
  actions: TOPOLOGY_ACTIONS,
};

// --- sdk.fanOut / sdk.fanIn ---------------------------------------------------

function fanFactory(capability: "core.fan-out" | "core.fan-in"): SdkComponentFactory {
  return (props, _slots, context) => {
    const componentId = capability === "core.fan-out" ? "sdk.fanOut" : "sdk.fanIn";
    const fromKey = capability === "core.fan-out" ? "single" : "list";
    const from =
      fromKey === "single" ? endpointFromJson(props.from) : endpointListFromJson(props.from);
    const to =
      fromKey === "single" ? endpointListFromJson(props.to) : endpointFromJson(props.to);

    if (from === undefined) {
      return fail(
        context,
        "SDK_FAN_FROM_REQUIRED",
        `${componentId} "${context.instanceId}" has an unresolvable "from" endpoint.`,
      );
    }
    if (to === undefined) {
      return fail(
        context,
        "SDK_FAN_TO_REQUIRED",
        `${componentId} "${context.instanceId}" has an unresolvable "to" endpoint.`,
      );
    }
    if (capability === "core.fan-out" && (!Array.isArray(to) || to.length < 2)) {
      return fail(
        context,
        "SDK_FANOUT_BRANCHES_REQUIRED",
        `${componentId} "${context.instanceId}" requires at least two "to" branch endpoints.`,
        "Provide an array of two or more endpoints or refs for to.",
      );
    }
    if (capability === "core.fan-in" && (!Array.isArray(from) || from.length < 2)) {
      return fail(
        context,
        "SDK_FANIN_BRANCHES_REQUIRED",
        `${componentId} "${context.instanceId}" requires at least two "from" branch endpoints.`,
        "Provide an array of two or more endpoints or refs for from.",
      );
    }

    const axis = axisFromJson(props.axis);
    const junction = pointFromJson(props.junction);
    const id = nodeId(context);
    const label =
      stringProp(props, "label") ?? (capability === "core.fan-out" ? "fan out" : "fan in");
    const styleOverride = isJsonRecord(props.style) ? scalarStyleFromJson(props.style) : {};

    const node: RenderNodeIr = withOrigin(
      {
        kind: "fan",
        id,
        capability,
        capabilityId: capability,
        geometry: geometryFromProps(props),
        style: styleOverride,
        accessibility: { label },
        fallback: label,
        sourceMap: context.sourceMap,
        from,
        to,
        ...(axis !== undefined ? { axis } : {}),
        ...(junction !== undefined ? { junction } : {}),
      } satisfies FanNodeIr,
      context,
      componentId,
      "root",
    );

    const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: id } };
    // Cardinality is already enforced above: fan-out has a single `from`
    // trunk and an array `to`; fan-in is the mirror image.
    const trunk = (capability === "core.fan-out" ? from : to) as ConnectorEndpointIr;
    const branches = (
      capability === "core.fan-out" ? to : from
    ) as readonly ConnectorEndpointIr[];
    ports.trunk = trunk;
    branches.forEach((endpoint, index) => {
      ports[`branch[${index}]`] = endpoint;
    });
    return ok([node], ports, { enter: [id], draw: [id], trace: [id], emphasis: [id] });
  };
}

const FAN_ENDPOINT_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  from: { type: "endpoint | endpoint[]", required: true },
  to: { type: "endpoint | endpoint[]", required: true },
  axis: { type: "string", required: false },
  junction: { type: "object", required: false },
  label: { type: "string", required: false },
  style: { type: "object", required: false },
  ...GEOMETRY_PROPS,
};

export const SDK_FAN_OUT: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.fanOut",
    capabilityId: "core.fan-out",
    props: {
      id: { type: "string", required: true },
      ...FAN_ENDPOINT_PROPS,
    },
  }),
  factory: fanFactory("core.fan-out"),
  actions: FAN_ACTIONS,
};

export const SDK_FAN_IN: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.fanIn",
    capabilityId: "core.fan-in",
    props: {
      id: { type: "string", required: true },
      ...FAN_ENDPOINT_PROPS,
    },
  }),
  factory: fanFactory("core.fan-in"),
  actions: FAN_ACTIONS,
};

/** Generic SDK topology pack, ready for `registry.ts` to splice in place of stubs. */
export const GENERIC_TOPOLOGY_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  SDK_EDGE,
  SDK_ROUTE,
  SDK_PIPELINE,
  SDK_FAN_OUT,
  SDK_FAN_IN,
];
