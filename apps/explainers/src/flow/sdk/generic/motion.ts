/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Generic SDK motion factories: `sdk.signal`, `sdk.pulse`, `sdk.flow`.
 *
 * `sdk.signal` emits a `motion.signal` connector bound to an edge (`edge`),
 * node-anchored (`from`/`to`), or path-anchored (`path`/`points`), with the
 * standard traveling-dot stroke defaults (`opacity: 0.55`,
 * `strokeWidth: 2.4`). `sdk.pulse` emits a
 * `motion.pulse` overlay rect — either standalone (explicit geometry, the
 * "hollow rect overlay" replacement) or wrapping a `target` slot fragment
 * (re-emitting its roots alongside a matching halo, the `style.pulse: true`
 * hack replacement). `sdk.flow` composes a signal with an optional static
 * backing edge so a single call can draw the pipe and animate the traveling
 * dot together.
 */

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
import type {
  ConnectorEndpointIr,
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
const SDK_PENDING_REF_PREFIX = "sdk-ref::";

function isJsonRecord(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberProp(props: PropRecord, key: string, fallback: number): number {
  const value = props[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function numberPropOrUndefined(props: PropRecord, key: string): number | undefined {
  const value = props[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function stringProp(props: PropRecord, key: string): string | undefined {
  const value = props[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function boolProp(props: PropRecord, key: string, fallback: boolean): boolean {
  const value = props[key];
  return typeof value === "boolean" ? value : fallback;
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

function hasExplicitGeometry(geometry: GeometryIr): boolean {
  return geometry.x !== 0 || geometry.y !== 0 || geometry.width !== 0 || geometry.height !== 0;
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

function flattenRoots(fragments: readonly SceneFragment[]): readonly RenderNodeIr[] {
  return fragments.flatMap((fragment) => fragment.roots);
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

/** Resolves `from` / `to` into a literal endpoint or a pending semantic-ref sentinel. */
function endpointFromJson(value: JsonValue | undefined): ConnectorEndpointIr | undefined {
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

function boundingGeometryFromPoints(
  points: ReadonlyArray<PointIr | ConnectorEndpointIr> | undefined,
): GeometryIr | undefined {
  if (points === undefined || points.length === 0) {
    return undefined;
  }
  const xs = points
    .map((point) => (typeof point.x === "number" ? point.x : undefined))
    .filter((value): value is number => value !== undefined);
  const ys = points
    .map((point) => (typeof point.y === "number" ? point.y : undefined))
    .filter((value): value is number => value !== undefined);
  if (xs.length === 0 || ys.length === 0) {
    return undefined;
  }
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  return {
    x: minX,
    y: minY,
    width: Math.max(...xs) - minX,
    height: Math.max(...ys) - minY,
  };
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

const GEOMETRY_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: 0 },
  height: { type: "number", required: false, default: 0 },
};

const SIGNAL_DEFAULT_OPACITY = 0.55;
const SIGNAL_DEFAULT_STROKE_WIDTH = 2.4;

// --- sdk.signal --------------------------------------------------------------

/** Builds the motion-signal connector node shared by `sdk.signal` and `sdk.flow`. */
function buildSignalNode(
  props: PropRecord,
  context: SdkExpansionContext,
  componentId: string,
  role: string,
): Result<RenderNodeIr> {
  const edgeRef = stringProp(props, "edge");
  const from = endpointFromJson(props.from);
  const to = endpointFromJson(props.to);
  const path = stringProp(props, "path");
  const points = pointsFromJson(props.points);
  const hasEndpointInput = from !== undefined || to !== undefined;
  const isNodeMode = from !== undefined && to !== undefined;
  const isPathMode = path !== undefined || points !== undefined;
  const modeCount = [edgeRef !== undefined, hasEndpointInput, isPathMode].filter(Boolean).length;
  if (modeCount !== 1 || (hasEndpointInput && !isNodeMode)) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "SDK_SIGNAL_MODE_CONFLICT",
          "error",
          `${componentId} "${context.instanceId}" requires exactly one motion mode: edge, from + to, or path/points.`,
          context.sourceMap,
          "Remove conflicting geometry and provide one complete motion mode.",
        ),
      ],
    };
  }

  const id = nodeId(context, role);
  const styleOverride = isJsonRecord(props.style) ? scalarStyleFromJson(props.style) : {};
  const authoredGeometry = geometryFromProps(props);
  const geometry = hasExplicitGeometry(authoredGeometry)
    ? authoredGeometry
    : boundingGeometryFromPoints(points) ?? { x: 0, y: 0, width: 0, height: 0 };
  const style: Record<string, StyleValueIr> = {
    fill: "none",
    markerEnd: "none",
    motion: "signal",
    opacity: numberProp(props, "opacity", SIGNAL_DEFAULT_OPACITY),
    strokeWidth: numberProp(props, "strokeWidth", SIGNAL_DEFAULT_STROKE_WIDTH),
    ...styleOverride,
  };
  const label = stringProp(props, "label") ?? "motion signal";

  const node: RenderNodeIr = {
    kind: "connector",
    id,
    capabilityId: "motion.signal",
    geometry,
    style,
    accessibility: { label },
    fallback: label,
    sourceMap: context.sourceMap,
    ...(edgeRef !== undefined ? { edgeRef } : {}),
    ...(isNodeMode ? { from, to } : {}),
    ...(path !== undefined ? { path } : {}),
    ...(points !== undefined ? { points } : {}),
  };
  return { ok: true, value: withOrigin(node, context, componentId, role), diagnostics: [] };
}

const signalFactory: SdkComponentFactory = (props, _slots, context) => {
  const result = buildSignalNode(props, context, "sdk.signal", "root");
  if (!result.ok) {
    return result;
  }
  const node = result.value;
  return ok([node], { self: { nodeId: node.id } }, {
    enter: [node.id],
    draw: [node.id],
    trace: [node.id],
    fade: [node.id],
  });
};

const SIGNAL_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  id: { type: "string", required: true },
  edge: { type: "string", required: false },
  from: { type: "endpoint", required: false },
  to: { type: "endpoint", required: false },
  path: { type: "string", required: false },
  points: { type: "array", required: false },
  opacity: { type: "number", required: false, default: SIGNAL_DEFAULT_OPACITY },
  strokeWidth: { type: "number", required: false, default: SIGNAL_DEFAULT_STROKE_WIDTH },
  label: { type: "string", required: false },
  style: { type: "object", required: false },
  ...GEOMETRY_PROPS,
};

export const SDK_SIGNAL: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.signal",
    capabilityId: "motion.signal",
    props: SIGNAL_PROPS,
  }),
  factory: signalFactory,
  actions: ["enter", "draw", "trace", "fade"] as const satisfies readonly SdkActionName[],
};

// --- sdk.pulse -----------------------------------------------------------------

const PULSE_DEFAULT_OPACITY = 0.45;
const PULSE_DEFAULT_STROKE_WIDTH = 2;

const pulseFactory: SdkComponentFactory = (props, slots, context) => {
  const targetSlot = slotFragments(slots, "target");
  const targetRoot = targetSlot[0]?.roots[0];
  const explicitGeometry = geometryFromProps(props);
  const geometry = hasExplicitGeometry(explicitGeometry) ? explicitGeometry : targetRoot?.geometry;
  if (geometry === undefined) {
    return fail(
      context,
      "SDK_PULSE_GEOMETRY_REQUIRED",
      `sdk.pulse "${context.instanceId}" requires explicit geometry or a "target" slot fragment to overlay.`,
      "Provide x/y/width/height, or a positioned target slot fragment.",
    );
  }

  const id = nodeId(context);
  const label = stringProp(props, "label") ?? "motion pulse";
  const radius = numberPropOrUndefined(props, "radius");
  const styleOverride = isJsonRecord(props.style) ? scalarStyleFromJson(props.style) : {};
  const style: Record<string, StyleValueIr> = {
    fill: "none",
    opacity: numberProp(props, "opacity", PULSE_DEFAULT_OPACITY),
    strokeWidth: numberProp(props, "strokeWidth", PULSE_DEFAULT_STROKE_WIDTH),
    ...(radius !== undefined ? { radius } : {}),
    ...styleOverride,
  };

  const node: RenderNodeIr = withOrigin(
    {
      kind: "rect",
      id,
      capabilityId: "motion.pulse",
      geometry,
      style,
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
    },
    context,
    "sdk.pulse",
    "root",
  );

  const roots = targetSlot.length > 0 ? [...flattenRoots(targetSlot), node] : [node];
  return ok(roots, { self: { nodeId: id } }, { enter: [id], pulse: [id], fade: [id] });
};

export const SDK_PULSE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.pulse",
    capabilityId: "motion.pulse",
    props: {
      id: { type: "string", required: true },
      opacity: { type: "number", required: false, default: PULSE_DEFAULT_OPACITY },
      strokeWidth: { type: "number", required: false, default: PULSE_DEFAULT_STROKE_WIDTH },
      radius: { type: "number", required: false },
      label: { type: "string", required: false },
      style: { type: "object", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: {
      target: { accepts: "sdk.*", required: false },
    },
  }),
  factory: pulseFactory,
  actions: ["enter", "pulse", "fade"] as const satisfies readonly SdkActionName[],
};

// --- sdk.flow --------------------------------------------------------------------

function geometryFromEndpointsOrZero(
  from: ConnectorEndpointIr,
  to: ConnectorEndpointIr,
): GeometryIr {
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
  return { x: 0, y: 0, width: 0, height: 0 };
}

/** Optional static backing edge for `sdk.flow`, bound to the same `draw`/`trace` actions as the signal. */
function buildFlowEdgeNode(
  props: PropRecord,
  context: SdkExpansionContext,
): Result<RenderNodeIr> {
  const from = endpointFromJson(props.from);
  const to = endpointFromJson(props.to);
  if (from === undefined || to === undefined) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "SDK_FLOW_EDGE_ENDPOINTS_REQUIRED",
          "error",
          `sdk.flow "${context.instanceId}" edge requires resolvable from/to endpoints.`,
          context.sourceMap,
          'Provide literal endpoints or {ref: "instance.port"} for from/to, or set edge:false.',
        ),
      ],
    };
  }
  const isRoute = stringProp(props, "edgeMode") === "route";
  const id = nodeId(context, "edge");
  const label = stringProp(props, "label") ?? "flow edge";
  const style: Record<string, StyleValueIr> = isRoute
    ? { route: "elbow", fill: "none" }
    : { fill: "none" };
  const node: RenderNodeIr = {
    kind: "connector",
    id,
    capabilityId: isRoute ? "core.route" : "core.connector",
    geometry: geometryFromEndpointsOrZero(from, to),
    style,
    accessibility: { label },
    fallback: label,
    sourceMap: context.sourceMap,
    from,
    to,
  };
  return { ok: true, value: withOrigin(node, context, "sdk.flow", "edge"), diagnostics: [] };
}

const flowFactory: SdkComponentFactory = (props, _slots, context) => {
  const roots: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};
  const drawTargets: string[] = [];

  let signalProps: PropRecord = props;
  if (boolProp(props, "edge", false)) {
    const edgeResult = buildFlowEdgeNode(props, context);
    if (!edgeResult.ok) {
      return edgeResult;
    }
    roots.push(edgeResult.value);
    ports.edge = { nodeId: edgeResult.value.id };
    drawTargets.push(edgeResult.value.id);
    // Bind the companion signal to the edge so resolution shares one path
    // instead of duplicating from/to geometry as a standalone signal.
    const {
      from: _from,
      to: _to,
      path: _path,
      points: _points,
      edge: _edgeFlag,
      ...rest
    } = props;
    signalProps = {
      ...rest,
      edge: edgeResult.value.id,
    };
  }

  const signalResult = buildSignalNode(signalProps, context, "sdk.flow", "signal");
  if (!signalResult.ok) {
    return signalResult;
  }
  const signalNode = signalResult.value;

  roots.push(signalNode);
  ports.signal = { nodeId: signalNode.id };
  ports.self = { nodeId: signalNode.id };
  drawTargets.push(signalNode.id);

  return ok(roots, ports, {
    enter: roots.map((root) => root.id),
    draw: drawTargets,
    // Keep `trace` identical to `draw` so authored `trace <flow>` cues also
    // stroke the optional backing edge (IR missing-draw-cue otherwise).
    trace: drawTargets,
    fade: [signalNode.id],
  });
};

export const SDK_FLOW: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.flow",
    capabilityId: "motion.signal",
    props: {
      ...SIGNAL_PROPS,
      edge: { type: "boolean", required: false, default: false },
      edgeMode: { type: "string", required: false, default: "connector" },
    },
  }),
  factory: flowFactory,
  actions: ["enter", "draw", "trace", "fade"] as const satisfies readonly SdkActionName[],
};

/** Generic SDK motion pack, ready for `registry.ts` to splice in place of stubs. */
export const GENERIC_MOTION_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  SDK_SIGNAL,
  SDK_PULSE,
  SDK_FLOW,
];
