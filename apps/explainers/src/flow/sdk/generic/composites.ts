/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Generic SDK composite factories: `sdk.matrix`, `sdk.layerStack`,
 * `sdk.hubSpoke`, `sdk.tree`, `sdk.bidirectionalLink`, `sdk.stateTransition`.
 *
 * These finish the generic pack's layout, topology, and motion families
 * alongside `generic/layout.ts`, `generic/topology.ts`, and
 * `generic/motion.ts`:
 *
 * - `sdk.matrix` is an explicit row/column grid over already-expanded child
 *   fragments, exposing both flat (`cell[i]`) and 2D (`cell[r][c]`)
 *   addressing over the existing `layout.grid` arrangement `SceneRenderer`
 *   already computes.
 * - `sdk.layerStack` overlays child fragments in the same footprint with a
 *   small per-layer depth offset (later slot entries paint on top, matching
 *   SVG paint order), replacing manual "deck of cards" coordinate math.
 * - `sdk.hubSpoke` and `sdk.tree` wire auto-generated edges between a
 *   `hub`/`root` fragment and its `spokes`/`children` fragments without
 *   repositioning them, the same auto-wiring convention `sdk.pipeline` uses
 *   for linear chains. `sdk.tree` exposes its own `root` port so nested
 *   `sdk.tree` instances compose into multi-level hierarchies purely through
 *   bottom-up factory expansion.
 * - `sdk.bidirectionalLink` emits two coincident `core.connector` /
 *   `core.route` edges (`forward` + `backward`, each with its own
 *   `markerEnd` arrowhead) instead of relying on a two-sided arrowhead style
 *   `SceneRenderer` does not support, optionally separated by a small
 *   perpendicular `gap` when literal endpoint coordinates are resolvable.
 * - `sdk.stateTransition` composes `from` / `to` state fragments with a
 *   motion-styled transition edge and an optional `trigger` label, binding
 *   `pulse` to the two states and `trace`/`fade` to the transition edge so a
 *   single call can drive a full FSM-transition animation beat.
 *
 * Every factory is pure: no DOM, React, network, wall clock, or mutable
 * global state. Generated node ids are seeded from `context.instanceId`
 * (`${instanceId}` for the fragment root, `${instanceId}__role` for
 * generated children) so expansion is stable across repeated calls.
 *
 * This module is deliberately self-contained (mirroring `generic/layout.ts`,
 * `generic/topology.ts`, and `generic/motion.ts`) so it can be integrated
 * into `sdk/registry.ts` by another change: import
 * `GENERIC_COMPOSITE_SDK_COMPONENTS` and splice it into
 * `GENERIC_SDK_COMPONENTS`, replacing the matching `createStubDefinition`
 * entries for the six component ids implemented here.
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
import { portOrRootEndpoint } from "./topology.js";

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

function primaryRootId(fragment: SceneFragment): string | undefined {
  return fragment.roots[0]?.id;
}

/** Ports for each slot fragment: `${role}[i]` plus forwarded `${role}[i].port`. */
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

function geometryFromEndpointsOrProps(
  props: PropRecord,
  from: ConnectorEndpointIr,
  to: ConnectorEndpointIr,
): GeometryIr {
  const authored = geometryFromProps(props);
  if (hasExplicitGeometry(authored)) {
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

const LAYOUT_ACTIONS = ["enter", "stagger"] as const satisfies readonly SdkActionName[];
const TOPOLOGY_ACTIONS = ["enter", "draw", "trace"] as const satisfies readonly SdkActionName[];
const MOTION_ACTIONS = [
  "enter",
  "pulse",
  "trace",
  "fade",
] as const satisfies readonly SdkActionName[];

const GEOMETRY_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: 0 },
  height: { type: "number", required: false, default: 0 },
};

// --- sdk.matrix --------------------------------------------------------------

/** Explicit `cols`/`rows` sizing: `cols` wins, else derives from `rows`, else a square. */
function resolveMatrixCols(cols: number | undefined, rows: number | undefined, count: number): number {
  if (cols !== undefined) {
    return Math.max(1, Math.round(cols));
  }
  if (rows !== undefined) {
    const rowCount = Math.max(1, Math.round(rows));
    return Math.max(1, Math.ceil(count / rowCount));
  }
  return Math.max(1, Math.ceil(Math.sqrt(Math.max(count, 1))));
}

const matrixFactory: SdkComponentFactory = (props, slots, context) => {
  const cellSlot = slotFragments(slots, "cells");
  const children = cellSlot.length > 0 ? cellSlot : slotFragments(slots, "children");
  const roots = flattenRoots(children);
  const cols = resolveMatrixCols(
    numberPropOrUndefined(props, "cols"),
    numberPropOrUndefined(props, "rows"),
    roots.length,
  );
  const gap = numberProp(props, "gap", 32.4);
  const id = nodeId(context);
  const label = stringProp(props, "label") ?? "matrix";
  const group: RenderNodeIr = withOrigin(
    {
      kind: "group",
      id,
      capabilityId: "layout.grid",
      geometry: geometryFromProps(props),
      style: { cols, gap },
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      children: roots,
    },
    context,
    "sdk.matrix",
    "root",
  );

  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: id },
    ...mergeChildPorts(children, "cell"),
  };
  children.forEach((fragment, index) => {
    const rootId = primaryRootId(fragment);
    if (rootId !== undefined) {
      const row = Math.floor(index / cols);
      const col = index % cols;
      ports[`cell[${row}][${col}]`] = { nodeId: rootId };
    }
  });

  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (roots.length > 0) {
    actions.stagger = roots.map((root) => root.id);
  }
  return ok([group], ports, actions);
};

export const SDK_MATRIX: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.matrix",
    capabilityId: "layout.grid",
    props: {
      id: { type: "string", required: true },
      cols: { type: "number", required: false },
      rows: { type: "number", required: false },
      gap: { type: "number", required: false, default: 32.4 },
      label: { type: "string", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: {
      cells: { accepts: "sdk.*", required: false },
      children: { accepts: "sdk.*", required: false },
    },
  }),
  factory: matrixFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.layerStack ------------------------------------------------------------

const LAYER_STACK_DEFAULT_OFFSET = 21.6;

function shiftGeometry<T extends RenderNodeIr>(node: T, dx: number, dy: number): T {
  return { ...node, geometry: { ...node.geometry, x: node.geometry.x + dx, y: node.geometry.y + dy } };
}

function unionGeometry(geometries: readonly GeometryIr[]): GeometryIr {
  if (geometries.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const geometry of geometries) {
    minX = Math.min(minX, geometry.x);
    minY = Math.min(minY, geometry.y);
    maxX = Math.max(maxX, geometry.x + geometry.width);
    maxY = Math.max(maxY, geometry.y + geometry.height);
  }
  return { x: minX, y: minY, width: Math.max(maxX - minX, 0), height: Math.max(maxY - minY, 0) };
}

const layerStackFactory: SdkComponentFactory = (props, slots, context) => {
  const layerSlot = slotFragments(slots, "layers");
  const children = layerSlot.length > 0 ? layerSlot : slotFragments(slots, "children");
  const layerRoots = flattenRoots(children);
  if (layerRoots.length === 0) {
    return fail(
      context,
      "SDK_LAYER_STACK_LAYERS_REQUIRED",
      `sdk.layerStack "${context.instanceId}" requires at least one "layers" slot entry.`,
      "Provide one or more component invocations in the layers slot.",
    );
  }

  const offsetX = numberProp(props, "offsetX", LAYER_STACK_DEFAULT_OFFSET);
  const offsetY = numberProp(props, "offsetY", LAYER_STACK_DEFAULT_OFFSET);
  // Later slot entries shift further and paint last (SVG order), i.e. the
  // top-of-stack / front layer is the last authored entry.
  const shifted = layerRoots.map((root, index) => shiftGeometry(root, index * offsetX, index * offsetY));

  const authoredGeometry = geometryFromProps(props);
  const geometry = hasExplicitGeometry(authoredGeometry)
    ? authoredGeometry
    : unionGeometry(shifted.map((node) => node.geometry));

  const id = nodeId(context);
  const label = stringProp(props, "label") ?? "layer stack";
  const group: RenderNodeIr = withOrigin(
    {
      kind: "group",
      id,
      capabilityId: "core.group",
      geometry,
      style: {},
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      children: shifted,
    },
    context,
    "sdk.layerStack",
    "root",
  );

  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: id },
    ...mergeChildPorts(children, "layer"),
  };
  const bottomId = primaryRootId(children[0]!);
  const topId = primaryRootId(children[children.length - 1]!);
  if (bottomId !== undefined) {
    ports.bottom = { nodeId: bottomId };
  }
  if (topId !== undefined) {
    ports.top = { nodeId: topId };
  }

  return ok([group], ports, { enter: [id], stagger: shifted.map((node) => node.id) });
};

export const SDK_LAYER_STACK: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.layerStack",
    capabilityId: "core.group",
    props: {
      id: { type: "string", required: true },
      offsetX: { type: "number", required: false, default: LAYER_STACK_DEFAULT_OFFSET },
      offsetY: { type: "number", required: false, default: LAYER_STACK_DEFAULT_OFFSET },
      label: { type: "string", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: {
      layers: { accepts: "sdk.*", required: false },
      children: { accepts: "sdk.*", required: false },
    },
  }),
  factory: layerStackFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.hubSpoke --------------------------------------------------------------

type LinkEdgeMode = "connector" | "route";

function linkEdgeMode(value: string | undefined): LinkEdgeMode {
  return value === "route" ? "route" : "connector";
}

function linkEdgeCapabilityAndStyle(
  mode: LinkEdgeMode,
): Readonly<{ capabilityId: string; style: Record<string, StyleValueIr> }> {
  return mode === "route"
    ? { capabilityId: "core.route", style: { route: "elbow", fill: "none", markerEnd: "arrow" } }
    : { capabilityId: "core.connector", style: { fill: "none", markerEnd: "arrow" } };
}

const hubSpokeFactory: SdkComponentFactory = (props, slots, context) => {
  const hubFragment = slotFragments(slots, "hub")[0];
  if (hubFragment === undefined) {
    return fail(
      context,
      "SDK_HUB_SPOKE_HUB_REQUIRED",
      `sdk.hubSpoke "${context.instanceId}" requires a "hub" slot entry.`,
      "Provide exactly one component invocation in the hub slot.",
    );
  }
  const spokes = slotFragments(slots, "spokes");
  if (spokes.length === 0) {
    return fail(
      context,
      "SDK_HUB_SPOKE_SPOKES_REQUIRED",
      `sdk.hubSpoke "${context.instanceId}" requires at least one "spokes" slot entry.`,
      "Provide one or more component invocations in the spokes slot.",
    );
  }

  const mode = linkEdgeMode(stringProp(props, "edgeMode"));
  const { capabilityId, style } = linkEdgeCapabilityAndStyle(mode);
  const hubAnchor = portOrRootEndpoint(hubFragment, ["output", "self"], "e");
  if (hubAnchor === undefined) {
    return fail(
      context,
      "SDK_HUB_SPOKE_HUB_ENDPOINT_UNRESOLVED",
      `sdk.hubSpoke "${context.instanceId}" hub slot entry has no resolvable port or root id.`,
      "Ensure the hub slot's component invocation expands to at least one root node.",
    );
  }

  const edges: RenderNodeIr[] = [];
  const spokePorts: Record<string, ConnectorEndpointIr> = {};
  for (const [index, spoke] of spokes.entries()) {
    const spokeAnchor = portOrRootEndpoint(spoke, ["input", "self"], "w");
    if (spokeAnchor === undefined) {
      return fail(
        context,
        "SDK_HUB_SPOKE_SPOKE_ENDPOINT_UNRESOLVED",
        `sdk.hubSpoke "${context.instanceId}" spoke ${index} has no resolvable port or root id.`,
        "Ensure every spokes slot entry expands to at least one root node.",
      );
    }
    const edgeId = nodeId(context, `edge-${index}`);
    const label = `${context.instanceId} hub to spoke ${index}`;
    edges.push(
      withOrigin(
        {
          kind: "connector",
          id: edgeId,
          capabilityId,
          geometry: geometryFromEndpointsOrProps({}, hubAnchor, spokeAnchor),
          style,
          accessibility: { label },
          fallback: label,
          sourceMap: context.sourceMap,
          from: hubAnchor,
          to: spokeAnchor,
        },
        context,
        "sdk.hubSpoke",
        `edge-${index}`,
      ),
    );
    const spokeRootId = primaryRootId(spoke);
    if (spokeRootId !== undefined) {
      spokePorts[`spoke[${index}]`] = { nodeId: spokeRootId };
    }
  }

  const hubRoots = flattenRoots([hubFragment]);
  const spokeRoots = flattenRoots(spokes);
  const roots = [...hubRoots, ...spokeRoots, ...edges];
  const hubRootId = primaryRootId(hubFragment);
  const edgeIds = edges.map((edge) => edge.id);
  const ports: Record<string, ConnectorEndpointIr> = {
    ...(hubRootId !== undefined ? { hub: { nodeId: hubRootId } } : {}),
    ...spokePorts,
  };
  edges.forEach((edge, index) => {
    ports[`edge[${index}]`] = { nodeId: edge.id };
  });

  return ok(roots, ports, {
    enter: [...hubRoots.map((root) => root.id), ...spokeRoots.map((root) => root.id), ...edgeIds],
    draw: edgeIds,
    trace: edgeIds,
  });
};

const HUB_SPOKE_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  id: { type: "string", required: true },
  edgeMode: { type: "string", required: false, default: "connector" },
  label: { type: "string", required: false },
};

export const SDK_HUB_SPOKE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.hubSpoke",
    capabilityId: "core.group",
    props: HUB_SPOKE_PROPS,
    slots: {
      hub: { accepts: "sdk.*", required: true },
      spokes: { accepts: "sdk.*", required: true },
    },
  }),
  factory: hubSpokeFactory,
  actions: TOPOLOGY_ACTIONS,
};

// --- sdk.tree --------------------------------------------------------------------

const treeFactory: SdkComponentFactory = (props, slots, context) => {
  const rootFragment = slotFragments(slots, "root")[0];
  if (rootFragment === undefined) {
    return fail(
      context,
      "SDK_TREE_ROOT_REQUIRED",
      `sdk.tree "${context.instanceId}" requires a "root" slot entry.`,
      "Provide exactly one component invocation in the root slot.",
    );
  }
  const childFragments = slotFragments(slots, "children");

  const mode = linkEdgeMode(stringProp(props, "edgeMode"));
  const { capabilityId, style } = linkEdgeCapabilityAndStyle(mode);
  // Preferring the fragment's own "root" port lets a nested `sdk.tree`
  // compose as a child/root here without exposing its generated node ids.
  const rootAnchor = portOrRootEndpoint(rootFragment, ["root", "output", "self"], "s");
  if (rootAnchor === undefined) {
    return fail(
      context,
      "SDK_TREE_ROOT_ENDPOINT_UNRESOLVED",
      `sdk.tree "${context.instanceId}" root slot entry has no resolvable port or root id.`,
      "Ensure the root slot's component invocation expands to at least one root node.",
    );
  }

  const edges: RenderNodeIr[] = [];
  const childPorts: Record<string, ConnectorEndpointIr> = {};
  for (const [index, child] of childFragments.entries()) {
    const childAnchor = portOrRootEndpoint(child, ["root", "input", "self"], "n");
    if (childAnchor === undefined) {
      return fail(
        context,
        "SDK_TREE_CHILD_ENDPOINT_UNRESOLVED",
        `sdk.tree "${context.instanceId}" child ${index} has no resolvable port or root id.`,
        "Ensure every children slot entry expands to at least one root node.",
      );
    }
    const edgeId = nodeId(context, `edge-${index}`);
    const label = `${context.instanceId} branch ${index}`;
    edges.push(
      withOrigin(
        {
          kind: "connector",
          id: edgeId,
          capabilityId,
          geometry: geometryFromEndpointsOrProps({}, rootAnchor, childAnchor),
          style,
          accessibility: { label },
          fallback: label,
          sourceMap: context.sourceMap,
          from: rootAnchor,
          to: childAnchor,
        },
        context,
        "sdk.tree",
        `edge-${index}`,
      ),
    );
    const childRootId = primaryRootId(child);
    if (childRootId !== undefined) {
      childPorts[`child[${index}]`] = { nodeId: childRootId };
    }
  }

  const rootRoots = flattenRoots([rootFragment]);
  const childRoots = flattenRoots(childFragments);
  const roots = [...rootRoots, ...childRoots, ...edges];
  const rootRootId = primaryRootId(rootFragment);
  const edgeIds = edges.map((edge) => edge.id);
  const ports: Record<string, ConnectorEndpointIr> = {
    ...(rootRootId !== undefined ? { root: { nodeId: rootRootId }, self: { nodeId: rootRootId } } : {}),
    ...childPorts,
  };
  edges.forEach((edge, index) => {
    ports[`edge[${index}]`] = { nodeId: edge.id };
  });

  return ok(roots, ports, {
    enter: [...rootRoots.map((root) => root.id), ...childRoots.map((root) => root.id), ...edgeIds],
    draw: edgeIds,
    trace: edgeIds,
  });
};

export const SDK_TREE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.tree",
    capabilityId: "core.group",
    props: {
      id: { type: "string", required: true },
      edgeMode: { type: "string", required: false, default: "connector" },
      label: { type: "string", required: false },
    },
    slots: {
      root: { accepts: "sdk.*", required: true },
      children: { accepts: "sdk.*", required: false },
    },
  }),
  factory: treeFactory,
  actions: TOPOLOGY_ACTIONS,
};

// --- sdk.bidirectionalLink -------------------------------------------------------

const BIDIRECTIONAL_LINK_DEFAULT_GAP = 16.2;

/**
 * Splits `from`/`to` into two perpendicular-offset endpoint pairs so the
 * forward and backward edges render as visually distinct parallel lines.
 * Only possible when both endpoints resolve to literal coordinates; node/ref
 * anchored endpoints fall back to coincident forward/backward pairs (still
 * correct: two opposite-direction arrows sharing the same anchors).
 */
function bidirectionalOffsets(
  from: ConnectorEndpointIr,
  to: ConnectorEndpointIr,
  gap: number,
):
  | Readonly<{
      forward: readonly [ConnectorEndpointIr, ConnectorEndpointIr];
      backward: readonly [ConnectorEndpointIr, ConnectorEndpointIr];
    }>
  | undefined {
  if (
    gap === 0 ||
    typeof from.x !== "number" ||
    typeof from.y !== "number" ||
    typeof to.x !== "number" ||
    typeof to.y !== "number"
  ) {
    return undefined;
  }
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  if (length === 0) {
    return undefined;
  }
  const nx = (-dy / length) * (gap / 2);
  const ny = (dx / length) * (gap / 2);
  return {
    forward: [
      { ...from, x: from.x + nx, y: from.y + ny },
      { ...to, x: to.x + nx, y: to.y + ny },
    ],
    backward: [
      { ...to, x: to.x - nx, y: to.y - ny },
      { ...from, x: from.x - nx, y: from.y - ny },
    ],
  };
}

const bidirectionalLinkFactory: SdkComponentFactory = (props, _slots, context) => {
  const mode = linkEdgeMode(stringProp(props, "mode"));
  const capabilityId = mode === "route" ? "core.route" : "core.connector";

  const from = endpointFromJson(props.from);
  if (from === undefined) {
    return fail(
      context,
      "SDK_BIDIRECTIONAL_LINK_FROM_REQUIRED",
      `sdk.bidirectionalLink "${context.instanceId}" requires a resolvable "from" endpoint.`,
      'Provide {nodeId}, {x, y}, a bare string node id, or {ref: "instance.port"}.',
    );
  }
  const to = endpointFromJson(props.to);
  if (to === undefined) {
    return fail(
      context,
      "SDK_BIDIRECTIONAL_LINK_TO_REQUIRED",
      `sdk.bidirectionalLink "${context.instanceId}" requires a resolvable "to" endpoint.`,
      'Provide {nodeId}, {x, y}, a bare string node id, or {ref: "instance.port"}.',
    );
  }

  const gap = numberProp(props, "gap", BIDIRECTIONAL_LINK_DEFAULT_GAP);
  const offsets = bidirectionalOffsets(from, to, gap);
  const [forwardFrom, forwardTo] = offsets?.forward ?? [from, to];
  const [backwardFrom, backwardTo] = offsets?.backward ?? [to, from];

  const via = pointFromJson(props.via);
  const axis = axisFromJson(props.axis);
  const styleOverride = isJsonRecord(props.style) ? scalarStyleFromJson(props.style) : {};
  const baseStyle: Record<string, StyleValueIr> = {
    fill: "none",
    markerEnd: "arrow",
    ...(mode === "route" ? { route: "elbow" } : {}),
    ...styleOverride,
  };
  const label = stringProp(props, "label") ?? "bidirectional link";
  const description = stringProp(props, "description");

  const forwardId = nodeId(context, "forward");
  const backwardId = nodeId(context, "backward");

  const forwardNode: RenderNodeIr = withOrigin(
    {
      kind: "connector",
      id: forwardId,
      capabilityId,
      geometry: geometryFromEndpointsOrProps(props, forwardFrom, forwardTo),
      style: baseStyle,
      accessibility: { label: `${label} forward`, ...(description !== undefined ? { description } : {}) },
      fallback: label,
      sourceMap: context.sourceMap,
      from: forwardFrom,
      to: forwardTo,
      ...(via !== undefined ? { via } : {}),
      ...(axis !== undefined ? { axis } : {}),
    },
    context,
    "sdk.bidirectionalLink",
    "forward",
  );

  const backwardNode: RenderNodeIr = withOrigin(
    {
      kind: "connector",
      id: backwardId,
      capabilityId,
      geometry: geometryFromEndpointsOrProps(props, backwardFrom, backwardTo),
      style: baseStyle,
      accessibility: { label: `${label} backward`, ...(description !== undefined ? { description } : {}) },
      fallback: label,
      sourceMap: context.sourceMap,
      from: backwardFrom,
      to: backwardTo,
      ...(via !== undefined ? { via } : {}),
      ...(axis !== undefined ? { axis } : {}),
    },
    context,
    "sdk.bidirectionalLink",
    "backward",
  );

  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: forwardId },
    forward: { nodeId: forwardId },
    backward: { nodeId: backwardId },
    source: from,
    target: to,
  };
  return ok([forwardNode, backwardNode], ports, {
    enter: [forwardId, backwardId],
    draw: [forwardId, backwardId],
    trace: [forwardId, backwardId],
  });
};

export const SDK_BIDIRECTIONAL_LINK: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.bidirectionalLink",
    capabilityId: "core.connector",
    props: {
      id: { type: "string", required: true },
      mode: { type: "string", required: false, default: "connector" },
      from: { type: "endpoint", required: true },
      to: { type: "endpoint", required: true },
      gap: { type: "number", required: false, default: BIDIRECTIONAL_LINK_DEFAULT_GAP },
      via: { type: "object", required: false },
      axis: { type: "string", required: false },
      label: { type: "string", required: false },
      description: { type: "string", required: false },
      style: { type: "object", required: false },
      ...GEOMETRY_PROPS,
    },
  }),
  factory: bidirectionalLinkFactory,
  actions: TOPOLOGY_ACTIONS,
};

// --- sdk.stateTransition -----------------------------------------------------------

const STATE_TRANSITION_DEFAULT_OPACITY = 0.85;
const STATE_TRANSITION_DEFAULT_STROKE_WIDTH = 4.73;
const STATE_TRANSITION_LABEL_WIDTH = 259.2;
const STATE_TRANSITION_LABEL_HEIGHT = 43.2;

function midpointOf(a: ConnectorEndpointIr, b: ConnectorEndpointIr): PointIr | undefined {
  if (typeof a.x === "number" && typeof a.y === "number" && typeof b.x === "number" && typeof b.y === "number") {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }
  return undefined;
}

const stateTransitionFactory: SdkComponentFactory = (props, slots, context) => {
  const fromFragment = slotFragments(slots, "from")[0];
  if (fromFragment === undefined) {
    return fail(
      context,
      "SDK_STATE_TRANSITION_FROM_REQUIRED",
      `sdk.stateTransition "${context.instanceId}" requires a "from" slot entry.`,
      "Provide exactly one component invocation in the from slot.",
    );
  }
  const toFragment = slotFragments(slots, "to")[0];
  if (toFragment === undefined) {
    return fail(
      context,
      "SDK_STATE_TRANSITION_TO_REQUIRED",
      `sdk.stateTransition "${context.instanceId}" requires a "to" slot entry.`,
      "Provide exactly one component invocation in the to slot.",
    );
  }

  const mode = linkEdgeMode(stringProp(props, "mode"));
  const capabilityId = mode === "route" ? "core.route" : "core.connector";
  const fromAnchor = portOrRootEndpoint(fromFragment, ["exit", "output", "self"], "e");
  if (fromAnchor === undefined) {
    return fail(
      context,
      "SDK_STATE_TRANSITION_FROM_ENDPOINT_UNRESOLVED",
      `sdk.stateTransition "${context.instanceId}" from slot entry has no resolvable port or root id.`,
      "Ensure the from slot's component invocation expands to at least one root node.",
    );
  }
  const toAnchor = portOrRootEndpoint(toFragment, ["entry", "input", "self"], "w");
  if (toAnchor === undefined) {
    return fail(
      context,
      "SDK_STATE_TRANSITION_TO_ENDPOINT_UNRESOLVED",
      `sdk.stateTransition "${context.instanceId}" to slot entry has no resolvable port or root id.`,
      "Ensure the to slot's component invocation expands to at least one root node.",
    );
  }

  const edgeId = nodeId(context, "edge");
  const edgeLabel = stringProp(props, "label") ?? "state transition";
  const styleOverride = isJsonRecord(props.style) ? scalarStyleFromJson(props.style) : {};
  const edgeStyle: Record<string, StyleValueIr> = {
    fill: "none",
    motion: "signal",
    markerEnd: "arrow",
    opacity: numberProp(props, "opacity", STATE_TRANSITION_DEFAULT_OPACITY),
    strokeWidth: numberProp(props, "strokeWidth", STATE_TRANSITION_DEFAULT_STROKE_WIDTH),
    ...(mode === "route" ? { route: "elbow" } : {}),
    ...styleOverride,
  };

  const edge: RenderNodeIr = withOrigin(
    {
      kind: "connector",
      id: edgeId,
      capabilityId,
      geometry: geometryFromEndpointsOrProps(props, fromAnchor, toAnchor),
      style: edgeStyle,
      accessibility: { label: edgeLabel },
      fallback: edgeLabel,
      sourceMap: context.sourceMap,
      from: fromAnchor,
      to: toAnchor,
    },
    context,
    "sdk.stateTransition",
    "edge",
  );

  const fromRoots = flattenRoots([fromFragment]);
  const toRoots = flattenRoots([toFragment]);
  const roots: RenderNodeIr[] = [...fromRoots, ...toRoots, edge];

  const fromRootId = primaryRootId(fromFragment);
  const toRootId = primaryRootId(toFragment);
  const ports: Record<string, ConnectorEndpointIr> = {
    self: { nodeId: edgeId },
    edge: { nodeId: edgeId },
    ...(fromRootId !== undefined ? { from: { nodeId: fromRootId } } : {}),
    ...(toRootId !== undefined ? { to: { nodeId: toRootId } } : {}),
  };

  const triggerText = stringProp(props, "trigger");
  let labelId: string | undefined;
  if (triggerText !== undefined) {
    const mid = midpointOf(fromAnchor, toAnchor) ?? { x: 0, y: 0 };
    labelId = nodeId(context, "trigger");
    roots.push(
      withOrigin(
        {
          kind: "text",
          id: labelId,
          capabilityId: "core.text",
          geometry: {
            x: mid.x - STATE_TRANSITION_LABEL_WIDTH / 2,
            y: mid.y - STATE_TRANSITION_LABEL_HEIGHT / 2,
            width: STATE_TRANSITION_LABEL_WIDTH,
            height: STATE_TRANSITION_LABEL_HEIGHT,
          },
          style: { fontSize: 27, textAnchor: "middle" },
          accessibility: { label: triggerText },
          fallback: triggerText,
          sourceMap: context.sourceMap,
          text: triggerText,
        },
        context,
        "sdk.stateTransition",
        "trigger",
      ),
    );
    ports.trigger = { nodeId: labelId };
  }

  const pulseTargets = [fromRootId, toRootId].filter((value): value is string => value !== undefined);
  const fadeTargets = labelId !== undefined ? [edgeId, labelId] : [edgeId];

  return ok(roots, ports, {
    enter: roots.map((root) => root.id),
    pulse: pulseTargets,
    trace: [edgeId],
    fade: fadeTargets,
  });
};

export const SDK_STATE_TRANSITION: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.stateTransition",
    capabilityId: "core.group",
    props: {
      id: { type: "string", required: true },
      mode: { type: "string", required: false, default: "connector" },
      trigger: { type: "string", required: false },
      label: { type: "string", required: false },
      opacity: { type: "number", required: false, default: STATE_TRANSITION_DEFAULT_OPACITY },
      strokeWidth: {
        type: "number",
        required: false,
        default: STATE_TRANSITION_DEFAULT_STROKE_WIDTH,
      },
      style: { type: "object", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: {
      from: { accepts: "sdk.*", required: true },
      to: { accepts: "sdk.*", required: true },
    },
  }),
  factory: stateTransitionFactory,
  actions: MOTION_ACTIONS,
};

/** Generic SDK composite pack, ready for `registry.ts` to splice in place of stubs. */
export const GENERIC_COMPOSITE_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  SDK_MATRIX,
  SDK_LAYER_STACK,
  SDK_HUB_SPOKE,
  SDK_TREE,
  SDK_BIDIRECTIONAL_LINK,
  SDK_STATE_TRANSITION,
];
