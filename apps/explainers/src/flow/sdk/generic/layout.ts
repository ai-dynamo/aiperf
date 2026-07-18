/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Generic SDK layout factories: `sdk.stack`, `sdk.grid`, `sdk.rail`,
 * `sdk.lane`, `sdk.swimlane`, `sdk.band`, `sdk.stepper`.
 *
 * Each factory is a pure, deterministic composer over already-expanded child
 * `SceneFragment`s supplied through `slots`. `sdk.stack` / `sdk.grid` /
 * `sdk.rail` emit a first-class `layout.*` group whose children arrangement
 * is computed by `SceneRenderer` at render time (direction/cols/gap style),
 * so this module only needs to size and tag the group. `sdk.lane` /
 * `sdk.swimlane` / `sdk.band` / `sdk.stepper` reuse the existing
 * `desugarPackageNode` macro geometry so behavior matches the package-form
 * authoring these components replace.
 */

import { desugarPackageNode } from "../../compiler/desugar-scene-primitives.js";
import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
import type { ConnectorEndpointIr, GeometryIr, RenderNodeIr } from "../../schema/ir.js";
import type { JsonValue } from "../../schema/json-value.js";
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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

const GEOMETRY_PROPS: Readonly<Record<string, ComponentPropDescriptor>> = {
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: 0 },
  height: { type: "number", required: false, default: 0 },
};

const CHILDREN_SLOT: Readonly<Record<string, ComponentSlotDescriptor>> = {
  children: { accepts: "sdk.*", required: false },
};

// --- sdk.stack ---------------------------------------------------------

const stackFactory: SdkComponentFactory = (props, slots, context) => {
  const direction = stringProp(props, "direction") === "column" ? "column" : "row";
  const gap = numberProp(props, "gap", 12);
  const children = slotFragments(slots, "children");
  const roots = flattenRoots(children);
  const id = nodeId(context);
  const label = stringProp(props, "label") ?? `${direction} stack`;
  const group: RenderNodeIr = withOrigin(
    {
      kind: "group",
      id,
      capabilityId: "layout.stack",
      geometry: geometryFromProps(props),
      style: { direction, gap },
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      children: roots,
    },
    context,
    "sdk.stack",
    "root",
  );
  const ports = { self: { nodeId: id }, ...mergeChildPorts(children, "child") };
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (roots.length > 0) {
    actions.stagger = roots.map((root) => root.id);
  }
  return ok([group], ports, actions);
};

export const SDK_STACK: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.stack",
    capabilityId: "layout.stack",
    props: {
      id: { type: "string", required: true },
      direction: { type: "string", required: false, default: "row" },
      gap: { type: "number", required: false, default: 12 },
      label: { type: "string", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: stackFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.grid ------------------------------------------------------------

const gridFactory: SdkComponentFactory = (props, slots, context) => {
  const children = slotFragments(slots, "children");
  const roots = flattenRoots(children);
  const cols = Math.max(
    1,
    Math.round(
      numberProp(
        props,
        "cols",
        Math.max(1, Math.ceil(Math.sqrt(Math.max(roots.length, 1)))),
      ),
    ),
  );
  const gap = numberProp(props, "gap", 12);
  const id = nodeId(context);
  const label = stringProp(props, "label") ?? "grid";
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
    "sdk.grid",
    "root",
  );
  const ports = { self: { nodeId: id }, ...mergeChildPorts(children, "cell") };
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (roots.length > 0) {
    actions.stagger = roots.map((root) => root.id);
  }
  return ok([group], ports, actions);
};

export const SDK_GRID: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.grid",
    capabilityId: "layout.grid",
    props: {
      id: { type: "string", required: true },
      cols: { type: "number", required: false },
      gap: { type: "number", required: false, default: 12 },
      label: { type: "string", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: gridFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.rail --------------------------------------------------------------

const railFactory: SdkComponentFactory = (props, slots, context) => {
  const direction = stringProp(props, "direction") === "column" ? "column" : "row";
  const gap = numberProp(props, "gap", 12);
  const children = slotFragments(slots, "children");
  const roots = flattenRoots(children);
  const id = nodeId(context);
  const label = stringProp(props, "label") ?? `${direction} rail`;
  const group: RenderNodeIr = withOrigin(
    {
      kind: "group",
      id,
      capabilityId: "layout.rail",
      geometry: geometryFromProps(props),
      style: { direction, gap },
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      children: roots,
    },
    context,
    "sdk.rail",
    "root",
  );
  const ports = { self: { nodeId: id }, ...mergeChildPorts(children, "child") };
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (roots.length > 0) {
    actions.stagger = roots.map((root) => root.id);
  }
  return ok([group], ports, actions);
};

export const SDK_RAIL: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.rail",
    capabilityId: "layout.rail",
    props: {
      id: { type: "string", required: true },
      direction: { type: "string", required: false, default: "row" },
      gap: { type: "number", required: false, default: 12 },
      label: { type: "string", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: railFactory,
  actions: LAYOUT_ACTIONS,
};

// --- shared desugar-macro reuse (lane / swimlane / band / stepper) --------

/**
 * Builds a pseudo package-form record from authored props so the existing
 * `desugarPackageNode` geometry macros (`core.lane` / `core.swimlane` /
 * `core.band` / `core.stepper`) can be reused verbatim for SDK expansion.
 */
function pseudoPackageNode(props: PropRecord): Record<string, unknown> {
  return { ...props, geometry: geometryFromProps(props) };
}

function desugarOrFail(
  context: SdkExpansionContext,
  capability: string,
  props: PropRecord,
  children: readonly RenderNodeIr[],
  label: string,
): Result<RenderNodeIr> {
  const id = nodeId(context);
  try {
    const node = desugarPackageNode(pseudoPackageNode(props), {
      id,
      capability,
      children,
      label,
      fallback: label,
    });
    if (node === undefined) {
      return {
        ok: false,
        diagnostics: [
          diagnostic(
            "SDK_LAYOUT_DESUGAR_FAILED",
            "error",
            `"${capability}" macro expansion returned no node for instance "${id}".`,
            context.sourceMap,
          ),
        ],
      };
    }
    return { ok: true, value: node, diagnostics: [] };
  } catch (error) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "SDK_LAYOUT_INVALID",
          "error",
          `"${capability}" instance "${id}" is invalid: ${errorMessage(error)}`,
          context.sourceMap,
        ),
      ],
    };
  }
}

// --- sdk.lane --------------------------------------------------------------

const laneFactory: SdkComponentFactory = (props, slots, context) => {
  const children = slotFragments(slots, "children");
  const childRoots = flattenRoots(children);
  const label = stringProp(props, "title") ?? stringProp(props, "text") ?? "lane";
  const result = desugarOrFail(context, "core.lane", props, childRoots, label);
  if (!result.ok) {
    return result;
  }
  const node = withOrigin(result.value, context, "sdk.lane", "root");
  const id = nodeId(context);
  const ports = { self: { nodeId: id }, ...mergeChildPorts(children, "child") };
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (childRoots.length > 0) {
    actions.stagger = childRoots.map((root) => root.id);
  }
  return ok([node], ports, actions);
};

export const SDK_LANE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.lane",
    capabilityId: "core.lane",
    props: {
      id: { type: "string", required: true },
      title: { type: "string", required: false },
      gap: { type: "number", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: laneFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.swimlane ------------------------------------------------------------

const swimlaneFactory: SdkComponentFactory = (props, slots, context) => {
  const rowFragments = slotFragments(slots, "rows");
  const children = rowFragments.length > 0 ? rowFragments : slotFragments(slots, "children");
  const childRoots = flattenRoots(children);
  const label = stringProp(props, "title") ?? "swimlane";
  const result = desugarOrFail(context, "core.swimlane", props, childRoots, label);
  if (!result.ok) {
    return result;
  }
  const node = withOrigin(result.value, context, "sdk.swimlane", "root");
  const id = nodeId(context);
  const ports = { self: { nodeId: id }, ...mergeChildPorts(children, "row") };
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (childRoots.length > 0) {
    actions.stagger = childRoots.map((root) => root.id);
  }
  return ok([node], ports, actions);
};

export const SDK_SWIMLANE: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.swimlane",
    capabilityId: "core.swimlane",
    props: {
      id: { type: "string", required: true },
      gap: { type: "number", required: false, default: 8 },
      labelWidth: { type: "number", required: false, default: 72 },
      labels: { type: "array", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: {
      rows: { accepts: "sdk.*", required: false },
      children: { accepts: "sdk.*", required: false },
    },
  }),
  factory: swimlaneFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.band --------------------------------------------------------------

const bandFactory: SdkComponentFactory = (props, slots, context) => {
  const children = slotFragments(slots, "children");
  const childRoots = flattenRoots(children);
  const label = stringProp(props, "title") ?? stringProp(props, "text") ?? "band";
  const result = desugarOrFail(context, "core.band", props, childRoots, label);
  if (!result.ok) {
    return result;
  }
  const node = withOrigin(result.value, context, "sdk.band", "root");
  const id = nodeId(context);
  const ports = { self: { nodeId: id }, ...mergeChildPorts(children, "child") };
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (childRoots.length > 0) {
    actions.stagger = childRoots.map((root) => root.id);
  }
  return ok([node], ports, actions);
};

export const SDK_BAND: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.band",
    capabilityId: "core.band",
    props: {
      id: { type: "string", required: true },
      title: { type: "string", required: false },
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: bandFactory,
  actions: LAYOUT_ACTIONS,
};

// --- sdk.stepper -------------------------------------------------------------

function stringArrayProp(props: PropRecord, key: string): readonly string[] {
  const value = props[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0,
  );
}

const stepperFactory: SdkComponentFactory = (props, slots, context) => {
  const stepTexts = stringArrayProp(props, "steps");
  const stepFragments = slotFragments(slots, "steps").length > 0
    ? slotFragments(slots, "steps")
    : slotFragments(slots, "children");
  const stepRoots = flattenRoots(stepFragments);
  const label = "stepper";
  const result = desugarOrFail(context, "core.stepper", props, stepRoots, label);
  if (!result.ok) {
    return result;
  }
  const node = withOrigin(result.value, context, "sdk.stepper", "root");
  const id = nodeId(context);
  const stepIds =
    stepTexts.length > 0
      ? stepTexts.map((_, index) => `${id}-step-${index}`)
      : stepRoots.map((root) => root.id);
  const ports: Record<string, ConnectorEndpointIr> = { self: { nodeId: id } };
  stepIds.forEach((stepId, index) => {
    ports[`step[${index}]`] = { nodeId: stepId };
  });
  const actions: Partial<Record<SdkActionName, readonly string[]>> = { enter: [id] };
  if (stepIds.length > 0) {
    actions.stagger = stepIds;
  }
  return ok([node], ports, actions);
};

export const SDK_STEPPER: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.stepper",
    capabilityId: "core.stepper",
    props: {
      id: { type: "string", required: true },
      steps: { type: "array", required: false },
      linked: { type: "boolean", required: false, default: false },
      gap: { type: "number", required: false, default: 12 },
      ...GEOMETRY_PROPS,
    },
    slots: {
      steps: { accepts: "sdk.*", required: false },
      children: { accepts: "sdk.*", required: false },
    },
  }),
  factory: stepperFactory,
  actions: LAYOUT_ACTIONS,
};

/** Generic SDK layout pack, ready for `registry.ts` to splice in place of stubs. */
export const GENERIC_LAYOUT_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  SDK_STACK,
  SDK_GRID,
  SDK_RAIL,
  SDK_LANE,
  SDK_SWIMLANE,
  SDK_BAND,
  SDK_STEPPER,
];
