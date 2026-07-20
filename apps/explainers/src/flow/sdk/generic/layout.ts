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
 * Every factory emits one native semantic layout node. The shared capability
 * layout registry computes child placement and intrinsic bounds at render and
 * verification time.
 */

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import type { Result } from "../../schema/diagnostic.js";
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

function booleanProp(props: PropRecord, key: string, fallback = false): boolean {
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

const MANAGED_LAYOUT_PROPS: Readonly<Record<
  string,
  ComponentPropDescriptor
>> = {
  padding: { type: "number", required: false, default: 0 },
  align: { type: "string", required: false, default: "start" },
  justify: { type: "string", required: false, default: "start" },
  fixedWidth: { type: "boolean", required: false, default: false },
  fixedHeight: { type: "boolean", required: false, default: false },
};

function managedStyle(
  props: PropRecord,
  extras: Readonly<Record<string, StyleValueIr>> = {},
): Readonly<Record<string, StyleValueIr>> {
  return {
    coordinateSpace: "local",
    padding: Math.max(0, numberProp(props, "padding", 0)),
    align: stringProp(props, "align") ?? "start",
    justify: stringProp(props, "justify") ?? "start",
    fixedWidth: booleanProp(props, "fixedWidth"),
    fixedHeight: booleanProp(props, "fixedHeight"),
    ...extras,
  };
}

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
      style: managedStyle(props, { direction, gap }),
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
      ...MANAGED_LAYOUT_PROPS,
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
      style: managedStyle(props, { cols, gap }),
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
      ...MANAGED_LAYOUT_PROPS,
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
      style: managedStyle(props, { direction, gap }),
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
      ...MANAGED_LAYOUT_PROPS,
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: railFactory,
  actions: LAYOUT_ACTIONS,
};

function managedContainerFactory(
  capabilityId: "layout.overlay" | "layout.frame",
  componentId: "sdk.overlay" | "sdk.frame",
): SdkComponentFactory {
  return (props, slots, context) => {
    const children = slotFragments(slots, "children");
    const roots = flattenRoots(children);
    const id = nodeId(context);
    const title = stringProp(props, "title");
    const detail = stringProp(props, "detail");
    const label =
      stringProp(props, "label") ??
      title ??
      (capabilityId === "layout.frame" ? "frame" : "overlay");
    const group: RenderNodeIr = withOrigin(
      {
        kind: "group",
        id,
        capabilityId,
        geometry: geometryFromProps(props),
        style: managedStyle(props, {
          gap: numberProp(props, "gap", capabilityId === "layout.frame" ? 12 : 0),
        }),
        props:
          capabilityId === "layout.frame"
            ? {
                ...(title === undefined ? {} : { title }),
                ...(detail === undefined ? {} : { detail }),
              }
            : {},
        accessibility: { label },
        fallback: label,
        sourceMap: context.sourceMap,
        children: roots,
      },
      context,
      componentId,
      "root",
    );
    const ports = {
      self: { nodeId: id },
      ...mergeChildPorts(children, "child"),
    };
    const actions: Partial<Record<SdkActionName, readonly string[]>> = {
      enter: [id],
    };
    if (roots.length > 0) {
      actions.stagger = roots.map((root) => root.id);
    }
    return ok([group], ports, actions);
  };
}

/** Intentional-overlap managed container. */
export const SDK_OVERLAY: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.overlay",
    capabilityId: "layout.overlay",
    props: {
      id: { type: "string", required: true },
      gap: { type: "number", required: false, default: 0 },
      label: { type: "string", required: false },
      ...MANAGED_LAYOUT_PROPS,
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: managedContainerFactory("layout.overlay", "sdk.overlay"),
  actions: LAYOUT_ACTIONS,
};

/** Titled content-safe managed container. */
export const SDK_FRAME: SdkComponentDefinition = {
  descriptor: createDescriptor({
    id: "sdk.frame",
    capabilityId: "layout.frame",
    props: {
      id: { type: "string", required: true },
      title: { type: "string", required: true },
      detail: { type: "string", required: false },
      gap: { type: "number", required: false, default: 12 },
      label: { type: "string", required: false },
      ...MANAGED_LAYOUT_PROPS,
      ...GEOMETRY_PROPS,
    },
    slots: CHILDREN_SLOT,
  }),
  factory: managedContainerFactory("layout.frame", "sdk.frame"),
  actions: LAYOUT_ACTIONS,
};

// --- shared native semantic layout node ------------------------------------

function semanticLayoutNode(
  context: SdkExpansionContext,
  capability: string,
  props: PropRecord,
  children: readonly RenderNodeIr[],
  label: string,
  componentId: string,
): RenderNodeIr {
  const gap = numberProp(props, "gap", 0);
  return withOrigin(
    {
      kind: "group",
      id: nodeId(context),
      capabilityId: capability,
      geometry: geometryFromProps(props),
      style: { coordinateSpace: "local", gap },
      props,
      accessibility: { label },
      fallback: label,
      sourceMap: context.sourceMap,
      children,
    },
    context,
    componentId,
    "root",
  );
}

// --- sdk.lane --------------------------------------------------------------

const laneFactory: SdkComponentFactory = (props, slots, context) => {
  const children = slotFragments(slots, "children");
  const childRoots = flattenRoots(children);
  const label = stringProp(props, "title") ?? stringProp(props, "text") ?? "lane";
  const node = semanticLayoutNode(
    context,
    "core.lane",
    { ...props, gap: numberProp(props, "gap", 8) },
    childRoots,
    label,
    "sdk.lane",
  );
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
  const node = semanticLayoutNode(
    context,
    "core.swimlane",
    { ...props, gap: numberProp(props, "gap", 8) },
    childRoots,
    label,
    "sdk.swimlane",
  );
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
  const node = semanticLayoutNode(
    context,
    "core.band",
    props,
    childRoots,
    label,
    "sdk.band",
  );
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
  const authoredStepRoots = flattenRoots(stepFragments);
  const label = "stepper";
  const id = nodeId(context);
  const stepRoots =
    stepTexts.length > 0
      ? stepTexts.map(
          (text, index): RenderNodeIr => ({
            kind: "group",
            id: `${id}-step-${index}`,
            capabilityId: "core.step",
            // Semantic steps remain indexable/verifiable Scene nodes. Native
            // stepper layout replaces x/width at render time from label text.
            geometry: { x: 0, y: 0, width: 72, height: 26 },
            style: {},
            props: { label: text, index },
            accessibility: { label: text },
            fallback: text,
            sourceMap: context.sourceMap,
            children: [],
          }),
        )
      : authoredStepRoots;
  const node = semanticLayoutNode(
    context,
    "core.stepper",
    { ...props, gap: numberProp(props, "gap", 12) },
    stepRoots,
    label,
    "sdk.stepper",
  );
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
  SDK_OVERLAY,
  SDK_FRAME,
  SDK_LANE,
  SDK_SWIMLANE,
  SDK_BAND,
  SDK_STEPPER,
];
