/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! AIPerf domain SDK pack: request execution composites.
//!
//! `aiperf.requestPipeline`, `aiperf.segmentPool`, `aiperf.warmupHandoff`,
//! `aiperf.veloEnvelope`, and `aiperf.phaseLifecycle` compose deterministic
//! Scene IR fragments describing per-worker request staging, content-addressed
//! segment storage, warmup-to-profiling handoff, cross-host Velo transport
//! envelopes, and phase lifecycle sequencing. Components carry no
//! deck-specific prose or fixed slide ids; callers supply labels and theme
//! roles as props.
//!
//! Labeled stages use native semantic `core.chip` Scene IR, while the segment
//! pool and Velo envelope use semantic `core.panel` containers.

import type { ComponentPropDescriptor } from "../../schema/component-descriptor.js";
import { diagnostic, type Result } from "../../schema/diagnostic.js";
import type {
  ConnectorEndpointIr,
  ConnectorNodeIr,
  GeometryIr,
  GroupNodeIr,
  RenderNodeIr,
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
  options?: Readonly<{
    capabilityId?: string;
    style?: Readonly<Record<string, StyleValueIr>>;
  }>,
): ConnectorNodeIr {
  return {
    kind: "connector",
    id,
    capabilityId: options?.capabilityId ?? "core.connector",
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: {
      stroke: themeRole("line.structural"),
      strokeWidth: 1.5,
      fill: "none",
      markerEnd: "arrow",
      ...options?.style,
    },
    accessibility: { label },
    fallback: label,
    sourceMap,
    from,
    to,
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

function semanticPanel(args: {
  id: string;
  label: string;
  geometry: GeometryIr;
  sourceMap: SourceRange;
  surfaceRole: ThemeRole;
  inkRole: ThemeRole;
  lineRole: ThemeRole;
  children: readonly RenderNodeIr[];
  dashed?: boolean;
}): GroupNodeIr {
  return {
    kind: "group",
    id: args.id,
    capabilityId: "core.panel",
    geometry: args.geometry,
    style: {
      coordinateSpace: "local",
      fill: themeRole(args.surfaceRole),
      stroke: themeRole(args.lineRole),
      strokeWidth: args.dashed === true ? 1.4 : 1.2,
      radius: 10,
      ...(args.dashed === true ? { strokeDasharray: "4 3" } : {}),
    },
    props: { title: args.label, inkRole: args.inkRole },
    accessibility: { label: args.label },
    fallback: args.label,
    sourceMap: args.sourceMap,
    children: args.children,
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

// --- aiperf.requestPipeline --------------------------------------------------

const STAGE_WIDTH = 116;
const STAGE_HEIGHT = 44;
const STAGE_GAP = 40;

/**
 * Ordered per-request stages (schedule, admission, transport, capture) owned
 * by a self-contained worker or sub-cell, connected in sequence.
 */
function requestPipelineFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.requestPipeline";
  const stageLabels = readStringArray(props, "stageLabels", [
    "Schedule",
    "Admission",
    "Transport",
    "Capture",
  ]);
  const failure = requireLabels(stageLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");

  const origin = authoredOrigin(props);
  const { instanceId, sourceMap } = context;
  const width = stageLabels.length * STAGE_WIDTH + (stageLabels.length - 1) * STAGE_GAP;
  const height = STAGE_HEIGHT;

  const stages = stageLabels.map((label, index) =>
    semanticChip({
      id: nodeId(instanceId, `stage-${index}`),
      label,
      x: index * (STAGE_WIDTH + STAGE_GAP),
      y: 0,
      width: STAGE_WIDTH,
      height: STAGE_HEIGHT,
      sourceMap,
      surfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const edges: ConnectorNodeIr[] = [];
  for (let index = 0; index < stages.length - 1; index += 1) {
    const from = stages[index];
    const to = stages[index + 1];
    if (from === undefined || to === undefined) {
      continue;
    }
    edges.push(
      connectorNode(
        nodeId(instanceId, `edge-${index}`),
        { nodeId: from.id, anchor: "e" },
        { nodeId: to.id, anchor: "w" },
        sourceMap,
        `${from.accessibility.label} to ${to.accessibility.label}`,
      ),
    );
  }

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    "Request pipeline",
    [...stages, ...edges],
  );

  const ports: Record<string, ConnectorEndpointIr> = {};
  stages.forEach((stage, index) => {
    ports[`stage[${index}]`] = { nodeId: stage.id, anchor: "e" };
  });
  const firstStage = stages[0];
  const lastStage = stages[stages.length - 1];
  if (firstStage !== undefined) {
    ports.input = { nodeId: firstStage.id, anchor: "w" };
  }
  if (lastStage !== undefined) {
    ports.output = { nodeId: lastStage.id, anchor: "e" };
  }

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: stages.map((stage) => stage.id),
        draw: edges.map((edge) => edge.id),
        trace: edges.map((edge) => edge.id),
        emphasis: stages.map((stage) => stage.id),
      },
    },
    diagnostics: [],
  };
}

// --- aiperf.segmentPool -------------------------------------------------------

const SEGMENT_WIDTH = 96;
const SEGMENT_HEIGHT = 40;
const SEGMENT_GAP = 16;
const POOL_PAD = 16;
const POOL_HEADER = 28;

/**
 * Content-addressed segment store: a pool container holding BLAKE3-identified
 * segments materialized from stored serialized bytes.
 */
function segmentPoolFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.segmentPool";
  const segmentLabels = readStringArray(props, "segmentLabels", [
    "Segment A",
    "Segment B",
    "Segment C",
  ]);
  const failure = requireLabels(segmentLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const poolLabel = readString(props, "poolLabel", "Segment Store");
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.raised");
  const segmentSurfaceRole = readThemeRole(props, "segmentSurfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");

  const origin = authoredOrigin(props);
  const { instanceId, sourceMap } = context;
  const rowWidth =
    segmentLabels.length * SEGMENT_WIDTH + (segmentLabels.length - 1) * SEGMENT_GAP;
  const width = rowWidth + POOL_PAD * 2;
  const height = POOL_HEADER + SEGMENT_HEIGHT + POOL_PAD * 2;

  const poolId = instanceId;
  const segments = segmentLabels.map((label, index) =>
    semanticChip({
      id: nodeId(instanceId, `segment-${index}`),
      label,
      x: POOL_PAD + index * (SEGMENT_WIDTH + SEGMENT_GAP),
      y: POOL_HEADER + POOL_PAD,
      width: SEGMENT_WIDTH,
      height: SEGMENT_HEIGHT,
      sourceMap,
      surfaceRole: segmentSurfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const root = semanticPanel({
    id: poolId,
    label: poolLabel,
    geometry: { x: origin.x, y: origin.y, width, height },
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
    children: segments,
  });

  const ports: Record<string, ConnectorEndpointIr> = {
    pool: { nodeId: poolId, anchor: "w" },
  };
  segments.forEach((segment, index) => {
    ports[`segment[${index}]`] = { nodeId: segment.id, anchor: "s" };
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: [poolId, ...segments.map((segment) => segment.id)],
        draw: [],
        trace: [],
        emphasis: segments.map((segment) => segment.id),
      },
    },
    diagnostics: [],
  };
}

// --- aiperf.warmupHandoff ------------------------------------------------------

const HANDOFF_WIDTH = 132;
const HANDOFF_HEIGHT = 48;
const HANDOFF_GAP = 72;

/** Warmup phase handing off to the profiling phase across a single edge. */
function warmupHandoffFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const fromLabel = readString(props, "fromLabel", "Warmup");
  const toLabel = readString(props, "toLabel", "Profiling");
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");

  const origin = authoredOrigin(props);
  const { instanceId, sourceMap } = context;
  const width = HANDOFF_WIDTH * 2 + HANDOFF_GAP;
  const height = HANDOFF_HEIGHT;

  const fromId = nodeId(instanceId, "from");
  const toId = nodeId(instanceId, "to");
  const fromBox = semanticChip({
    id: fromId,
    label: fromLabel,
    x: 0,
    y: 0,
    width: HANDOFF_WIDTH,
    height: HANDOFF_HEIGHT,
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
  });
  const toBox = semanticChip({
    id: toId,
    label: toLabel,
    x: HANDOFF_WIDTH + HANDOFF_GAP,
    y: 0,
    width: HANDOFF_WIDTH,
    height: HANDOFF_HEIGHT,
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
  });
  const handoffId = nodeId(instanceId, "handoff");
  const handoff = connectorNode(
    handoffId,
    { nodeId: fromId, anchor: "e" },
    { nodeId: toId, anchor: "w" },
    sourceMap,
    `${fromLabel} to ${toLabel} handoff`,
  );

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    `${fromLabel} to ${toLabel} handoff`,
    [fromBox, handoff, toBox],
  );

  return {
    ok: true,
    value: {
      roots: [root],
      ports: {
        from: { nodeId: fromId, anchor: "w" },
        to: { nodeId: toId, anchor: "e" },
      },
      actions: {
        enter: [fromId, toId],
        draw: [handoffId],
        trace: [handoffId],
        emphasis: [fromId, toId],
      },
    },
    diagnostics: [],
  };
}

// --- aiperf.veloEnvelope --------------------------------------------------------

const ENVELOPE_PAD = 14;
const ENVELOPE_TITLE_BAND = 20;
const PAYLOAD_WIDTH = 108;
const PAYLOAD_HEIGHT = 40;

/**
 * Cross-host cell transport envelope carried over the Velo framework,
 * wrapping a payload for controller/cell cellular protocol exchange.
 */
function veloEnvelopeFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const envelopeLabel = readString(props, "envelopeLabel", "Velo Envelope");
  const payloadLabel = readString(props, "payloadLabel", "Payload");
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.raised");
  const payloadSurfaceRole = readThemeRole(props, "payloadSurfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");

  const origin = authoredOrigin(props);
  const { instanceId, sourceMap } = context;
  const width = PAYLOAD_WIDTH + ENVELOPE_PAD * 2;
  const height = PAYLOAD_HEIGHT + ENVELOPE_PAD * 2 + ENVELOPE_TITLE_BAND;

  const envelopeId = instanceId;
  const payloadId = nodeId(instanceId, "payload");
  const payload = semanticChip({
    id: payloadId,
    label: payloadLabel,
    x: ENVELOPE_PAD,
    y: ENVELOPE_TITLE_BAND + ENVELOPE_PAD - 6,
    width: PAYLOAD_WIDTH,
    height: PAYLOAD_HEIGHT,
    sourceMap,
    surfaceRole: payloadSurfaceRole,
    inkRole,
    lineRole,
  });

  const root = semanticPanel({
    id: envelopeId,
    label: envelopeLabel,
    geometry: { x: origin.x, y: origin.y, width, height },
    sourceMap,
    surfaceRole,
    inkRole,
    lineRole,
    children: [payload],
    dashed: true,
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports: {
        envelope: { nodeId: envelopeId, anchor: "w" },
        payload: { nodeId: payloadId, anchor: "center" },
      },
      actions: {
        enter: [envelopeId, payloadId],
        draw: [],
        trace: [],
        emphasis: [envelopeId, payloadId],
      },
    },
    diagnostics: [],
  };
}

// --- aiperf.phaseLifecycle -----------------------------------------------------

const PHASE_WIDTH = 108;
const PHASE_HEIGHT = 40;
const PHASE_GAP = 36;

/**
 * Warmup / profiling / grace / drain phase sequence, matching phase
 * orchestration's lifecycle policy, grace, cancellation, drain, and force
 * escalation stages.
 */
function phaseLifecycleFactory(
  props: Readonly<Record<string, JsonValue>>,
  _slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const componentId = "aiperf.phaseLifecycle";
  const phaseLabels = readStringArray(props, "phaseLabels", [
    "Warmup",
    "Profiling",
    "Grace",
    "Drain",
  ]);
  const failure = requireLabels(phaseLabels, componentId, context);
  if (failure !== undefined) {
    return failure;
  }
  const surfaceRole = readThemeRole(props, "surfaceRole", "surface.panel");
  const inkRole = readThemeRole(props, "inkRole", "ink.primary");
  const lineRole = readThemeRole(props, "lineRole", "line.structural");

  const origin = authoredOrigin(props);
  const { instanceId, sourceMap } = context;
  const width = phaseLabels.length * PHASE_WIDTH + (phaseLabels.length - 1) * PHASE_GAP;
  const height = PHASE_HEIGHT;

  const phases = phaseLabels.map((label, index) =>
    semanticChip({
      id: nodeId(instanceId, `phase-${index}`),
      label,
      x: index * (PHASE_WIDTH + PHASE_GAP),
      y: 0,
      width: PHASE_WIDTH,
      height: PHASE_HEIGHT,
      sourceMap,
      surfaceRole,
      inkRole,
      lineRole,
    }),
  );

  const transitions: ConnectorNodeIr[] = [];
  for (let index = 0; index < phases.length - 1; index += 1) {
    const from = phases[index];
    const to = phases[index + 1];
    if (from === undefined || to === undefined) {
      continue;
    }
    transitions.push(
      connectorNode(
        nodeId(instanceId, `transition-${index}`),
        { nodeId: from.id, anchor: "e" },
        { nodeId: to.id, anchor: "w" },
        sourceMap,
        `${from.accessibility.label} to ${to.accessibility.label} transition`,
        { capabilityId: "core.route", style: { route: "elbow" } },
      ),
    );
  }

  const root = groupNode(
    instanceId,
    { x: origin.x, y: origin.y, width, height },
    sourceMap,
    "Phase lifecycle",
    [...phases, ...transitions],
  );

  const ports: Record<string, ConnectorEndpointIr> = {};
  phases.forEach((phase, index) => {
    ports[`phase[${index}]`] = { nodeId: phase.id, anchor: "e" };
  });

  return {
    ok: true,
    value: {
      roots: [root],
      ports,
      actions: {
        enter: phases.map((phase) => phase.id),
        draw: transitions.map((transition) => transition.id),
        trace: transitions.map((transition) => transition.id),
        emphasis: phases.map((phase) => phase.id),
      },
    },
    diagnostics: [],
  };
}

/**
 * AIPerf execution pack: request pipeline, segment pool, warmup handoff,
 * Velo envelope, and phase lifecycle composites.
 */
export const AIPERF_EXECUTION_COMPONENTS: readonly SdkComponentDefinition[] = [
  defineAiperfComponent({
    id: "aiperf.requestPipeline",
    props: {
      stageLabels: stringArrayProp(["Schedule", "Admission", "Transport", "Capture"]),
      surfaceRole: stringProp("surface.panel"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
    ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: requestPipelineFactory,
  }),
  defineAiperfComponent({
    id: "aiperf.segmentPool",
    props: {
      poolLabel: stringProp("Segment Store"),
      segmentLabels: stringArrayProp(["Segment A", "Segment B", "Segment C"]),
      segmentSurfaceRole: stringProp("surface.panel"),
      surfaceRole: stringProp("surface.raised"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
    ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: segmentPoolFactory,
  }),
  defineAiperfComponent({
    id: "aiperf.warmupHandoff",
    props: {
      fromLabel: stringProp("Warmup"),
      toLabel: stringProp("Profiling"),
      surfaceRole: stringProp("surface.panel"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
    ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: warmupHandoffFactory,
  }),
  defineAiperfComponent({
    id: "aiperf.veloEnvelope",
    props: {
      envelopeLabel: stringProp("Velo Envelope"),
      payloadLabel: stringProp("Payload"),
      payloadSurfaceRole: stringProp("surface.panel"),
      surfaceRole: stringProp("surface.raised"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
    ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: veloEnvelopeFactory,
  }),
  defineAiperfComponent({
    id: "aiperf.phaseLifecycle",
    props: {
      phaseLabels: stringArrayProp(["Warmup", "Profiling", "Grace", "Drain"]),
      surfaceRole: stringProp("surface.panel"),
      inkRole: stringProp("ink.primary"),
      lineRole: stringProp("line.structural"),
    ...GEOMETRY_ORIGIN_PROPS,
    },
    actions: ["enter", "draw", "trace", "emphasis"],
    factory: phaseLifecycleFactory,
  }),
];
