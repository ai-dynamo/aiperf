/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generic SDK chrome and content factories.
//!
//! `sdk.header` / `sdk.panel` / `sdk.card` / `sdk.chip` / `sdk.note` /
//! `sdk.label` / `sdk.legend` / `sdk.callout` / `sdk.divider` / `sdk.bracket`
//! emit native semantic Scene IR fragments. Renderer-owned chrome and text are
//! retained as semantic props rather than serialized visual-only children.
//! Every factory is pure: no DOM, React, network, wall clock, or mutable
//! global state. Generated node ids are seeded from `context.instanceId`
//! (`${instanceId}` for the fragment root, `${instanceId}__role` for
//! generated children) so expansion is stable across repeated calls.
//!
//! This module is deliberately self-contained so it can be integrated into
//! `sdk/registry.ts` by another change: import `GENERIC_CHROME_COMPONENTS`
//! (or the individual `*_DEFINITION` consts) and splice them into
//! `GENERIC_SDK_COMPONENTS`, replacing the matching `createStubDefinition`
//! entries for the ten component ids implemented here.

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ConnectorEndpointIr,
  ConnectorNodeIr,
  Diagnostic,
  GeometryIr,
  GroupNodeIr,
  JsonValue,
  NodeAccessibilityIr,
  PointIr,
  RectNodeIr,
  RelativePositionIr,
  RenderNodeIr,
  Result,
  SourceRange,
  StyleValueIr,
  TextNodeIr,
} from "../../schema/index.js";
import { diagnostic } from "../../schema/index.js";
import { estimateTextWidth } from "../../../core/diagram/text-metrics.js";
import { attachSdkOrigin, type SdkOrigin } from "../provenance.js";
import type {
  SceneFragment,
  SdkActionName,
  SdkComponentDefinition,
  SdkComponentFactory,
  SdkExpansionContext,
} from "../types.js";

// ---------------------------------------------------------------------------
// Shared geometry constants (kept in sync with diagram text-metrics / chrome layout).
// ---------------------------------------------------------------------------

const INSET = 8;
const TITLE_HEIGHT = 22;

const HEADER_DEFAULT_GEOMETRY = { x: 18, y: 16, width: 664, height: 66 } as const;
const PANEL_DEFAULT_GEOMETRY = { width: 160, height: 70 } as const;
const CHIP_DEFAULT_GEOMETRY = { width: 84, height: 26 } as const;
const NOTE_DEFAULT_GEOMETRY = { width: 160, height: 48 } as const;
const LABEL_DEFAULT_GEOMETRY = { width: 120, height: 22 } as const;
const CALLOUT_DEFAULT_GEOMETRY = { width: 140, height: 48 } as const;
const CALLOUT_STEM_DROP = 24;
const DIVIDER_DEFAULT_LENGTH = 200;
const BRACKET_DEFAULT_VERTICAL = { width: 24, height: 120 } as const;
const BRACKET_DEFAULT_HORIZONTAL = { width: 120, height: 24 } as const;

const LEGEND_ROW_HEIGHT = 20;
const LEGEND_ROW_GAP = 4;
const LEGEND_SWATCH_SIZE = 10;
const LEGEND_LABEL_GAP = 8;
const LEGEND_DEFAULT_WIDTH = 180;
const LEGEND_DEFAULT_SWATCH_ROLE = "@theme.accent.control";

type CardSizePreset = Readonly<{ width: number; height: number }>;

/** Named `sdk.card` size presets replacing raw pixel clusters at call sites. */
const CARD_SIZE_PRESETS: Readonly<Record<string, CardSizePreset>> = {
  compact: { width: 150, height: 88 },
  standard: { width: 190, height: 88 },
  wide: { width: 250, height: 88 },
};

/** Public actions shared by every generic chrome / content component. */
const CHROME_ACTIONS = [
  "enter",
  "emphasis",
  "exit",
] as const satisfies readonly SdkActionName[];

// ---------------------------------------------------------------------------
// Prop reading helpers. Props arrive as loosely-typed JsonValue; each helper
// narrows to the concrete type a factory needs and leaves invalid / absent
// values as `undefined` so callers can apply defaults or raise diagnostics.
// ---------------------------------------------------------------------------

function jsonString(value: JsonValue | undefined): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function jsonNumber(value: JsonValue | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function jsonRecord(
  value: JsonValue | undefined,
): Readonly<Record<string, JsonValue>> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, JsonValue>)
    : undefined;
}

function jsonArray(value: JsonValue | undefined): readonly JsonValue[] | undefined {
  return Array.isArray(value) ? value : undefined;
}

function stringProp(
  props: Readonly<Record<string, JsonValue>>,
  key: string,
): string | undefined {
  return jsonString(props[key]);
}

function numberProp(
  props: Readonly<Record<string, JsonValue>>,
  key: string,
): number | undefined {
  return jsonNumber(props[key]);
}

/**
 * Reads an optional `position = { relativeTo, anchor?, dx?, dy? }` prop
 * object, authoring an IR `relativePosition` so the node's own x/y resolve
 * at render time against an already-declared sibling instead of a hardcoded
 * literal. `relativeTo` names the target's root instance id (the sibling's
 * `id` prop). Absent or malformed input yields `undefined`.
 */
function relativePositionProp(
  props: Readonly<Record<string, JsonValue>>,
): RelativePositionIr | undefined {
  const record = jsonRecord(props["position"]);
  if (record === undefined) {
    return undefined;
  }
  const nodeId = jsonString(record["relativeTo"]);
  if (nodeId === undefined) {
    return undefined;
  }
  const anchor = jsonString(record["anchor"]);
  const dx = jsonNumber(record["dx"]);
  const dy = jsonNumber(record["dy"]);
  return {
    nodeId,
    ...(anchor !== undefined ? { anchor } : {}),
    ...(dx !== undefined ? { dx } : {}),
    ...(dy !== undefined ? { dy } : {}),
  };
}

/** Preserves explicitly authored geometry inside a managed layout container. */
function managedLayoutEscapeHatch(
  props: Readonly<Record<string, JsonValue>>,
): Readonly<Record<string, StyleValueIr>> {
  const style = jsonRecord(props["style"]);
  return style?.["position"] === "absolute" ? { position: "absolute" } : {};
}

/** Reads a required non-empty string prop, recording a diagnostic when absent. */
function requireStringProp(
  props: Readonly<Record<string, JsonValue>>,
  key: string,
  componentId: string,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): string | undefined {
  const value = stringProp(props, key);
  if (value === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "${componentId}" requires a non-empty string prop "${key}".`,
        sourceMap,
        `Provide "${key}" as a non-empty string.`,
      ),
    );
  }
  return value;
}

// ---------------------------------------------------------------------------
// Result helpers.
// ---------------------------------------------------------------------------

function succeed(fragment: SceneFragment): Result<SceneFragment> {
  return {
    ok: true,
    value: fragment,
    diagnostics: [],
  };
}

function fail(diagnostics: readonly Diagnostic[]): Result<SceneFragment> {
  return { ok: false, diagnostics };
}

// ---------------------------------------------------------------------------
// Provenance helper.
// ---------------------------------------------------------------------------

function makeOrigin(
  componentId: string,
  context: SdkExpansionContext,
  generatedRole: string,
): SdkOrigin {
  return {
    componentId,
    instanceId: context.instanceId,
    sourceMap: context.sourceMap,
    generatedRole,
  };
}

// ---------------------------------------------------------------------------
// Node builders. Every node carries the fragment's `sourceMap` (the SDK call
// site) and an accessibility label; children use coordinates local to their
// parent, matching native semantic `core.panel` / `core.header` /
// `core.callout` local-layout convention.
// ---------------------------------------------------------------------------

function buildAccessibility(label: string, description?: string): NodeAccessibilityIr {
  return description !== undefined ? { label, description } : { label };
}

function buildText(args: {
  id: string;
  text: string;
  geometry: GeometryIr;
  style: Readonly<Record<string, StyleValueIr>>;
  sourceMap: SourceRange;
}): TextNodeIr {
  return {
    kind: "text",
    id: args.id,
    capabilityId: "core.text",
    geometry: args.geometry,
    style: args.style,
    accessibility: buildAccessibility(args.text),
    fallback: args.text,
    sourceMap: args.sourceMap,
    text: args.text,
  };
}

function buildRect(args: {
  id: string;
  geometry: GeometryIr;
  style: Readonly<Record<string, StyleValueIr>>;
  label: string;
  sourceMap: SourceRange;
}): RectNodeIr {
  return {
    kind: "rect",
    id: args.id,
    capabilityId: "core.rect",
    geometry: args.geometry,
    style: args.style,
    accessibility: buildAccessibility(args.label),
    fallback: args.label,
    sourceMap: args.sourceMap,
  };
}

function buildGroup(args: {
  id: string;
  capabilityId: string;
  geometry: GeometryIr;
  relativePosition?: RelativePositionIr;
  style: Readonly<Record<string, StyleValueIr>>;
  props?: Readonly<Record<string, JsonValue>>;
  children: readonly RenderNodeIr[];
  label: string;
  sourceMap: SourceRange;
}): GroupNodeIr {
  return {
    kind: "group",
    id: args.id,
    capabilityId: args.capabilityId,
    geometry: args.geometry,
    ...(args.relativePosition !== undefined
      ? { relativePosition: args.relativePosition }
      : {}),
    style: args.style,
    ...(args.props !== undefined ? { props: args.props } : {}),
    accessibility: buildAccessibility(args.label),
    fallback: args.label,
    sourceMap: args.sourceMap,
    children: args.children,
  };
}

function buildConnector(args: {
  id: string;
  capabilityId: string;
  geometry: GeometryIr;
  style: Readonly<Record<string, StyleValueIr>>;
  from: ConnectorEndpointIr;
  to: ConnectorEndpointIr;
  path?: string;
  label: string;
  sourceMap: SourceRange;
}): ConnectorNodeIr {
  return {
    kind: "connector",
    id: args.id,
    capabilityId: args.capabilityId,
    geometry: args.geometry,
    style: args.style,
    accessibility: buildAccessibility(args.label),
    fallback: args.label,
    sourceMap: args.sourceMap,
    from: args.from,
    to: args.to,
    ...(args.path !== undefined ? { path: args.path } : {}),
  };
}

/** Approximates a curly brace along the left / right / top / bottom span. */
function bracePath(geometry: GeometryIr, side: "left" | "right" | "top" | "bottom"): string {
  const { x, y, width, height } = geometry;
  switch (side) {
    case "right": {
      const mid = y + height / 2;
      return `M${x} ${y} C${x + width} ${y}, ${x + width} ${mid - height * 0.15}, ${x + width * 0.35} ${mid} C${x + width} ${mid + height * 0.15}, ${x + width} ${y + height}, ${x} ${y + height}`;
    }
    case "top": {
      const mid = x + width / 2;
      return `M${x} ${y + height} C${x} ${y}, ${mid - width * 0.15} ${y}, ${mid} ${y + height * 0.65} C${mid + width * 0.15} ${y}, ${x + width} ${y}, ${x + width} ${y + height}`;
    }
    case "bottom": {
      const mid = x + width / 2;
      return `M${x} ${y} C${x} ${y + height}, ${mid - width * 0.15} ${y + height}, ${mid} ${y + height * 0.35} C${mid + width * 0.15} ${y + height}, ${x + width} ${y + height}, ${x + width} ${y}`;
    }
    case "left":
    default: {
      const mid = y + height / 2;
      return `M${x + width} ${y} C${x} ${y}, ${x} ${mid - height * 0.15}, ${x + width * 0.65} ${mid} C${x} ${mid + height * 0.15}, ${x} ${y + height}, ${x + width} ${y + height}`;
    }
  }
}

function normalizeBracketSide(value: string | undefined): "left" | "right" | "top" | "bottom" {
  return value === "right" || value === "top" || value === "bottom" ? value : "left";
}

/** Port anchors for bracePath endpoints on each bracket side. */
function bracketPortAnchors(
  side: "left" | "right" | "top" | "bottom",
): Readonly<{ start: string; end: string }> {
  switch (side) {
    case "right":
      return { start: "nw", end: "sw" };
    case "top":
      return { start: "sw", end: "se" };
    case "bottom":
      return { start: "nw", end: "ne" };
    case "left":
    default:
      return { start: "ne", end: "se" };
  }
}

// ---------------------------------------------------------------------------
// Descriptor helper.
// ---------------------------------------------------------------------------

function makeDescriptor(
  id: string,
  symbolExport: string,
  capabilityId: string,
  props: Readonly<Record<string, ComponentPropDescriptor>>,
): ComponentDescriptor {
  return {
    id,
    symbolExport,
    version: "1.0.0",
    classification: "flow-only",
    props: { id: { type: "string", required: true }, ...props },
    slots: {},
    events: [],
    capabilityId,
    deterministic: true,
  };
}

// ---------------------------------------------------------------------------
// sdk.header — chrome geometry (18,16,664,66) with theme surface / ink roles.
// Exposes `title` / `caption` ports; binds enter/emphasis/exit to the header
// group id.
// ---------------------------------------------------------------------------

const HEADER_DESCRIPTOR = makeDescriptor("sdk.header", "Header", "core.header", {
  title: { type: "string", required: false },
  caption: { type: "string", required: false },
  x: { type: "number", required: false, default: HEADER_DEFAULT_GEOMETRY.x },
  y: { type: "number", required: false, default: HEADER_DEFAULT_GEOMETRY.y },
  width: { type: "number", required: false, default: HEADER_DEFAULT_GEOMETRY.width },
  height: { type: "number", required: false, default: HEADER_DEFAULT_GEOMETRY.height },
  surfaceRole: { type: "string", required: false },
  inkRole: { type: "string", required: false },
  position: { type: "object", required: false },
});

const headerFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.header", context.sourceMap, diagnostics);
  if (id === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const title = stringProp(props, "title");
  const caption = stringProp(props, "caption");
  const surfaceRole = stringProp(props, "surfaceRole");
  const relativePosition = relativePositionProp(props);
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? HEADER_DEFAULT_GEOMETRY.x,
    y: numberProp(props, "y") ?? HEADER_DEFAULT_GEOMETRY.y,
    width: numberProp(props, "width") ?? HEADER_DEFAULT_GEOMETRY.width,
    height: numberProp(props, "height") ?? HEADER_DEFAULT_GEOMETRY.height,
  };

  const ports: Record<string, ConnectorEndpointIr> = {};

  if (title !== undefined) {
    const titleId = `${rootId}__title`;
    ports.title = { nodeId: titleId };
  }

  if (caption !== undefined) {
    const captionId = `${rootId}__caption`;
    ports.caption = { nodeId: captionId };
  }

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.header",
      geometry,
      ...(relativePosition !== undefined ? { relativePosition } : {}),
      style: {
        coordinateSpace: "local",
        ...(surfaceRole !== undefined ? { fill: surfaceRole } : {}),
      },
      props: {
        ...(title !== undefined ? { title } : {}),
        ...(caption !== undefined ? { caption } : {}),
      },
      children: [],
      label: title ?? caption ?? "Header",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.header", context, "root"),
  );

  return succeed({
    roots: [root],
    ports,
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const HEADER_DEFINITION: SdkComponentDefinition = {
  descriptor: HEADER_DESCRIPTOR,
  factory: headerFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.panel — title + detail (maps to native semantic `core.panel`).
// Exposes `title` / `detail` ports.
// ---------------------------------------------------------------------------

const PANEL_DESCRIPTOR = makeDescriptor("sdk.panel", "Panel", "core.panel", {
  title: { type: "string", required: false },
  detail: { type: "string", required: false },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: PANEL_DEFAULT_GEOMETRY.width },
  height: { type: "number", required: false, default: PANEL_DEFAULT_GEOMETRY.height },
  surfaceRole: { type: "string", required: false },
  strokeRole: { type: "string", required: false },
  position: { type: "object", required: false },
  style: { type: "object", required: false },
});

const panelFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.panel", context.sourceMap, diagnostics);
  if (id === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const title = stringProp(props, "title");
  const detail = stringProp(props, "detail");
  const surfaceRole = stringProp(props, "surfaceRole");
  const strokeRole = stringProp(props, "strokeRole");
  const relativePosition = relativePositionProp(props);
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? PANEL_DEFAULT_GEOMETRY.width,
    height: numberProp(props, "height") ?? PANEL_DEFAULT_GEOMETRY.height,
  };

  const ports: Record<string, ConnectorEndpointIr> = {};

  if (title !== undefined) {
    const titleId = `${rootId}__title`;
    ports.title = { nodeId: titleId };
  }

  if (detail !== undefined) {
    const detailId = `${rootId}__detail`;
    ports.detail = { nodeId: detailId };
  }

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.panel",
      geometry,
      ...(relativePosition !== undefined ? { relativePosition } : {}),
      style: {
        coordinateSpace: "local",
        ...managedLayoutEscapeHatch(props),
        ...(surfaceRole !== undefined ? { fill: surfaceRole } : {}),
        ...(strokeRole !== undefined ? { stroke: strokeRole } : {}),
      },
      props: {
        ...(title !== undefined ? { title } : {}),
        ...(detail !== undefined ? { detail } : {}),
      },
      children: [],
      label: title ?? detail ?? "Panel",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.panel", context, "root"),
  );

  return succeed({
    roots: [root],
    ports,
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const PANEL_DEFINITION: SdkComponentDefinition = {
  descriptor: PANEL_DESCRIPTOR,
  factory: panelFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.card — title + detail + subtitle (absorbs bespoke rect+text three-line
// signatures). Accepts preset sizes (`compact`, `standard`, `wide`) instead
// of raw pixel clusters. Exposes `title` / `detail` / `subtitle` ports.
// ---------------------------------------------------------------------------

const CARD_DESCRIPTOR = makeDescriptor("sdk.card", "Card", "core.panel", {
  title: { type: "string", required: false },
  detail: { type: "string", required: false },
  subtitle: { type: "string", required: false },
  size: { type: "string", required: false, default: "standard" },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false },
  height: { type: "number", required: false },
  surfaceRole: { type: "string", required: false },
  strokeRole: { type: "string", required: false },
});

const cardFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.card", context.sourceMap, diagnostics);
  if (id === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const title = stringProp(props, "title");
  const detail = stringProp(props, "detail");
  const subtitle = stringProp(props, "subtitle");
  const surfaceRole = stringProp(props, "surfaceRole");
  const strokeRole = stringProp(props, "strokeRole");
  const sizeName = stringProp(props, "size") ?? "standard";
  const preset = CARD_SIZE_PRESETS[sizeName] ?? CARD_SIZE_PRESETS.standard!;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? preset.width,
    height: numberProp(props, "height") ?? preset.height,
  };

  const ports: Record<string, ConnectorEndpointIr> = {};

  if (title !== undefined) {
    const titleId = `${rootId}__title`;
    ports.title = { nodeId: titleId };
  }

  if (detail !== undefined) {
    const detailId = `${rootId}__detail`;
    ports.detail = { nodeId: detailId };
  }

  if (subtitle !== undefined) {
    const subtitleId = `${rootId}__subtitle`;
    ports.subtitle = { nodeId: subtitleId };
  }

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.panel",
      geometry,
      style: {
        coordinateSpace: "local",
        ...(surfaceRole !== undefined ? { fill: surfaceRole } : {}),
        ...(strokeRole !== undefined ? { stroke: strokeRole } : {}),
      },
      props: {
        ...(title !== undefined ? { title } : {}),
        ...(detail !== undefined ? { detail } : {}),
        ...(subtitle !== undefined ? { subtitle } : {}),
      },
      children: [],
      label: title ?? detail ?? subtitle ?? "Card",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.card", context, "root"),
  );

  return succeed({
    roots: [root],
    ports,
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const CARD_DEFINITION: SdkComponentDefinition = {
  descriptor: CARD_DESCRIPTOR,
  factory: cardFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.chip — small rounded label chip (matches `core.chip` semantic geometry).
// Exposes a `label` port.
// ---------------------------------------------------------------------------

const CHIP_DESCRIPTOR = makeDescriptor("sdk.chip", "Chip", "core.chip", {
  label: { type: "string", required: true },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: CHIP_DEFAULT_GEOMETRY.width },
  height: { type: "number", required: false, default: CHIP_DEFAULT_GEOMETRY.height },
  radius: { type: "number", required: false, default: 0 },
  surfaceRole: { type: "string", required: false },
  strokeRole: { type: "string", required: false },
  style: { type: "object", required: false },
});

const chipFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.chip", context.sourceMap, diagnostics);
  const label = requireStringProp(props, "label", "sdk.chip", context.sourceMap, diagnostics);
  if (id === undefined || label === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? CHIP_DEFAULT_GEOMETRY.width,
    height: numberProp(props, "height") ?? CHIP_DEFAULT_GEOMETRY.height,
  };
  const radius = numberProp(props, "radius") ?? 0;
  const surfaceRole = stringProp(props, "surfaceRole");
  const strokeRole = stringProp(props, "strokeRole");

  const labelId = `${rootId}__label`;

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.chip",
      geometry,
      style: {
        coordinateSpace: "local",
        ...managedLayoutEscapeHatch(props),
        radius,
        rx: radius,
        ...(surfaceRole !== undefined ? { fill: surfaceRole } : {}),
        ...(strokeRole !== undefined ? { stroke: strokeRole } : {}),
      },
      props: { label },
      children: [],
      label,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.chip", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { label: { nodeId: labelId } },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const CHIP_DEFINITION: SdkComponentDefinition = {
  descriptor: CHIP_DESCRIPTOR,
  factory: chipFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.note — annotation card (matches `core.note` semantic geometry / paint
// defaults). Exposes a `caption` port.
// ---------------------------------------------------------------------------

const NOTE_DESCRIPTOR = makeDescriptor("sdk.note", "Note", "core.note", {
  text: { type: "string", required: true },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: NOTE_DEFAULT_GEOMETRY.width },
  height: { type: "number", required: false, default: NOTE_DEFAULT_GEOMETRY.height },
  radius: { type: "number", required: false, default: 0 },
  strokeWidth: { type: "number", required: false, default: 1 },
  surfaceRole: { type: "string", required: false, default: "@theme.surface.elevated" },
  strokeRole: { type: "string", required: false, default: "@theme.ink.secondary" },
  inkRole: { type: "string", required: false, default: "@theme.ink.secondary" },
});

const noteFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.note", context.sourceMap, diagnostics);
  const text = requireStringProp(props, "text", "sdk.note", context.sourceMap, diagnostics);
  if (id === undefined || text === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? NOTE_DEFAULT_GEOMETRY.width,
    height: numberProp(props, "height") ?? NOTE_DEFAULT_GEOMETRY.height,
  };
  const surfaceRole = stringProp(props, "surfaceRole") ?? "@theme.surface.elevated";
  const strokeRole = stringProp(props, "strokeRole") ?? "@theme.ink.secondary";
  const inkRole = stringProp(props, "inkRole") ?? "@theme.ink.secondary";
  const radius = numberProp(props, "radius") ?? 0;
  const strokeWidth = numberProp(props, "strokeWidth") ?? 1;

  const captionId = `${rootId}__caption`;

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.note",
      geometry,
      style: {
        coordinateSpace: "local",
        fill: surfaceRole,
        stroke: strokeRole,
        strokeWidth,
        radius,
      },
      props: { text, inkRole },
      children: [],
      label: text,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.note", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { caption: { nodeId: captionId } },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const NOTE_DEFINITION: SdkComponentDefinition = {
  descriptor: NOTE_DESCRIPTOR,
  factory: noteFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.label — single text node. Exposes a `label` port referencing itself.
// ---------------------------------------------------------------------------

const LABEL_DESCRIPTOR = makeDescriptor("sdk.label", "Label", "core.text", {
  text: { type: "string", required: true },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: LABEL_DEFAULT_GEOMETRY.width },
  height: { type: "number", required: false, default: LABEL_DEFAULT_GEOMETRY.height },
  fontSize: { type: "number", required: false, default: 12 },
  weight: { type: "string", required: false, default: "regular" },
  align: { type: "string", required: false, default: "middle" },
  inkRole: { type: "string", required: false },
});

const labelFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.label", context.sourceMap, diagnostics);
  const text = requireStringProp(props, "text", "sdk.label", context.sourceMap, diagnostics);
  if (id === undefined || text === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? LABEL_DEFAULT_GEOMETRY.width,
    height: numberProp(props, "height") ?? LABEL_DEFAULT_GEOMETRY.height,
  };
  const fontSize = numberProp(props, "fontSize") ?? 12;
  const weight = stringProp(props, "weight") ?? "regular";
  const align = stringProp(props, "align") ?? "middle";
  const inkRole = stringProp(props, "inkRole");

  const root = attachSdkOrigin(
    buildText({
      id: rootId,
      text,
      geometry,
      style: {
        fontSize,
        fontWeight: weight === "bold" ? "bold" : "normal",
        textAnchor: align,
        ...(inkRole !== undefined ? { fill: inkRole } : {}),
      },
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.label", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { label: { nodeId: rootId } },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const LABEL_DEFINITION: SdkComponentDefinition = {
  descriptor: LABEL_DESCRIPTOR,
  factory: labelFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.legend — swatch + label rows. Exposes an indexed `entry[i]` port family
// (and a `title` port when a heading is authored).
// ---------------------------------------------------------------------------

type LegendEntry = Readonly<{ label: string; colorRole?: string }>;

function parseLegendEntries(
  props: Readonly<Record<string, JsonValue>>,
  componentId: string,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): readonly LegendEntry[] | undefined {
  const raw = jsonArray(props.entries);
  if (raw === undefined || raw.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "${componentId}" requires a non-empty "entries" array prop.`,
        sourceMap,
        `Provide "entries" as an array of {label, colorRole?} objects.`,
      ),
    );
    return undefined;
  }

  const entries: LegendEntry[] = [];
  let valid = true;
  raw.forEach((rawEntry, index) => {
    const record = jsonRecord(rawEntry);
    const label = record !== undefined ? jsonString(record.label) : undefined;
    if (label === undefined) {
      valid = false;
      diagnostics.push(
        diagnostic(
          "SDK_PROP_INVALID_TYPE",
          "error",
          `Component "${componentId}" entries[${index}] requires a non-empty string "label".`,
          sourceMap,
          `Provide entries[${index}].label as a non-empty string.`,
        ),
      );
      return;
    }
    const colorRoleRaw = record !== undefined ? record.colorRole : undefined;
    let colorRole: string | undefined;
    if (typeof colorRoleRaw === "string" && colorRoleRaw.length > 0) {
      colorRole = colorRoleRaw.startsWith("@theme.")
        ? colorRoleRaw
        : colorRoleRaw.includes(".")
          ? `@theme.${colorRoleRaw}`
          : colorRoleRaw;
    }
    entries.push({ label, ...(colorRole !== undefined ? { colorRole } : {}) });
  });

  return valid ? entries : undefined;
}

const LEGEND_DESCRIPTOR = makeDescriptor("sdk.legend", "Legend", "core.group", {
  title: { type: "string", required: false },
  entries: { type: "json", required: true },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: LEGEND_DEFAULT_WIDTH },
  height: { type: "number", required: false },
  rowHeight: { type: "number", required: false, default: LEGEND_ROW_HEIGHT },
});

const legendFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.legend", context.sourceMap, diagnostics);
  const entries = parseLegendEntries(props, "sdk.legend", context.sourceMap, diagnostics);
  if (id === undefined || entries === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const title = stringProp(props, "title");
  const rowHeight = numberProp(props, "rowHeight") ?? LEGEND_ROW_HEIGHT;
  const authoredWidth = numberProp(props, "width") ?? LEGEND_DEFAULT_WIDTH;
  const titleWidth =
    title !== undefined ? estimateTextWidth(title, 12, "bold") + INSET : 0;
  const entryWidth = entries.reduce(
    (maximum, entry) =>
      Math.max(
        maximum,
        LEGEND_SWATCH_SIZE +
          LEGEND_LABEL_GAP +
          estimateTextWidth(entry.label, 11) +
          INSET,
      ),
    0,
  );
  const width = Math.max(authoredWidth, titleWidth, entryWidth);

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};
  let cursorY = 0;

  if (title !== undefined) {
    const titleId = `${rootId}__title`;
    children.push(
      attachSdkOrigin(
        buildText({
          id: titleId,
          text: title,
          geometry: { x: 0, y: cursorY, width, height: TITLE_HEIGHT },
          style: { fontSize: 12, fontWeight: "bold", textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.legend", context, "title"),
      ),
    );
    ports.title = { nodeId: titleId };
    cursorY += TITLE_HEIGHT + INSET / 2;
  }

  entries.forEach((entry, index) => {
    const rowId = `${rootId}__entry-${index}`;
    const swatchId = `${rowId}__swatch`;
    const labelId = `${rowId}__label`;

    const swatch = attachSdkOrigin(
      buildRect({
        id: swatchId,
        geometry: {
          x: 0,
          y: Math.max((rowHeight - LEGEND_SWATCH_SIZE) / 2, 0),
          width: LEGEND_SWATCH_SIZE,
          height: LEGEND_SWATCH_SIZE,
        },
        style: { fill: entry.colorRole ?? LEGEND_DEFAULT_SWATCH_ROLE, radius: 0 },
        label: entry.label,
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.legend", context, "swatch"),
    );

    const labelNode = attachSdkOrigin(
      buildText({
        id: labelId,
        text: entry.label,
        geometry: {
          x: LEGEND_SWATCH_SIZE + LEGEND_LABEL_GAP,
          y: 0,
          width: Math.max(width - LEGEND_SWATCH_SIZE - LEGEND_LABEL_GAP, 0),
          height: rowHeight,
        },
        style: { fontSize: 11, textAnchor: "start" },
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.legend", context, "label"),
    );

    const row = attachSdkOrigin(
      buildGroup({
        id: rowId,
        capabilityId: "core.group",
        geometry: { x: 0, y: cursorY, width, height: rowHeight },
        style: { coordinateSpace: "local" },
        children: [swatch, labelNode],
        label: entry.label,
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.legend", context, "entry"),
    );

    children.push(row);
    ports[`entry[${index}]`] = { nodeId: rowId };
    cursorY += rowHeight + LEGEND_ROW_GAP;
  });

  const contentHeight = entries.length > 0 ? cursorY - LEGEND_ROW_GAP : cursorY;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width,
    height: numberProp(props, "height") ?? Math.max(contentHeight, rowHeight),
  };

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry,
      style: { coordinateSpace: "local" },
      children,
      label: title ?? "Legend",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.legend", context, "root"),
  );

  return succeed({
    roots: [root],
    ports,
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const LEGEND_DEFINITION: SdkComponentDefinition = {
  descriptor: LEGEND_DESCRIPTOR,
  factory: legendFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.callout — label + stem pointing at an absolute target (matches
// `core.callout` semantic geometry). Exposes `label` / `target` ports.
// ---------------------------------------------------------------------------

const CALLOUT_DESCRIPTOR = makeDescriptor("sdk.callout", "Callout", "core.callout", {
  text: { type: "string", required: true },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false, default: CALLOUT_DEFAULT_GEOMETRY.width },
  height: { type: "number", required: false, default: CALLOUT_DEFAULT_GEOMETRY.height },
  target: { type: "json", required: false },
  strokeRole: { type: "string", required: false },
});

const calloutFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.callout", context.sourceMap, diagnostics);
  const text = requireStringProp(props, "text", "sdk.callout", context.sourceMap, diagnostics);
  if (id === undefined || text === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? CALLOUT_DEFAULT_GEOMETRY.width,
    height: numberProp(props, "height") ?? CALLOUT_DEFAULT_GEOMETRY.height,
  };
  const strokeRole = stringProp(props, "strokeRole");
  const targetRecord = jsonRecord(props.target);
  const targetX = targetRecord !== undefined ? jsonNumber(targetRecord.x) : undefined;
  const targetY = targetRecord !== undefined ? jsonNumber(targetRecord.y) : undefined;
  const target: PointIr =
    targetX !== undefined && targetY !== undefined
      ? { x: targetX, y: targetY }
      : {
          x: geometry.x + geometry.width / 2,
          y: geometry.y + geometry.height + CALLOUT_STEM_DROP,
        };

  const localAnchor: PointIr = { x: geometry.width / 2, y: geometry.height };
  const localTarget: PointIr = { x: target.x - geometry.x, y: target.y - geometry.y };
  const stemPath = `M${localAnchor.x} ${localAnchor.y} L${localTarget.x} ${localTarget.y}`;

  const textId = `${rootId}__label`;
  const stemId = `${rootId}__stem`;

  const textChild = attachSdkOrigin(
    buildText({
      id: textId,
      text,
      geometry: { x: 0, y: 0, width: geometry.width, height: geometry.height },
      style: { fontSize: 12, textAnchor: "middle" },
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.callout", context, "label"),
  );

  const stem = attachSdkOrigin(
    buildConnector({
      id: stemId,
      capabilityId: "core.path",
      geometry: {
        x: Math.min(localAnchor.x, localTarget.x),
        y: Math.min(localAnchor.y, localTarget.y),
        width: Math.abs(localTarget.x - localAnchor.x),
        height: Math.abs(localTarget.y - localAnchor.y),
      },
      style: {
        strokeWidth: 1.25,
        fill: "none",
        markerEnd: "none",
        ...(strokeRole !== undefined ? { stroke: strokeRole } : {}),
      },
      from: localAnchor,
      to: localTarget,
      path: stemPath,
      label: `${text} stem`,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.callout", context, "stem"),
  );

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.callout",
      geometry,
      style: {},
      children: [textChild, stem],
      label: text,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.callout", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { label: { nodeId: textId }, target: { nodeId: stemId } },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const CALLOUT_DEFINITION: SdkComponentDefinition = {
  descriptor: CALLOUT_DESCRIPTOR,
  factory: calloutFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.divider — straight rule along `axis` (matches `core.divider` semantic
// geometry). Exposes `start` / `end` ports.
// ---------------------------------------------------------------------------

const DIVIDER_DESCRIPTOR = makeDescriptor("sdk.divider", "Divider", "core.divider", {
  axis: { type: "string", required: false, default: "x" },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  length: { type: "number", required: false, default: DIVIDER_DEFAULT_LENGTH },
  strokeRole: { type: "string", required: false },
  strokeWidth: { type: "number", required: false, default: 1.2 },
});

const dividerFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.divider", context.sourceMap, diagnostics);
  if (id === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const axis: "x" | "y" = stringProp(props, "axis") === "y" ? "y" : "x";
  const length = numberProp(props, "length") ?? DIVIDER_DEFAULT_LENGTH;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;
  const strokeRole = stringProp(props, "strokeRole");
  const strokeWidth = numberProp(props, "strokeWidth") ?? 1.2;

  const geometry: GeometryIr =
    axis === "y" ? { x, y, width: 0, height: length } : { x, y, width: length, height: 0 };
  const from: ConnectorEndpointIr = { x, y };
  const to: ConnectorEndpointIr =
    axis === "y" ? { x, y: y + length } : { x: x + length, y };
  const path = axis === "y" ? `M${x} ${y} V${y + length}` : `M${x} ${y} H${x + length}`;
  const startAnchor = axis === "y" ? "n" : "w";
  const endAnchor = axis === "y" ? "s" : "e";

  const root = attachSdkOrigin(
    buildConnector({
      id: rootId,
      capabilityId: "core.divider",
      geometry,
      style: {
        fill: "none",
        markerEnd: "none",
        strokeWidth,
        ...(strokeRole !== undefined ? { stroke: strokeRole } : {}),
      },
      from,
      to,
      path,
      label: "Divider",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.divider", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: {
      start: { nodeId: rootId, anchor: startAnchor },
      end: { nodeId: rootId, anchor: endAnchor },
    },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const DIVIDER_DEFINITION: SdkComponentDefinition = {
  descriptor: DIVIDER_DESCRIPTOR,
  factory: dividerFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.bracket — curly brace along `side` (matches `core.bracket` semantic
// geometry). Exposes `start` / `end` ports.
// ---------------------------------------------------------------------------

const BRACKET_DESCRIPTOR = makeDescriptor("sdk.bracket", "Bracket", "core.bracket", {
  side: { type: "string", required: false, default: "left" },
  x: { type: "number", required: false, default: 0 },
  y: { type: "number", required: false, default: 0 },
  width: { type: "number", required: false },
  height: { type: "number", required: false },
  strokeRole: { type: "string", required: false },
  strokeWidth: { type: "number", required: false, default: 1.5 },
});

const bracketFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.bracket", context.sourceMap, diagnostics);
  if (id === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const side = normalizeBracketSide(stringProp(props, "side"));
  const defaultSize =
    side === "top" || side === "bottom" ? BRACKET_DEFAULT_HORIZONTAL : BRACKET_DEFAULT_VERTICAL;
  const geometry: GeometryIr = {
    x: numberProp(props, "x") ?? 0,
    y: numberProp(props, "y") ?? 0,
    width: numberProp(props, "width") ?? defaultSize.width,
    height: numberProp(props, "height") ?? defaultSize.height,
  };
  const strokeRole = stringProp(props, "strokeRole");
  const strokeWidth = numberProp(props, "strokeWidth") ?? 1.5;
  const path = bracePath(geometry, side);
  const anchors = bracketPortAnchors(side);

  const root = attachSdkOrigin(
    buildConnector({
      id: rootId,
      capabilityId: "core.bracket",
      geometry,
      style: {
        fill: "none",
        markerEnd: "none",
        strokeWidth,
        ...(strokeRole !== undefined ? { stroke: strokeRole } : {}),
      },
      from: { x: geometry.x, y: geometry.y },
      to: { x: geometry.x + geometry.width, y: geometry.y + geometry.height },
      path,
      label: "Bracket",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.bracket", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: {
      start: { nodeId: rootId, anchor: anchors.start },
      end: { nodeId: rootId, anchor: anchors.end },
    },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const BRACKET_DEFINITION: SdkComponentDefinition = {
  descriptor: BRACKET_DESCRIPTOR,
  factory: bracketFactory,
  actions: CHROME_ACTIONS,
};

// ---------------------------------------------------------------------------
// Bundle: the controller integrates this into `sdk/registry.ts`'s
// `GENERIC_SDK_COMPONENTS` list in place of the matching stub entries.
// ---------------------------------------------------------------------------

/** Generic SDK chrome / content component definitions (working factories). */
export const GENERIC_CHROME_COMPONENTS: readonly SdkComponentDefinition[] = [
  HEADER_DEFINITION,
  PANEL_DEFINITION,
  CARD_DEFINITION,
  CHIP_DEFINITION,
  NOTE_DEFINITION,
  LABEL_DEFINITION,
  LEGEND_DEFINITION,
  CALLOUT_DEFINITION,
  DIVIDER_DEFINITION,
  BRACKET_DEFINITION,
];
