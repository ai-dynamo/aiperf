/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deck-port SDK composite factories: `sdk.sectionDivider`, `sdk.stepChain`,
 * `sdk.bigStat`, `sdk.compareGrid` (Task 2 appends four more to this file).
 *
 * These composites capture recurring visual shapes in the NVIDIA-branded
 * "Rust architecture" reference deck that no existing generic primitive covers
 * on its own:
 *
 * - `sdk.sectionDivider` is the right-aligned chapter-break block (a huge mono
 *   green number over a bold title and gray subtitle), matching the deck's
 *   "Divider 01-04" slides.
 * - `sdk.stepChain` is an ordered, arrow-linked run of numbered step boxes —
 *   horizontal (the "Orientation" slide's `01 VALIDATE → … → 06 EMIT` row) or
 *   vertical (the "Flow diagram" slide's `Python → aiperf-runner → … → stdout`
 *   pipeline) — each box carrying a green accent edge.
 * - `sdk.bigStat` is the oversized hero figure (the "Three modes" slide's giant
 *   green `3`) with an optional title and description stacked beneath it.
 * - `sdk.compareGrid` is an N-column takeaway grid of green-top-accented cells
 *   (the "Thesis" slide's 3-column row, the "Failure funnel" `0 / 1 / 2` row).
 *
 * Every factory is pure: no DOM, React, network, wall clock, or mutable global
 * state. Generated node ids are seeded from `context.instanceId`
 * (`${instanceId}` for the fragment root, `${instanceId}__role` for generated
 * children) so expansion is stable across repeated calls. Prop errors are
 * reported as diagnostics (never thrown), mirroring `generic/chrome.ts` and
 * `generic/composites.ts`.
 *
 * This module is deliberately self-contained (mirroring `generic/chrome.ts`)
 * so `sdk/registry.ts` can integrate it by importing
 * `DECK_COMPOSITE_SDK_COMPONENTS` and appending it to the generic pack.
 */

import type {
  ComponentDescriptor,
  ComponentPropDescriptor,
  ComponentSlotDescriptor,
} from "../../schema/component-descriptor.js";
import { diagnostic, type Diagnostic, type Result } from "../../schema/diagnostic.js";
import type {
  ConnectorEndpointIr,
  ConnectorNodeIr,
  GeometryIr,
  GroupNodeIr,
  RectNodeIr,
  RenderNodeIr,
  TextNodeIr,
} from "../../schema/ir.js";
import type { JsonValue } from "../../schema/json-value.js";
import type { SourceRange } from "../../schema/source.js";
import type { StyleValueIr } from "../../schema/theme.js";
import { attachSdkOrigin, type SdkOrigin } from "../provenance.js";
import type {
  SceneFragment,
  SdkActionName,
  SdkComponentDefinition,
  SdkComponentFactory,
  SdkExpansionContext,
} from "../types.js";

// ---------------------------------------------------------------------------
// Palette (deck brand tokens). Kept as literal hex so the composites read the
// same whether or not a theme is bound at the call site.
// ---------------------------------------------------------------------------

const COLOR_ACCENT = "#76B900"; // NVIDIA green
const COLOR_INK = "#111111"; // primary black text
const COLOR_SECONDARY = "#555555"; // body / subtitle gray
const COLOR_MUTED = "#999999"; // detail gray
const COLOR_BORDER = "#E4E4E4"; // thin border
const COLOR_SURFACE = "#ffffff"; // white surface
const MONO_FONT = "monospace";

/** Public actions shared by every deck composite (static layout, no motion). */
const DECK_ACTIONS = ["enter", "emphasis", "exit"] as const satisfies readonly SdkActionName[];

// ---------------------------------------------------------------------------
// Prop reading helpers (loosely-typed JsonValue → concrete, absent → undefined).
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

function stringProp(props: Readonly<Record<string, JsonValue>>, key: string): string | undefined {
  return jsonString(props[key]);
}

function numberProp(props: Readonly<Record<string, JsonValue>>, key: string): number | undefined {
  return jsonNumber(props[key]);
}

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
  return { ok: true, value: fragment, diagnostics: [] };
}

function fail(diagnostics: readonly Diagnostic[]): Result<SceneFragment> {
  return { ok: false, diagnostics };
}

// ---------------------------------------------------------------------------
// Provenance + node builders (children use coordinates local to their parent).
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
    accessibility: { label: args.text },
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
    accessibility: { label: args.label },
    fallback: args.label,
    sourceMap: args.sourceMap,
  };
}

function buildGroup(args: {
  id: string;
  capabilityId: string;
  geometry: GeometryIr;
  style: Readonly<Record<string, StyleValueIr>>;
  children: readonly RenderNodeIr[];
  label: string;
  sourceMap: SourceRange;
}): GroupNodeIr {
  return {
    kind: "group",
    id: args.id,
    capabilityId: args.capabilityId,
    geometry: args.geometry,
    style: args.style,
    accessibility: { label: args.label },
    fallback: args.label,
    sourceMap: args.sourceMap,
    children: args.children,
  };
}

function buildConnector(args: {
  id: string;
  geometry: GeometryIr;
  style: Readonly<Record<string, StyleValueIr>>;
  from: ConnectorEndpointIr;
  to: ConnectorEndpointIr;
  label: string;
  sourceMap: SourceRange;
}): ConnectorNodeIr {
  return {
    kind: "connector",
    id: args.id,
    capabilityId: "core.arrow",
    geometry: args.geometry,
    style: args.style,
    accessibility: { label: args.label },
    fallback: args.label,
    sourceMap: args.sourceMap,
    from: args.from,
    to: args.to,
  };
}

function makeDescriptor(
  id: string,
  capabilityId: string,
  props: Readonly<Record<string, ComponentPropDescriptor>>,
  slots: Readonly<Record<string, ComponentSlotDescriptor>> = {},
): ComponentDescriptor {
  const segment = id.includes(".") ? id.split(".", 2)[1]! : id;
  return {
    id,
    symbolExport: segment.charAt(0).toUpperCase() + segment.slice(1),
    version: "1.0.0",
    classification: "flow-only",
    props: { id: { type: "string", required: true }, ...props },
    slots,
    events: [],
    capabilityId,
    deterministic: true,
  };
}

// ---------------------------------------------------------------------------
// sdk.sectionDivider — right-aligned chapter break: huge mono green number over
// a bold title and gray subtitle, with an optional eyebrow kicker above.
// Root `core.group` composed of `core.text` children. Exposes
// `number` / `title` / `subtitle` / `eyebrow` ports.
// ---------------------------------------------------------------------------

const DIVIDER_WIDTH = 640;
const DIVIDER_EYEBROW_H = 22;
const DIVIDER_NUMBER_H = 148;
const DIVIDER_TITLE_H = 64;
const DIVIDER_SUBTITLE_H = 34;
const DIVIDER_GAP = 8;

const sectionDividerFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.sectionDivider", context.sourceMap, diagnostics);
  const number = requireStringProp(
    props,
    "number",
    "sdk.sectionDivider",
    context.sourceMap,
    diagnostics,
  );
  const title = requireStringProp(props, "title", "sdk.sectionDivider", context.sourceMap, diagnostics);
  if (id === undefined || number === undefined || title === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const eyebrow = stringProp(props, "eyebrow");
  const subtitle = stringProp(props, "subtitle");
  const width = numberProp(props, "width") ?? DIVIDER_WIDTH;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};
  let cursorY = 0;

  const rightAlignedText = (fontSize: number, fill: string, extra: Record<string, StyleValueIr>) =>
    ({ fontSize, fill, textAnchor: "end", ...extra }) as Record<string, StyleValueIr>;

  if (eyebrow !== undefined) {
    const eyebrowId = `${rootId}__eyebrow`;
    children.push(
      attachSdkOrigin(
        buildText({
          id: eyebrowId,
          text: eyebrow,
          geometry: { x: 0, y: cursorY, width, height: DIVIDER_EYEBROW_H },
          style: rightAlignedText(14, COLOR_ACCENT, { fontWeight: "bold", letterSpacing: 2 }),
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.sectionDivider", context, "eyebrow"),
      ),
    );
    ports.eyebrow = { nodeId: eyebrowId };
    cursorY += DIVIDER_EYEBROW_H + DIVIDER_GAP;
  }

  const numberId = `${rootId}__number`;
  children.push(
    attachSdkOrigin(
      buildText({
        id: numberId,
        text: number,
        geometry: { x: 0, y: cursorY, width, height: DIVIDER_NUMBER_H },
        style: rightAlignedText(120, COLOR_ACCENT, { fontFamily: MONO_FONT, fontWeight: "bold" }),
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.sectionDivider", context, "number"),
    ),
  );
  ports.number = { nodeId: numberId };
  cursorY += DIVIDER_NUMBER_H + DIVIDER_GAP;

  const titleId = `${rootId}__title`;
  children.push(
    attachSdkOrigin(
      buildText({
        id: titleId,
        text: title,
        geometry: { x: 0, y: cursorY, width, height: DIVIDER_TITLE_H },
        style: rightAlignedText(48, COLOR_INK, { fontWeight: "bold" }),
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.sectionDivider", context, "title"),
    ),
  );
  ports.title = { nodeId: titleId };
  cursorY += DIVIDER_TITLE_H + DIVIDER_GAP;

  if (subtitle !== undefined) {
    const subtitleId = `${rootId}__subtitle`;
    children.push(
      attachSdkOrigin(
        buildText({
          id: subtitleId,
          text: subtitle,
          geometry: { x: 0, y: cursorY, width, height: DIVIDER_SUBTITLE_H },
          style: rightAlignedText(20, COLOR_SECONDARY, {}),
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.sectionDivider", context, "subtitle"),
      ),
    );
    ports.subtitle = { nodeId: subtitleId };
    cursorY += DIVIDER_SUBTITLE_H + DIVIDER_GAP;
  }

  const height = Math.max(cursorY - DIVIDER_GAP, 0);
  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local" },
      children,
      label: `${number} ${title}`,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.sectionDivider", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const SECTION_DIVIDER_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.sectionDivider", "core.group", {
    number: { type: "string", required: true },
    title: { type: "string", required: true },
    subtitle: { type: "string", required: false },
    eyebrow: { type: "string", required: false },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
    width: { type: "number", required: false, default: DIVIDER_WIDTH },
  }),
  factory: sectionDividerFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.stepChain — ordered, arrow-linked run of numbered step boxes. Each box is
// a bordered `core.group` (thin `#E4E4E4` border) carrying a green accent edge
// (top border in row mode, left border in column mode, rendered as a child
// accent rect), a mono green number kicker, a bold label, and optional gray
// detail. A green `core.arrow` connects each consecutive pair. Exposes an
// indexed `step[i]` port family.
// ---------------------------------------------------------------------------

type StepEntry = Readonly<{ number: string; label: string; detail?: string }>;

const STEP_ACCENT_THICKNESS = 3;
const STEP_INSET = 14;

// Row-mode box + arrow-gap sizing.
const STEP_ROW_BOX_W = 168;
const STEP_ROW_BOX_H = 116;
const STEP_ROW_ARROW_GAP = 46;
// Column-mode box + arrow-gap sizing.
const STEP_COL_BOX_W = 280;
const STEP_COL_BOX_H = 92;
const STEP_COL_ARROW_GAP = 40;

function parseSteps(
  props: Readonly<Record<string, JsonValue>>,
  componentId: string,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): readonly StepEntry[] | undefined {
  const raw = jsonArray(props.steps);
  if (raw === undefined || raw.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "${componentId}" requires a non-empty "steps" array prop.`,
        sourceMap,
        `Provide "steps" as an array of {number, label, detail?} objects.`,
      ),
    );
    return undefined;
  }

  const steps: StepEntry[] = [];
  let valid = true;
  raw.forEach((rawStep, index) => {
    const record = jsonRecord(rawStep);
    const number = record !== undefined ? jsonString(record.number) : undefined;
    const label = record !== undefined ? jsonString(record.label) : undefined;
    if (number === undefined || label === undefined) {
      valid = false;
      diagnostics.push(
        diagnostic(
          "SDK_PROP_INVALID_TYPE",
          "error",
          `Component "${componentId}" steps[${index}] requires non-empty string "number" and "label".`,
          sourceMap,
          `Provide steps[${index}].number and steps[${index}].label as non-empty strings.`,
        ),
      );
      return;
    }
    const detail = record !== undefined ? jsonString(record.detail) : undefined;
    steps.push({ number, label, ...(detail !== undefined ? { detail } : {}) });
  });

  return valid ? steps : undefined;
}

const stepChainFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.stepChain", context.sourceMap, diagnostics);
  const steps = parseSteps(props, "sdk.stepChain", context.sourceMap, diagnostics);
  if (id === undefined || steps === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const isColumn = stringProp(props, "direction") === "column";
  const boxW = isColumn ? STEP_COL_BOX_W : STEP_ROW_BOX_W;
  const boxH = isColumn ? STEP_COL_BOX_H : STEP_ROW_BOX_H;
  const stride = isColumn ? boxH + STEP_COL_ARROW_GAP : boxW + STEP_ROW_ARROW_GAP;
  const originX = numberProp(props, "x") ?? 0;
  const originY = numberProp(props, "y") ?? 0;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};

  steps.forEach((step, index) => {
    const boxX = isColumn ? 0 : index * stride;
    const boxY = isColumn ? index * stride : 0;
    const boxId = `${rootId}__step-${index}`;
    const boxChildren: RenderNodeIr[] = [];

    // Green accent edge: top bar in row mode, left bar in column mode.
    const accentGeometry: GeometryIr = isColumn
      ? { x: 0, y: 0, width: STEP_ACCENT_THICKNESS, height: boxH }
      : { x: 0, y: 0, width: boxW, height: STEP_ACCENT_THICKNESS };
    boxChildren.push(
      attachSdkOrigin(
        buildRect({
          id: `${boxId}__accent`,
          geometry: accentGeometry,
          style: { fill: COLOR_ACCENT },
          label: `${step.number} accent`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "accent"),
      ),
    );

    boxChildren.push(
      attachSdkOrigin(
        buildText({
          id: `${boxId}__number`,
          text: step.number,
          geometry: { x: STEP_INSET, y: STEP_ACCENT_THICKNESS + 12, width: boxW - STEP_INSET * 2, height: 22 },
          style: { fontSize: 15, fontFamily: MONO_FONT, fontWeight: "bold", fill: COLOR_ACCENT, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "number"),
      ),
    );

    boxChildren.push(
      attachSdkOrigin(
        buildText({
          id: `${boxId}__label`,
          text: step.label,
          geometry: { x: STEP_INSET, y: STEP_ACCENT_THICKNESS + 40, width: boxW - STEP_INSET * 2, height: 26 },
          style: { fontSize: 18, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "label"),
      ),
    );

    if (step.detail !== undefined) {
      boxChildren.push(
        attachSdkOrigin(
          buildText({
            id: `${boxId}__detail`,
            text: step.detail,
            geometry: { x: STEP_INSET, y: STEP_ACCENT_THICKNESS + 70, width: boxW - STEP_INSET * 2, height: 22 },
            style: { fontSize: 13, fill: COLOR_MUTED, textAnchor: "start" },
            sourceMap: context.sourceMap,
          }),
          makeOrigin("sdk.stepChain", context, "detail"),
        ),
      );
    }

    children.push(
      attachSdkOrigin(
        buildGroup({
          id: boxId,
          capabilityId: "core.group",
          geometry: { x: boxX, y: boxY, width: boxW, height: boxH },
          style: {
            coordinateSpace: "local",
            fill: COLOR_SURFACE,
            stroke: COLOR_BORDER,
            strokeWidth: 1,
          },
          children: boxChildren,
          label: `${step.number} ${step.label}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "step"),
      ),
    );
    ports[`step[${index}]`] = { nodeId: boxId };

    // Arrow into the next box (green, in the gap between the two boxes).
    if (index < steps.length - 1) {
      const arrowId = `${rootId}__arrow-${index}`;
      const from: ConnectorEndpointIr = isColumn
        ? { x: boxW / 2, y: boxY + boxH }
        : { x: boxX + boxW, y: boxH / 2 };
      const to: ConnectorEndpointIr = isColumn
        ? { x: boxW / 2, y: boxY + stride }
        : { x: boxX + stride, y: boxH / 2 };
      const arrowGeometry: GeometryIr = {
        x: Math.min(from.x!, to.x!),
        y: Math.min(from.y!, to.y!),
        width: Math.abs(to.x! - from.x!),
        height: Math.abs(to.y! - from.y!),
      };
      children.push(
        attachSdkOrigin(
          buildConnector({
            id: arrowId,
            geometry: arrowGeometry,
            style: { fill: "none", stroke: COLOR_ACCENT, strokeWidth: 2, markerEnd: "arrow" },
            from,
            to,
            label: `step ${index} to ${index + 1}`,
            sourceMap: context.sourceMap,
          }),
          makeOrigin("sdk.stepChain", context, "arrow"),
        ),
      );
      ports[`arrow[${index}]`] = { nodeId: arrowId };
    }
  });

  const lastIndex = steps.length - 1;
  const width = isColumn ? boxW : lastIndex * stride + boxW;
  const height = isColumn ? lastIndex * stride + boxH : boxH;

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x: originX, y: originY, width, height },
      style: { coordinateSpace: "local" },
      children,
      label: "step chain",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.stepChain", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const STEP_CHAIN_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.stepChain", "core.group", {
    direction: { type: "string", required: false, default: "row" },
    steps: { type: "json", required: true },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
  }),
  factory: stepChainFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.bigStat — oversized hero figure (giant green value) with an optional
// bold title and gray description stacked beneath. Root `core.group` of
// `core.text` children. Exposes `value` / `title` / `description` ports.
// ---------------------------------------------------------------------------

const BIG_STAT_VALUE_FONT = 200;
const BIG_STAT_VALUE_H = 220;
const BIG_STAT_TITLE_H = 40;
const BIG_STAT_DESCRIPTION_H = 30;
const BIG_STAT_WIDTH = 480;
const BIG_STAT_GAP = 10;

const bigStatFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.bigStat", context.sourceMap, diagnostics);
  const value = requireStringProp(props, "value", "sdk.bigStat", context.sourceMap, diagnostics);
  if (id === undefined || value === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const title = stringProp(props, "title");
  const description = stringProp(props, "description");
  const width = numberProp(props, "width") ?? BIG_STAT_WIDTH;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};
  let cursorY = 0;

  const valueId = `${rootId}__value`;
  children.push(
    attachSdkOrigin(
      buildText({
        id: valueId,
        text: value,
        geometry: { x: 0, y: cursorY, width, height: BIG_STAT_VALUE_H },
        style: {
          fontSize: BIG_STAT_VALUE_FONT,
          fontWeight: "bold",
          fill: COLOR_ACCENT,
          textAnchor: "start",
        },
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.bigStat", context, "value"),
    ),
  );
  ports.value = { nodeId: valueId };
  cursorY += BIG_STAT_VALUE_H + BIG_STAT_GAP;

  if (title !== undefined) {
    const titleId = `${rootId}__title`;
    children.push(
      attachSdkOrigin(
        buildText({
          id: titleId,
          text: title,
          geometry: { x: 0, y: cursorY, width, height: BIG_STAT_TITLE_H },
          style: { fontSize: 28, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.bigStat", context, "title"),
      ),
    );
    ports.title = { nodeId: titleId };
    cursorY += BIG_STAT_TITLE_H + BIG_STAT_GAP;
  }

  if (description !== undefined) {
    const descriptionId = `${rootId}__description`;
    children.push(
      attachSdkOrigin(
        buildText({
          id: descriptionId,
          text: description,
          geometry: { x: 0, y: cursorY, width, height: BIG_STAT_DESCRIPTION_H },
          style: { fontSize: 16, fill: COLOR_SECONDARY, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.bigStat", context, "description"),
      ),
    );
    ports.description = { nodeId: descriptionId };
    cursorY += BIG_STAT_DESCRIPTION_H + BIG_STAT_GAP;
  }

  const height = Math.max(cursorY - BIG_STAT_GAP, 0);
  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local" },
      children,
      label: title !== undefined ? `${value} ${title}` : value,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.bigStat", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const BIG_STAT_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.bigStat", "core.group", {
    value: { type: "string", required: true },
    title: { type: "string", required: false },
    description: { type: "string", required: false },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
    width: { type: "number", required: false, default: BIG_STAT_WIDTH },
  }),
  factory: bigStatFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.compareGrid — N-column takeaway grid. The root is a `layout.grid`
// container (reusing the `SceneRenderer` grid arrangement, as `sdk.matrix`
// does) over cell groups; each cell is a green-top-accented block with a bold
// label and gray detail. Exposes an indexed `cell[i]` port family.
// ---------------------------------------------------------------------------

type CompareItem = Readonly<{ label: string; detail?: string }>;

const COMPARE_DEFAULT_COLUMNS = 3;
const COMPARE_CELL_W = 220;
const COMPARE_CELL_H = 120;
const COMPARE_GAP = 16;
const COMPARE_INSET = 16;

function parseCompareItems(
  props: Readonly<Record<string, JsonValue>>,
  componentId: string,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): readonly CompareItem[] | undefined {
  const raw = jsonArray(props.items);
  if (raw === undefined || raw.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "${componentId}" requires a non-empty "items" array prop.`,
        sourceMap,
        `Provide "items" as an array of {label, detail?} objects.`,
      ),
    );
    return undefined;
  }

  const items: CompareItem[] = [];
  let valid = true;
  raw.forEach((rawItem, index) => {
    const record = jsonRecord(rawItem);
    const label = record !== undefined ? jsonString(record.label) : undefined;
    if (label === undefined) {
      valid = false;
      diagnostics.push(
        diagnostic(
          "SDK_PROP_INVALID_TYPE",
          "error",
          `Component "${componentId}" items[${index}] requires a non-empty string "label".`,
          sourceMap,
          `Provide items[${index}].label as a non-empty string.`,
        ),
      );
      return;
    }
    const detail = record !== undefined ? jsonString(record.detail) : undefined;
    items.push({ label, ...(detail !== undefined ? { detail } : {}) });
  });

  return valid ? items : undefined;
}

const compareGridFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.compareGrid", context.sourceMap, diagnostics);
  const items = parseCompareItems(props, "sdk.compareGrid", context.sourceMap, diagnostics);
  if (id === undefined || items === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const columns = Math.max(1, Math.round(numberProp(props, "columns") ?? COMPARE_DEFAULT_COLUMNS));
  const gap = numberProp(props, "gap") ?? COMPARE_GAP;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};

  items.forEach((item, index) => {
    const cellId = `${rootId}__cell-${index}`;
    const cellChildren: RenderNodeIr[] = [
      attachSdkOrigin(
        buildRect({
          id: `${cellId}__accent`,
          geometry: { x: 0, y: 0, width: COMPARE_CELL_W, height: STEP_ACCENT_THICKNESS },
          style: { fill: COLOR_ACCENT },
          label: `${item.label} accent`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.compareGrid", context, "accent"),
      ),
      attachSdkOrigin(
        buildText({
          id: `${cellId}__label`,
          text: item.label,
          geometry: {
            x: COMPARE_INSET,
            y: STEP_ACCENT_THICKNESS + 18,
            width: COMPARE_CELL_W - COMPARE_INSET * 2,
            height: 28,
          },
          style: { fontSize: 20, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.compareGrid", context, "label"),
      ),
    ];

    if (item.detail !== undefined) {
      cellChildren.push(
        attachSdkOrigin(
          buildText({
            id: `${cellId}__detail`,
            text: item.detail,
            geometry: {
              x: COMPARE_INSET,
              y: STEP_ACCENT_THICKNESS + 54,
              width: COMPARE_CELL_W - COMPARE_INSET * 2,
              height: 48,
            },
            style: { fontSize: 14, fill: COLOR_MUTED, textAnchor: "start" },
            sourceMap: context.sourceMap,
          }),
          makeOrigin("sdk.compareGrid", context, "detail"),
        ),
      );
    }

    children.push(
      attachSdkOrigin(
        buildGroup({
          id: cellId,
          capabilityId: "core.group",
          geometry: { x: 0, y: 0, width: COMPARE_CELL_W, height: COMPARE_CELL_H },
          style: {
            coordinateSpace: "local",
            fill: COLOR_SURFACE,
            stroke: COLOR_BORDER,
            strokeWidth: 1,
          },
          children: cellChildren,
          label: item.label,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.compareGrid", context, "cell"),
      ),
    );
    ports[`cell[${index}]`] = { nodeId: cellId };
  });

  const rowCount = Math.ceil(items.length / columns);
  const width = columns * COMPARE_CELL_W + (columns - 1) * gap;
  const height = rowCount * COMPARE_CELL_H + (rowCount - 1) * gap;

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "layout.grid",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local", cols: columns, gap },
      children,
      label: "compare grid",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.compareGrid", context, "root"),
  );

  const actions: Partial<Record<SdkActionName, readonly string[]>> = {
    enter: [rootId],
    emphasis: [rootId],
    exit: [rootId],
  };
  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions,
  });
};

export const COMPARE_GRID_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.compareGrid", "layout.grid", {
    columns: { type: "number", required: false, default: COMPARE_DEFAULT_COLUMNS },
    items: { type: "json", required: true },
    gap: { type: "number", required: false, default: COMPARE_GAP },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
  }),
  factory: compareGridFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// Deck composite pack: `sdk/registry.ts` appends this to the generic pack.
// Task 2 appends four more definitions to this array.
// ---------------------------------------------------------------------------

/** Deck-port SDK composite component definitions (working factories). */
export const DECK_COMPOSITE_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  SECTION_DIVIDER_DEFINITION,
  STEP_CHAIN_DEFINITION,
  BIG_STAT_DEFINITION,
  COMPARE_GRID_DEFINITION,
];
