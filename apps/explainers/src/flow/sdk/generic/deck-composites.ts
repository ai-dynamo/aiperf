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
import { measuredWrappedHeight, scaledSceneFontSize } from "../../../core/diagram/text-metrics.js";
import { layoutFlow, type FlowBox, type FlowNode } from "../../../core/diagram/layout/flow-engine.js";
import { textFlowLeaf } from "../../../core/diagram/layout/text-flow-leaf.js";
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
// Auto-grow helpers. These composites emit `core.text` children directly (they
// do NOT flow through the `sdk.paragraph`/etc. catalog factories), so the
// SceneRenderer-mirrored wrap measurement (`measuredWrappedHeight`) must be
// applied here independently to size detail/prose boxes to their wrapped text.
// ---------------------------------------------------------------------------

/**
 * Height a free-text field's own box needs to hold its wrapped content: the
 * greater of the render-accurate wrapped-line stack and the prior fixed height,
 * so single-line specimens never shrink below their original size. `fontSize`
 * is the authored (unscaled) value, matching `SceneRenderer` paint scaling.
 */
function grownTextHeight(
  text: string,
  textWidth: number,
  fontSize: number,
  fixedHeight: number,
  weight: "normal" | "bold" = "normal",
): number {
  return Math.max(measuredWrappedHeight(text, textWidth, fontSize, weight), fixedHeight);
}

/**
 * Height a free-text field's own box needs to hold its wrapped content, sized
 * through the shared flow-layout engine (a single `textFlowLeaf` node) rather
 * than a bespoke line-count/`Math.max` computation — so every text-bearing box
 * routes through one measure-once sizing path. `minHeight` floors the result so
 * single-line specimens never shrink below their original box height, exactly
 * as the prior hand-rolled `Math.max(..., fixedHeight)` did. `fontSize` is the
 * authored (unscaled) value; it is scaled here as `SceneRenderer` paints
 * (`scaledSceneFontSize`) before measurement, matching `textFlowLeaf`'s
 * pre-scaled `scaledFontSize` contract.
 */
function flowTextHeight(
  text: string,
  textWidth: number,
  fontSize: number,
  minHeight: number,
  weight: "normal" | "bold" = "normal",
): number {
  const node: FlowNode = {
    id: "text",
    measure: textFlowLeaf(text, scaledSceneFontSize(fontSize), weight),
    minHeight,
  };
  return layoutFlow(node, { maxWidth: textWidth }).get("text")!.height;
}

// ---------------------------------------------------------------------------
// Grid layout via the flow-layout engine. `sdk.compareGrid` / `sdk.cardGrid`
// model an N-column grid as a `direction: "column"` root of `direction: "row"`
// FlowNodes (one row per grid row), each row `align: "stretch"` so every cell
// in it inherits that row's own cross-size — which the engine computes as the
// max of its children. Each cell is a `direction: "column"` node with a fixed
// column width; its detail body is a `textFlowLeaf` that measures its own
// wrapped height exactly once (cache-based; no measure/paint divergence). This
// yields PER-ROW uniform cell heights (the tallest cell's content in each row)
// without a separate "find the tallest cell" pass, replacing the older
// grid-wide single-height arithmetic.
// ---------------------------------------------------------------------------

/** One grid cell's detail-body measurement inputs (label/accent are fixed). */
type GridCellSpec = Readonly<{
  detail?: string;
  detailFontSize: number;
  detailWeight?: "normal" | "bold";
}>;

/** Flow-engine-resolved grid geometry: per-cell boxes plus overall extent. */
type GridLayoutResult = Readonly<{
  cellBoxes: readonly FlowBox[];
  detailBoxes: readonly (FlowBox | undefined)[];
  width: number;
  height: number;
}>;

/**
 * Resolve per-cell boxes for an N-column grid through the flow-layout engine.
 * Cell widths are fixed to `cellWidth`; each cell's height is the flow engine's
 * per-row max (so a row is uniform-height), and the detail leaf's own resolved
 * box supplies the exact wrapped-text height. Font sizes passed to
 * `textFlowLeaf` are pre-scaled with `scaledSceneFontSize` per its contract.
 */
function computeGridCellBoxes(args: {
  rootId: string;
  cellIdFor: (index: number) => string;
  detailLeafIdFor: (index: number) => string;
  cells: readonly GridCellSpec[];
  columns: number;
  cellWidth: number;
  detailWidth: number;
  detailTopY: number;
  bottomInset: number;
  minCellHeight: number;
  gap: number;
}): GridLayoutResult {
  const rows: FlowNode[] = [];
  for (let rowStart = 0; rowStart < args.cells.length; rowStart += args.columns) {
    const rowCells: FlowNode[] = [];
    for (let column = 0; column < args.columns; column += 1) {
      const index = rowStart + column;
      if (index >= args.cells.length) {
        break;
      }
      const spec = args.cells[index]!;
      const children: FlowNode[] =
        spec.detail !== undefined
          ? [
              {
                id: args.detailLeafIdFor(index),
                measure: textFlowLeaf(
                  spec.detail,
                  scaledSceneFontSize(spec.detailFontSize),
                  spec.detailWeight ?? "normal",
                ),
                fixedWidth: args.detailWidth,
                // Reserve the header band above the detail (accent + label) and
                // the bottom inset; the column sum is the cell's content height.
                margin: { top: args.detailTopY, bottom: args.bottomInset },
              },
            ]
          : [];
      rowCells.push({
        id: args.cellIdFor(index),
        direction: "column",
        fixedWidth: args.cellWidth,
        minHeight: args.minCellHeight,
        children,
      });
    }
    rows.push({
      id: `${args.rootId}__row-${rows.length}`,
      direction: "row",
      align: "stretch",
      columnGap: args.gap,
      children: rowCells,
    });
  }

  const width = args.columns * args.cellWidth + Math.max(args.columns - 1, 0) * args.gap;
  const root: FlowNode = {
    id: args.rootId,
    direction: "column",
    rowGap: args.gap,
    children: rows,
  };
  const out = layoutFlow(root, { maxWidth: width });

  const cellBoxes = args.cells.map(
    (_, index) =>
      out.get(args.cellIdFor(index)) ?? {
        x: 0,
        y: 0,
        width: args.cellWidth,
        height: args.minCellHeight,
      },
  );
  const detailBoxes = args.cells.map((spec, index) =>
    spec.detail !== undefined ? out.get(args.detailLeafIdFor(index)) : undefined,
  );
  const rootHeight = out.get(args.rootId)?.height ?? args.minCellHeight;

  return { cellBoxes, detailBoxes, width, height: rootHeight };
}

/**
 * Convert a flow-engine box (resolved in the layout root's coordinate space)
 * into a `GeometryIr` local to a `parent` group box, since the emitted scene
 * groups use `coordinateSpace: "local"` (children are positioned relative to
 * their parent). `layoutFlow` returns every node's box relative to the same
 * root origin, so subtracting the parent's origin yields the child's local
 * offset.
 */
function localGeometry(child: FlowBox, parent: FlowBox): GeometryIr {
  return {
    x: child.x - parent.x,
    y: child.y - parent.y,
    width: child.width,
    height: child.height,
  };
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

const DIVIDER_WIDTH = 1728;
const DIVIDER_EYEBROW_H = 59.4;
const DIVIDER_NUMBER_H = 399.6;
const DIVIDER_TITLE_H = 172.8;
const DIVIDER_SUBTITLE_H = 91.8;
const DIVIDER_GAP = 21.6;

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
          style: rightAlignedText(37.8, COLOR_ACCENT, { fontWeight: "bold", letterSpacing: 5.4 }),
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
        style: rightAlignedText(324, COLOR_ACCENT, { fontFamily: MONO_FONT, fontWeight: "bold" }),
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
        style: rightAlignedText(129.6, COLOR_INK, { fontWeight: "bold" }),
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.sectionDivider", context, "title"),
    ),
  );
  ports.title = { nodeId: titleId };
  cursorY += DIVIDER_TITLE_H + DIVIDER_GAP;

  if (subtitle !== undefined) {
    const subtitleId = `${rootId}__subtitle`;
    // The subtitle is the one free-form prose field; grow its own box (and thus
    // the divider group) to fit wrapped lines. No sibling shift is needed since
    // it is the last stacked element.
    const subtitleH = flowTextHeight(subtitle, width, 54, DIVIDER_SUBTITLE_H);
    children.push(
      attachSdkOrigin(
        buildText({
          id: subtitleId,
          text: subtitle,
          geometry: { x: 0, y: cursorY, width, height: subtitleH },
          style: rightAlignedText(54, COLOR_SECONDARY, {}),
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.sectionDivider", context, "subtitle"),
      ),
    );
    ports.subtitle = { nodeId: subtitleId };
    cursorY += subtitleH + DIVIDER_GAP;
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

const STEP_ACCENT_THICKNESS = 8.1;
const STEP_INSET = 37.8;

// Row-mode box + arrow-gap sizing.
const STEP_ROW_BOX_W = 453.6;
const STEP_ROW_BOX_H = 313.2;
const STEP_ROW_ARROW_GAP = 124.2;
// Column-mode box + arrow-gap sizing.
const STEP_COL_BOX_W = 756;
const STEP_COL_BOX_H = 248.4;
const STEP_COL_ARROW_GAP = 108;

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
  const baseBoxH = isColumn ? STEP_COL_BOX_H : STEP_ROW_BOX_H;
  const arrowGap = isColumn ? STEP_COL_ARROW_GAP : STEP_ROW_ARROW_GAP;
  const originX = numberProp(props, "x") ?? 0;
  const originY = numberProp(props, "y") ?? 0;

  // Layout is delegated wholesale to the flow engine: one axis-aligned container
  // of fixed-width step boxes, each a `column` of number/label/detail text
  // leaves the engine measures (via `textFlowLeaf`) and sizes. This replaces the
  // former hand-rolled stride/offset/grow arithmetic — box heights (grown to fit
  // wrapped detail) and inter-box placement now fall out of a single
  // `layoutFlow` call. Uniform `STEP_INSET` padding gives the accent-clearing
  // top inset and the left/right text inset in one shot; a `minHeight` keeps
  // detail-free steps at their original box height.
  //
  // Row mode stays a single non-wrapping line: the arrows connect consecutive
  // steps by node anchor, so letting the engine auto-wrap would draw arrows
  // across line breaks. A wrapping run is left to the deck author (multiple
  // stepChain instances) rather than forced here.
  const stepBoxNodes: FlowNode[] = steps.map((step, index) => {
    const boxId = `${rootId}__step-${index}`;
    const inner: FlowNode[] = [
      { id: `${boxId}__number`, measure: textFlowLeaf(step.number, scaledSceneFontSize(40.5), "bold") },
      {
        id: `${boxId}__label`,
        measure: textFlowLeaf(step.label, scaledSceneFontSize(48.6), "bold"),
        margin: { top: 16.2 },
      },
    ];
    if (step.detail !== undefined) {
      inner.push({
        id: `${boxId}__detail`,
        measure: textFlowLeaf(step.detail, scaledSceneFontSize(35.1)),
        margin: { top: 16.2 },
      });
    }
    return {
      id: boxId,
      direction: "column",
      padding: STEP_INSET,
      fixedWidth: boxW,
      minHeight: baseBoxH,
      children: inner,
    };
  });

  const rootNode: FlowNode = {
    id: rootId,
    direction: isColumn ? "column" : "row",
    gap: arrowGap,
    align: "start",
    children: stepBoxNodes,
  };
  const constraintWidth = isColumn
    ? boxW
    : steps.length * boxW + Math.max(steps.length - 1, 0) * arrowGap;
  const boxes = layoutFlow(rootNode, { maxWidth: constraintWidth });
  const rootBox = boxes.get(rootId)!;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};

  steps.forEach((step, index) => {
    const boxId = `${rootId}__step-${index}`;
    const stepBox = boxes.get(boxId)!;
    const boxH = stepBox.height;
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

    const numberBox = boxes.get(`${boxId}__number`)!;
    boxChildren.push(
      attachSdkOrigin(
        buildText({
          id: `${boxId}__number`,
          text: step.number,
          geometry: localGeometry(numberBox, stepBox),
          style: { fontSize: 40.5, fontFamily: MONO_FONT, fontWeight: "bold", fill: COLOR_ACCENT, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "number"),
      ),
    );

    const labelBox = boxes.get(`${boxId}__label`)!;
    boxChildren.push(
      attachSdkOrigin(
        buildText({
          id: `${boxId}__label`,
          text: step.label,
          geometry: localGeometry(labelBox, stepBox),
          style: { fontSize: 48.6, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "label"),
      ),
    );

    if (step.detail !== undefined) {
      const detailBox = boxes.get(`${boxId}__detail`)!;
      boxChildren.push(
        attachSdkOrigin(
          buildText({
            id: `${boxId}__detail`,
            text: step.detail,
            geometry: localGeometry(detailBox, stepBox),
            style: { fontSize: 35.1, fill: COLOR_MUTED, textAnchor: "start" },
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
          geometry: { x: stepBox.x, y: stepBox.y, width: boxW, height: boxH },
          style: {
            coordinateSpace: "local",
            fill: COLOR_SURFACE,
            stroke: COLOR_BORDER,
            strokeWidth: 2.7,
          },
          children: boxChildren,
          label: `${step.number} ${step.label}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.stepChain", context, "step"),
      ),
    );
    ports[`step[${index}]`] = { nodeId: boxId };

    // Arrow into the next box (green, in the gap between the two boxes). The
    // scene arrow resolves its endpoints from the node anchors below; the
    // geometry box is a bounding hint computed from the engine-resolved boxes.
    if (index < steps.length - 1) {
      const nextBox = boxes.get(`${rootId}__step-${index + 1}`)!;
      const from = isColumn
        ? { x: boxW / 2, y: stepBox.y + boxH }
        : { x: stepBox.x + boxW, y: stepBox.y + boxH / 2 };
      const to = isColumn
        ? { x: boxW / 2, y: nextBox.y }
        : { x: nextBox.x, y: nextBox.y + nextBox.height / 2 };
      const arrowId = `${rootId}__arrow-${index}`;
      const arrowGeometry: GeometryIr = {
        x: Math.min(from.x, to.x),
        y: Math.min(from.y, to.y),
        width: Math.abs(to.x - from.x),
        height: Math.abs(to.y - from.y),
      };
      children.push(
        attachSdkOrigin(
          buildConnector({
            id: arrowId,
            geometry: arrowGeometry,
            style: { fill: "none", stroke: COLOR_ACCENT, strokeWidth: 5.4, markerEnd: "arrow" },
            from: { nodeId: `${rootId}__step-${index}`, anchor: isColumn ? "s" : "e" },
            to: { nodeId: `${rootId}__step-${index + 1}`, anchor: isColumn ? "n" : "w" },
            label: `step ${index} to ${index + 1}`,
            sourceMap: context.sourceMap,
          }),
          makeOrigin("sdk.stepChain", context, "arrow"),
        ),
      );
      ports[`arrow[${index}]`] = { nodeId: arrowId };
    }
  });

  const width = rootBox.width;
  const height = rootBox.height;

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

const BIG_STAT_VALUE_FONT = 540;
const BIG_STAT_VALUE_H = 594;
const BIG_STAT_TITLE_H = 108;
const BIG_STAT_DESCRIPTION_H = 81;
const BIG_STAT_WIDTH = 1296;
const BIG_STAT_GAP = 27;

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
          style: { fontSize: 75.6, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
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
    // Description is the one free-form prose field; grow its own box (and the
    // stat group) to fit wrapped lines. It is the last stacked element, so no
    // sibling shift is needed.
    const descriptionH = flowTextHeight(description, width, 43.2, BIG_STAT_DESCRIPTION_H);
    children.push(
      attachSdkOrigin(
        buildText({
          id: descriptionId,
          text: description,
          geometry: { x: 0, y: cursorY, width, height: descriptionH },
          style: { fontSize: 43.2, fill: COLOR_SECONDARY, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.bigStat", context, "description"),
      ),
    );
    ports.description = { nodeId: descriptionId };
    cursorY += descriptionH + BIG_STAT_GAP;
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
const COMPARE_CELL_W = 594;
const COMPARE_CELL_H = 324;
const COMPARE_GAP = 43.2;
const COMPARE_INSET = 43.2;

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

  // Per-ROW uniform cell heights via the flow-layout engine: every cell in a
  // row auto-sizes to that row's tallest wrapped-detail content (see
  // `computeGridCellBoxes`), replacing the older grid-wide single-height math.
  const detailWidth = COMPARE_CELL_W - COMPARE_INSET * 2;
  const detailTopY = STEP_ACCENT_THICKNESS + 145.8;
  const layout = computeGridCellBoxes({
    rootId,
    cellIdFor: (index) => `${rootId}__cell-${index}`,
    detailLeafIdFor: (index) => `${rootId}__cell-${index}__detail`,
    cells: items.map((item) => ({
      ...(item.detail !== undefined ? { detail: item.detail } : {}),
      detailFontSize: 37.8,
    })),
    columns,
    cellWidth: COMPARE_CELL_W,
    detailWidth,
    detailTopY,
    bottomInset: COMPARE_INSET,
    minCellHeight: COMPARE_CELL_H,
    gap,
  });

  items.forEach((item, index) => {
    const cellId = `${rootId}__cell-${index}`;
    const cellBox = layout.cellBoxes[index]!;
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
            y: STEP_ACCENT_THICKNESS + 48.6,
            width: COMPARE_CELL_W - COMPARE_INSET * 2,
            height: 75.6,
          },
          style: { fontSize: 54, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.compareGrid", context, "label"),
      ),
    ];

    if (item.detail !== undefined) {
      const detailBox = layout.detailBoxes[index];
      cellChildren.push(
        attachSdkOrigin(
          buildText({
            id: `${cellId}__detail`,
            text: item.detail,
            geometry: {
              x: COMPARE_INSET,
              y: detailTopY,
              width: detailWidth,
              height: detailBox?.height ?? grownTextHeight(item.detail, detailWidth, 37.8, 129.6),
            },
            style: { fontSize: 37.8, fill: COLOR_MUTED, textAnchor: "start" },
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
          geometry: { x: cellBox.x, y: cellBox.y, width: COMPARE_CELL_W, height: cellBox.height },
          style: {
            coordinateSpace: "local",
            fill: COLOR_SURFACE,
            stroke: COLOR_BORDER,
            strokeWidth: 2.7,
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

  const width = layout.width;
  const height = layout.height;

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local" },
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
  descriptor: makeDescriptor("sdk.compareGrid", "core.group", {
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
// Extra palette token for the black index chips / accents that alternate with
// the green ones in the "Observer sequence" and "Crate topology" slides.
// ---------------------------------------------------------------------------

const COLOR_INK_FILL = "#000000"; // solid black chip / accent fill

/** Read an optional boolean flag from a loosely-typed record. */
function jsonFlag(value: JsonValue | undefined): boolean {
  return value === true;
}

// ---------------------------------------------------------------------------
// sdk.numberedSequence — vertical stack of rows. Each row is a small square
// index chip on the left (green fill when `emphasis`, black otherwise, mirroring
// the deck's alternating `1..6` squares) carrying the `number`, and a bordered
// box to its right with a bold mono `title` and gray `detail`. Root
// `core.group`; exposes an indexed `row[i]` port family. Source: the "Observer
// sequence" slide's `on_arrival` / `on_admit` / `on_token` / ... callback list.
// ---------------------------------------------------------------------------

type SequenceEntry = Readonly<{
  number: string;
  title: string;
  detail?: string;
  emphasis: boolean;
}>;

const SEQ_CHIP = 118.8; // square index chip edge length
const SEQ_ROW_H = 151.2; // box height per row
const SEQ_ROW_GAP = 32.4; // vertical gap between rows
const SEQ_CHIP_GAP = 37.8; // horizontal gap chip → box
const SEQ_BOX_W = 1026; // bordered detail box width
const SEQ_INSET = 37.8;

function parseSequenceItems(
  props: Readonly<Record<string, JsonValue>>,
  componentId: string,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): readonly SequenceEntry[] | undefined {
  const raw = jsonArray(props.items);
  if (raw === undefined || raw.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "${componentId}" requires a non-empty "items" array prop.`,
        sourceMap,
        `Provide "items" as an array of {number, title, detail?, emphasis?} objects.`,
      ),
    );
    return undefined;
  }

  const items: SequenceEntry[] = [];
  let valid = true;
  raw.forEach((rawItem, index) => {
    const record = jsonRecord(rawItem);
    const number = record !== undefined ? jsonString(record.number) : undefined;
    const title = record !== undefined ? jsonString(record.title) : undefined;
    if (number === undefined || title === undefined) {
      valid = false;
      diagnostics.push(
        diagnostic(
          "SDK_PROP_INVALID_TYPE",
          "error",
          `Component "${componentId}" items[${index}] requires non-empty string "number" and "title".`,
          sourceMap,
          `Provide items[${index}].number and items[${index}].title as non-empty strings.`,
        ),
      );
      return;
    }
    const detail = record !== undefined ? jsonString(record.detail) : undefined;
    const emphasis = record !== undefined ? jsonFlag(record.emphasis) : false;
    items.push({ number, title, emphasis, ...(detail !== undefined ? { detail } : {}) });
  });

  return valid ? items : undefined;
}

const numberedSequenceFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.numberedSequence", context.sourceMap, diagnostics);
  const items = parseSequenceItems(props, "sdk.numberedSequence", context.sourceMap, diagnostics);
  if (id === undefined || items === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;
  const boxX = SEQ_CHIP + SEQ_CHIP_GAP;
  const seqDetailWidth = SEQ_BOX_W - SEQ_INSET * 2;

  // Layout is delegated to the flow engine: a `column` of rows, each a `row` of
  // a fixed-size chip leaf and a bordered box. The box holds an inset content
  // column of title/detail text leaves the engine measures and sizes, so a
  // wrapped detail grows its box (and hence the row and the whole stack) with no
  // running y-offset accumulator. `align: "center"` on each row vertically
  // centers the shorter chip against the taller box; `justify: "center"` on the
  // box vertically centers the title/detail block within the (min-`SEQ_ROW_H`)
  // box. The `SEQ_INSET` horizontal inset is carried by a fixed-width content
  // node (not box padding) so wrap width stays `seqDetailWidth` WITHOUT adding
  // vertical padding — this keeps box heights matched to the prior layout.
  const rowNodes: FlowNode[] = items.map((item, index) => {
    const rowId = `${rootId}__row-${index}`;
    const boxInner: FlowNode[] = [
      { id: `${rowId}__title`, measure: textFlowLeaf(item.title, scaledSceneFontSize(43.2), "bold") },
    ];
    if (item.detail !== undefined) {
      boxInner.push({
        id: `${rowId}__detail`,
        measure: textFlowLeaf(item.detail, scaledSceneFontSize(35.1)),
        margin: { top: 10.8 },
      });
    }
    return {
      id: rowId,
      direction: "row",
      gap: SEQ_CHIP_GAP,
      align: "center",
      children: [
        { id: `${rowId}__chip`, fixedWidth: SEQ_CHIP, fixedHeight: SEQ_CHIP },
        {
          id: `${rowId}__box`,
          direction: "column",
          fixedWidth: SEQ_BOX_W,
          minHeight: SEQ_ROW_H,
          justify: "center",
          children: [
            {
              id: `${rowId}__box-content`,
              direction: "column",
              fixedWidth: seqDetailWidth,
              margin: { left: SEQ_INSET, right: SEQ_INSET },
              children: boxInner,
            },
          ],
        },
      ],
    };
  });

  const rootNode: FlowNode = {
    id: rootId,
    direction: "column",
    gap: SEQ_ROW_GAP,
    children: rowNodes,
  };
  const boxes = layoutFlow(rootNode, { maxWidth: boxX + SEQ_BOX_W });
  const rootBox = boxes.get(rootId)!;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};

  items.forEach((item, index) => {
    const rowId = `${rootId}__row-${index}`;
    const rowBox = boxes.get(rowId)!;
    const chipBox = boxes.get(`${rowId}__chip`)!;
    const boxBox = boxes.get(`${rowId}__box`)!;
    const rowChildren: RenderNodeIr[] = [];

    // Index chip: a filled square (green when emphasized, solid black otherwise)
    // with the number centered on top of it. The fill and the number
    // legitimately occupy the same square, so they are wrapped in a
    // `layout.overlay` chip container (the sanctioned home for intentional
    // overlap) with both children marked absolute — the fill still paints and
    // the number still reads, but the pair is not a flat sibling collision.
    const chipFill = item.emphasis ? COLOR_ACCENT : COLOR_INK_FILL;
    const chipChildren: RenderNodeIr[] = [
      attachSdkOrigin(
        buildRect({
          id: `${rowId}__chip-fill`,
          geometry: { x: 0, y: 0, width: SEQ_CHIP, height: SEQ_CHIP },
          style: { position: "absolute", fill: chipFill },
          label: `${item.number} chip`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.numberedSequence", context, "chip"),
      ),
      attachSdkOrigin(
        buildText({
          id: `${rowId}__number`,
          text: item.number,
          geometry: { x: 0, y: 32.4, width: SEQ_CHIP, height: 59.4 },
          style: {
            position: "absolute",
            fontSize: 48.6,
            fontFamily: MONO_FONT,
            fontWeight: "bold",
            fill: COLOR_SURFACE,
            textAnchor: "middle",
          },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.numberedSequence", context, "number"),
      ),
    ];
    rowChildren.push(
      attachSdkOrigin(
        buildGroup({
          id: `${rowId}__chip`,
          capabilityId: "layout.overlay",
          geometry: localGeometry(chipBox, rowBox),
          style: { coordinateSpace: "local" },
          children: chipChildren,
          label: `${item.number} chip`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.numberedSequence", context, "chip"),
      ),
    );

    // Bordered detail box to the right of the chip.
    const titleBox = boxes.get(`${rowId}__title`)!;
    const boxChildren: RenderNodeIr[] = [
      attachSdkOrigin(
        buildText({
          id: `${rowId}__title`,
          text: item.title,
          geometry: localGeometry(titleBox, boxBox),
          style: { fontSize: 43.2, fontFamily: MONO_FONT, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.numberedSequence", context, "title"),
      ),
    ];
    if (item.detail !== undefined) {
      const detailBox = boxes.get(`${rowId}__detail`)!;
      boxChildren.push(
        attachSdkOrigin(
          buildText({
            id: `${rowId}__detail`,
            text: item.detail,
            geometry: localGeometry(detailBox, boxBox),
            style: { fontSize: 35.1, fill: COLOR_MUTED, textAnchor: "start" },
            sourceMap: context.sourceMap,
          }),
          makeOrigin("sdk.numberedSequence", context, "detail"),
        ),
      );
    }
    rowChildren.push(
      attachSdkOrigin(
        buildGroup({
          id: `${rowId}__box`,
          capabilityId: "core.group",
          geometry: { ...localGeometry(boxBox, rowBox), width: SEQ_BOX_W, height: boxBox.height },
          style: { coordinateSpace: "local", fill: COLOR_SURFACE, stroke: COLOR_BORDER, strokeWidth: 2.7 },
          children: boxChildren,
          label: item.title,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.numberedSequence", context, "box"),
      ),
    );

    children.push(
      attachSdkOrigin(
        buildGroup({
          id: rowId,
          capabilityId: "core.group",
          geometry: { x: rowBox.x, y: rowBox.y, width: rowBox.width, height: rowBox.height },
          style: { coordinateSpace: "local" },
          children: rowChildren,
          label: `${item.number} ${item.title}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.numberedSequence", context, "row"),
      ),
    );
    ports[`row[${index}]`] = { nodeId: rowId };
  });

  const width = rootBox.width;
  const height = rootBox.height;
  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local" },
      children,
      label: "numbered sequence",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.numberedSequence", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const NUMBERED_SEQUENCE_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.numberedSequence", "core.group", {
    items: { type: "json", required: true },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
  }),
  factory: numberedSequenceFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.timelineAxis — a horizontal reference axis spanning `start`..`end` in
// domain units. Tick marks + labels sit under the axis at each `ticks` entry, a
// circle marker (filled green for `"exact"`, hollow gray for `"late"`) sits at
// each `markers` entry, and an optional `target` draws a dashed vertical
// reference line with its label. Root `core.group`; exposes indexed
// `tick[i]` / `marker[i]` and (when present) `target` ports. Source: the "Clock"
// slide's RealClock diagram.
// ---------------------------------------------------------------------------

type AxisTick = Readonly<{ at: number; label: string }>;
type AxisMarker = Readonly<{ at: number; label: string; late: boolean }>;

const AXIS_WIDTH = 1728; // pixel span of the axis line
const AXIS_LINE_Y = 162; // y of the horizontal axis within the group
const AXIS_LINE_THICKNESS = 5.4;
const AXIS_TICK_H = 27; // tick mark height below the axis
const AXIS_MARKER_R = 21.6; // marker circle radius
const AXIS_TARGET_TOP = 32.4; // dashed target line top y
const AXIS_HEIGHT = 345.6;
const AXIS_LABEL_FONT = 32.4; // tick / marker / target caption font size
const AXIS_TICK_LABEL_W = 216; // tick caption box width
const AXIS_MARKER_LABEL_W = 324; // marker / target caption box width
const AXIS_LABEL_H = 43.2; // single-line caption box height floor

function parseAxisTicks(value: JsonValue | undefined): readonly AxisTick[] {
  const raw = jsonArray(value);
  if (raw === undefined) {
    return [];
  }
  const ticks: AxisTick[] = [];
  raw.forEach((rawTick) => {
    const record = jsonRecord(rawTick);
    const at = record !== undefined ? jsonNumber(record.at) : undefined;
    const label = record !== undefined ? jsonString(record.label) : undefined;
    if (at !== undefined && label !== undefined) {
      ticks.push({ at, label });
    }
  });
  return ticks;
}

function parseAxisMarkers(value: JsonValue | undefined): readonly AxisMarker[] {
  const raw = jsonArray(value);
  if (raw === undefined) {
    return [];
  }
  const markers: AxisMarker[] = [];
  raw.forEach((rawMarker) => {
    const record = jsonRecord(rawMarker);
    const at = record !== undefined ? jsonNumber(record.at) : undefined;
    const label = record !== undefined ? jsonString(record.label) : undefined;
    if (at !== undefined && label !== undefined) {
      markers.push({ at, label, late: record?.style === "late" });
    }
  });
  return markers;
}

const timelineAxisFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.timelineAxis", context.sourceMap, diagnostics);
  const start = numberProp(props, "start");
  const end = numberProp(props, "end");
  if (start === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "sdk.timelineAxis" requires a finite number prop "start".`,
        context.sourceMap,
        `Provide "start" as a finite number.`,
      ),
    );
  }
  if (end === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "sdk.timelineAxis" requires a finite number prop "end".`,
        context.sourceMap,
        `Provide "end" as a finite number.`,
      ),
    );
  }
  if (id === undefined || start === undefined || end === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const width = numberProp(props, "width") ?? AXIS_WIDTH;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;
  const span = end - start;
  // Degenerate zero-span axes collapse every value onto the left edge.
  const toX = (at: number): number => (span === 0 ? 0 : ((at - start) / span) * width);

  const ticks = parseAxisTicks(props.ticks);
  const markers = parseAxisMarkers(props.markers);
  const targetRecord = jsonRecord(props.target);
  const targetAt = targetRecord !== undefined ? jsonNumber(targetRecord.at) : undefined;
  const targetLabel = targetRecord !== undefined ? jsonString(targetRecord.label) : undefined;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};

  // Horizontal axis line (thin rect).
  const axisId = `${rootId}__axis`;
  children.push(
    attachSdkOrigin(
      buildRect({
        id: axisId,
        geometry: { x: 0, y: AXIS_LINE_Y, width, height: AXIS_LINE_THICKNESS },
        style: { fill: COLOR_INK },
        label: `axis ${start} to ${end}`,
        sourceMap: context.sourceMap,
      }),
      makeOrigin("sdk.timelineAxis", context, "axis"),
    ),
  );
  ports.axis = { nodeId: axisId };

  // Optional dashed vertical target reference line + label.
  if (targetAt !== undefined && targetLabel !== undefined) {
    const targetX = toX(targetAt);
    const targetLineId = `${rootId}__target`;
    children.push(
      attachSdkOrigin(
        buildRect({
          id: targetLineId,
          // The deadline drop-line runs from above the axis down to the axis
          // line (stopping at it, not crossing it) so it does not overlap the
          // axis, ticks, or tick labels as a flat sibling.
          geometry: { x: targetX, y: AXIS_TARGET_TOP, width: 2.7, height: AXIS_LINE_Y - AXIS_TARGET_TOP },
          style: { fill: COLOR_ACCENT, strokeDasharray: "10.8 8.1" },
          label: `target ${targetLabel}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.timelineAxis", context, "target"),
      ),
    );
    children.push(
      attachSdkOrigin(
        buildText({
          id: `${targetLineId}__label`,
          text: targetLabel,
          // Sits fully above the drop-line's top edge; the caption box is sized
          // through the shared flow engine (wraps at its width, floors at the
          // single-line height) rather than a fixed line box.
          geometry: {
            x: targetX - AXIS_MARKER_LABEL_W / 2,
            y: AXIS_TARGET_TOP - 54,
            width: AXIS_MARKER_LABEL_W,
            height: flowTextHeight(targetLabel, AXIS_MARKER_LABEL_W, AXIS_LABEL_FONT, AXIS_LABEL_H, "bold"),
          },
          style: { fontSize: AXIS_LABEL_FONT, fontWeight: "bold", fill: COLOR_ACCENT, textAnchor: "middle" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.timelineAxis", context, "targetLabel"),
      ),
    );
    ports.target = { nodeId: targetLineId };
  }

  // Tick marks + labels below the axis.
  ticks.forEach((tick, index) => {
    const tickX = toX(tick.at);
    const tickId = `${rootId}__tick-${index}`;
    children.push(
      attachSdkOrigin(
        buildRect({
          id: tickId,
          // Tick marks hang just below the axis line (not starting on it) so they
          // touch rather than overlap the axis rect as flat siblings.
          geometry: { x: tickX, y: AXIS_LINE_Y + AXIS_LINE_THICKNESS, width: 2.7, height: AXIS_TICK_H },
          style: { fill: COLOR_SECONDARY },
          label: `tick ${tick.label}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.timelineAxis", context, "tick"),
      ),
    );
    children.push(
      attachSdkOrigin(
        buildText({
          id: `${tickId}__label`,
          text: tick.label,
          geometry: {
            x: tickX - AXIS_TICK_LABEL_W / 2,
            y: AXIS_LINE_Y + AXIS_LINE_THICKNESS + AXIS_TICK_H + 10.8,
            width: AXIS_TICK_LABEL_W,
            height: flowTextHeight(tick.label, AXIS_TICK_LABEL_W, AXIS_LABEL_FONT, AXIS_LABEL_H),
          },
          style: { fontSize: AXIS_LABEL_FONT, fill: COLOR_SECONDARY, textAnchor: "middle" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.timelineAxis", context, "tickLabel"),
      ),
    );
    ports[`tick[${index}]`] = { nodeId: tickId };
  });

  // Circle markers above the axis: filled green (exact) or hollow gray (late).
  markers.forEach((marker, index) => {
    const markerX = toX(marker.at);
    const markerId = `${rootId}__marker-${index}`;
    const markerCy = AXIS_LINE_Y - AXIS_MARKER_R - 16.2;
    // core.rect used as a circle proxy via borderRadius so we avoid a new
    // capability; a full radius on a square reads as a disc.
    children.push(
      attachSdkOrigin(
        buildRect({
          id: markerId,
          geometry: {
            x: markerX - AXIS_MARKER_R,
            y: markerCy - AXIS_MARKER_R,
            width: AXIS_MARKER_R * 2,
            height: AXIS_MARKER_R * 2,
          },
          style: marker.late
            ? { fill: "none", stroke: COLOR_MUTED, strokeWidth: 5.4, borderRadius: AXIS_MARKER_R }
            : { fill: COLOR_ACCENT, borderRadius: AXIS_MARKER_R },
          label: `marker ${marker.label}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.timelineAxis", context, "marker"),
      ),
    );
    children.push(
      attachSdkOrigin(
        buildText({
          id: `${markerId}__label`,
          text: marker.label,
          geometry: {
            x: markerX - AXIS_MARKER_LABEL_W / 2,
            y: markerCy - AXIS_MARKER_R - 48.6,
            width: AXIS_MARKER_LABEL_W,
            height: flowTextHeight(marker.label, AXIS_MARKER_LABEL_W, AXIS_LABEL_FONT, AXIS_LABEL_H, "bold"),
          },
          style: {
            fontSize: AXIS_LABEL_FONT,
            fontWeight: "bold",
            fill: marker.late ? COLOR_MUTED : COLOR_ACCENT,
            textAnchor: "middle",
          },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.timelineAxis", context, "markerLabel"),
      ),
    );
    ports[`marker[${index}]`] = { nodeId: markerId };
  });

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height: AXIS_HEIGHT },
      style: { coordinateSpace: "local" },
      children,
      label: `timeline ${start} to ${end}`,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.timelineAxis", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const TIMELINE_AXIS_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.timelineAxis", "core.group", {
    start: { type: "number", required: true },
    end: { type: "number", required: true },
    unit: { type: "string", required: false },
    ticks: { type: "json", required: false },
    markers: { type: "json", required: false },
    target: { type: "json", required: false },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
    width: { type: "number", required: false, default: AXIS_WIDTH },
  }),
  factory: timelineAxisFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.nodeTree — one root box over a row of child boxes, each connected to the
// root by a `core.arrow` line, with an optional `orderNote` caption underneath.
// The root box fills green when the root's own `emphasis` is set (the deck's
// "popped first" element), children fill white unless individually emphasized.
// Root `core.group`; exposes `rootBox` and indexed `child[i]` ports. Source: the
// "Clock" slide's SimClock `BinaryHeap<Sleeper>` diagram.
// ---------------------------------------------------------------------------

type TreeNode = Readonly<{ label: string; detail?: string; emphasis: boolean }>;

const TREE_BOX_W = 405;
const TREE_BOX_H = 162;
const TREE_CHILD_GAP = 108; // horizontal gap between child boxes
const TREE_LEVEL_GAP = 189; // vertical gap root → children row
const TREE_CAPTION_H = 64.8;
const TREE_INSET = 32.4;

function parseTreeNode(
  value: JsonValue | undefined,
): Readonly<{ label: string; detail?: string; emphasis: boolean }> | undefined {
  const record = jsonRecord(value);
  const label = record !== undefined ? jsonString(record.label) : undefined;
  if (label === undefined) {
    return undefined;
  }
  const detail = record !== undefined ? jsonString(record.detail) : undefined;
  const emphasis = record !== undefined ? jsonFlag(record.emphasis) : false;
  return { label, emphasis, ...(detail !== undefined ? { detail } : {}) };
}

function buildTreeBox(
  args: {
    componentId: string;
    role: string;
    id: string;
    geometry: GeometryIr;
    node: TreeNode;
  },
  context: SdkExpansionContext,
): GroupNodeIr {
  const filled = args.node.emphasis;
  const boxChildren: RenderNodeIr[] = [
    attachSdkOrigin(
      buildRect({
        id: `${args.id}__backdrop`,
        geometry: { x: 0, y: 0, width: TREE_BOX_W, height: TREE_BOX_H },
        style: {
          position: "absolute",
          fill: filled ? COLOR_ACCENT : COLOR_SURFACE,
          stroke: filled ? COLOR_ACCENT : COLOR_BORDER,
          strokeWidth: 2.7,
        },
        label: `${args.node.label} backdrop`,
        sourceMap: context.sourceMap,
      }),
      makeOrigin(args.componentId, context, `${args.role}Backdrop`),
    ),
    attachSdkOrigin(
      buildText({
        id: `${args.id}__label`,
        text: args.node.label,
        geometry: {
          x: TREE_INSET,
          y: args.node.detail !== undefined ? 32.4 : (TREE_BOX_H - 54) / 2,
          width: TREE_BOX_W - TREE_INSET * 2,
          height: 54,
        },
        style: {
          position: "absolute",
          fontSize: 43.2,
          fontFamily: MONO_FONT,
          fontWeight: "bold",
          fill: filled ? COLOR_SURFACE : COLOR_INK,
          textAnchor: "middle",
        },
        sourceMap: context.sourceMap,
      }),
      makeOrigin(args.componentId, context, `${args.role}Label`),
    ),
  ];
  if (args.node.detail !== undefined) {
    boxChildren.push(
      attachSdkOrigin(
        buildText({
          id: `${args.id}__detail`,
          text: args.node.detail,
          geometry: { x: TREE_INSET, y: 91.8, width: TREE_BOX_W - TREE_INSET * 2, height: 43.2 },
          style: {
            position: "absolute",
            fontSize: 32.4,
            fill: filled ? COLOR_SURFACE : COLOR_MUTED,
            textAnchor: "middle",
          },
          sourceMap: context.sourceMap,
        }),
        makeOrigin(args.componentId, context, `${args.role}Detail`),
      ),
    );
  }
  return attachSdkOrigin(
    buildGroup({
      id: args.id,
      capabilityId: "layout.overlay",
      geometry: args.geometry,
      style: { coordinateSpace: "local" },
      children: boxChildren,
      label: args.node.label,
      sourceMap: context.sourceMap,
    }),
    makeOrigin(args.componentId, context, args.role),
  );
}

const nodeTreeFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.nodeTree", context.sourceMap, diagnostics);
  const root = parseTreeNode(props.root);
  if (root === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "sdk.nodeTree" requires a "root" object with a non-empty string "label".`,
        context.sourceMap,
        `Provide "root" as {label, detail?, emphasis?}.`,
      ),
    );
  }
  const rawChildren = jsonArray(props.children);
  if (rawChildren === undefined || rawChildren.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "sdk.nodeTree" requires a non-empty "children" array prop.`,
        context.sourceMap,
        `Provide "children" as an array of {label, detail?, emphasis?} objects.`,
      ),
    );
  }
  if (id === undefined || root === undefined || rawChildren === undefined || rawChildren.length === 0) {
    return fail(diagnostics);
  }

  const childNodes: TreeNode[] = [];
  let valid = true;
  rawChildren.forEach((rawChild, index) => {
    const node = parseTreeNode(rawChild);
    if (node === undefined) {
      valid = false;
      diagnostics.push(
        diagnostic(
          "SDK_PROP_INVALID_TYPE",
          "error",
          `Component "sdk.nodeTree" children[${index}] requires a non-empty string "label".`,
          context.sourceMap,
          `Provide children[${index}].label as a non-empty string.`,
        ),
      );
      return;
    }
    childNodes.push(node);
  });
  if (!valid) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;
  const orderNote = stringProp(props, "orderNote");

  const childrenRowWidth =
    childNodes.length * TREE_BOX_W + (childNodes.length - 1) * TREE_CHILD_GAP;
  const width = Math.max(childrenRowWidth, TREE_BOX_W);
  const childrenY = TREE_BOX_H + TREE_LEVEL_GAP;

  const nodes: RenderNodeIr[] = [];

  // Root box centered over the children row.
  const rootBoxId = `${rootId}__root`;
  const rootBoxX = (width - TREE_BOX_W) / 2;
  nodes.push(
    buildTreeBox(
      {
        componentId: "sdk.nodeTree",
        role: "rootBox",
        id: rootBoxId,
        geometry: { x: rootBoxX, y: 0, width: TREE_BOX_W, height: TREE_BOX_H },
        node: root,
      },
      context,
    ),
  );

  const ports: Record<string, ConnectorEndpointIr> = { rootBox: { nodeId: rootBoxId } };

  childNodes.forEach((node, index) => {
    const childX = index * (TREE_BOX_W + TREE_CHILD_GAP);
    const childBoxId = `${rootId}__child-${index}`;

    // Connecting line from root bottom-center to child top-center.
    const from: ConnectorEndpointIr = { x: rootBoxX + TREE_BOX_W / 2, y: TREE_BOX_H };
    const to: ConnectorEndpointIr = { x: childX + TREE_BOX_W / 2, y: childrenY };
    const lineId = `${rootId}__line-${index}`;
    const lineGeometry: GeometryIr = {
      x: Math.min(from.x!, to.x!),
      y: Math.min(from.y!, to.y!),
      width: Math.abs(to.x! - from.x!),
      height: Math.abs(to.y! - from.y!),
    };
    nodes.push(
      attachSdkOrigin(
        buildConnector({
          id: lineId,
          geometry: lineGeometry,
          style: { fill: "none", stroke: COLOR_BORDER, strokeWidth: 4.05 },
          from: { nodeId: rootBoxId, anchor: "s" },
          to: { nodeId: childBoxId, anchor: "n" },
          label: `root to child ${index}`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.nodeTree", context, "line"),
      ),
    );

    nodes.push(
      buildTreeBox(
        {
          componentId: "sdk.nodeTree",
          role: "child",
          id: childBoxId,
          geometry: { x: childX, y: childrenY, width: TREE_BOX_W, height: TREE_BOX_H },
          node,
        },
        context,
      ),
    );
    ports[`child[${index}]`] = { nodeId: childBoxId };
  });

  let height = childrenY + TREE_BOX_H;
  if (orderNote !== undefined) {
    const captionId = `${rootId}__caption`;
    // The order-note caption is the one free-form prose field; grow its own box
    // (and the tree group) to fit wrapped lines. It is the last element, so no
    // sibling shift is needed.
    const captionH = flowTextHeight(orderNote, width, 35.1, TREE_CAPTION_H);
    nodes.push(
      attachSdkOrigin(
        buildText({
          id: captionId,
          text: orderNote,
          geometry: { x: 0, y: height + 21.6, width, height: captionH },
          style: { fontSize: 35.1, fill: COLOR_SECONDARY, textAnchor: "middle" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.nodeTree", context, "caption"),
      ),
    );
    ports.caption = { nodeId: captionId };
    height += 21.6 + captionH;
  }

  const rootNode = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local" },
      children: nodes,
      label: `node tree ${root.label}`,
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.nodeTree", context, "root"),
  );

  return succeed({
    roots: [rootNode],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const NODE_TREE_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.nodeTree", "core.group", {
    root: { type: "json", required: true },
    children: { type: "json", required: true },
    orderNote: { type: "string", required: false },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
  }),
  factory: nodeTreeFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// sdk.cardGrid — a `layout.grid` of bordered cards (reusing the grid layout as
// `sdk.compareGrid` does), each with a left-accent-colored border strip
// (green/black/gray per `accent`, default gray), a bold mono `title`, and gray
// `detail` body text. Exposes an indexed `card[i]` port family. Source: the
// "Crate topology" slide's 4-card grid.
// ---------------------------------------------------------------------------

type CardAccent = "green" | "black" | "gray";
type CardEntry = Readonly<{ title: string; detail: string; accent: CardAccent }>;

const CARD_DEFAULT_COLUMNS = 2;
const CARD_W = 702;
const CARD_H = 291.6;
const CARD_GAP = 43.2;
const CARD_INSET = 48.6;
const CARD_ACCENT_W = 10.8; // left border strip width

function cardAccentColor(accent: CardAccent): string {
  switch (accent) {
    case "green":
      return COLOR_ACCENT;
    case "black":
      return COLOR_INK_FILL;
    default:
      return COLOR_MUTED;
  }
}

function parseCards(
  props: Readonly<Record<string, JsonValue>>,
  componentId: string,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): readonly CardEntry[] | undefined {
  const raw = jsonArray(props.cards);
  if (raw === undefined || raw.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_PROP_REQUIRED",
        "error",
        `Component "${componentId}" requires a non-empty "cards" array prop.`,
        sourceMap,
        `Provide "cards" as an array of {title, detail, accent?} objects.`,
      ),
    );
    return undefined;
  }

  const cards: CardEntry[] = [];
  let valid = true;
  raw.forEach((rawCard, index) => {
    const record = jsonRecord(rawCard);
    const title = record !== undefined ? jsonString(record.title) : undefined;
    const detail = record !== undefined ? jsonString(record.detail) : undefined;
    if (title === undefined || detail === undefined) {
      valid = false;
      diagnostics.push(
        diagnostic(
          "SDK_PROP_INVALID_TYPE",
          "error",
          `Component "${componentId}" cards[${index}] requires non-empty string "title" and "detail".`,
          sourceMap,
          `Provide cards[${index}].title and cards[${index}].detail as non-empty strings.`,
        ),
      );
      return;
    }
    const rawAccent = record !== undefined ? jsonString(record.accent) : undefined;
    const accent: CardAccent =
      rawAccent === "green" || rawAccent === "black" ? rawAccent : "gray";
    cards.push({ title, detail, accent });
  });

  return valid ? cards : undefined;
}

const cardGridFactory: SdkComponentFactory = (props, _slots, context) => {
  const diagnostics: Diagnostic[] = [];
  const id = requireStringProp(props, "id", "sdk.cardGrid", context.sourceMap, diagnostics);
  const cards = parseCards(props, "sdk.cardGrid", context.sourceMap, diagnostics);
  if (id === undefined || cards === undefined) {
    return fail(diagnostics);
  }

  const rootId = context.instanceId;
  const columns = Math.max(1, Math.round(numberProp(props, "columns") ?? CARD_DEFAULT_COLUMNS));
  const gap = numberProp(props, "gap") ?? CARD_GAP;
  const x = numberProp(props, "x") ?? 0;
  const y = numberProp(props, "y") ?? 0;

  const children: RenderNodeIr[] = [];
  const ports: Record<string, ConnectorEndpointIr> = {};

  // Per-ROW uniform card heights via the flow-layout engine: every card in a
  // row auto-sizes to that row's tallest wrapped-detail content (see
  // `computeGridCellBoxes`), replacing the older grid-wide single-height math.
  const cardDetailWidth = CARD_W - CARD_INSET * 2;
  const cardDetailTopY = 145.8;
  const layout = computeGridCellBoxes({
    rootId,
    cellIdFor: (index) => `${rootId}__card-${index}`,
    detailLeafIdFor: (index) => `${rootId}__card-${index}__detail`,
    cells: cards.map((card) => ({ detail: card.detail, detailFontSize: 37.8 })),
    columns,
    cellWidth: CARD_W,
    detailWidth: cardDetailWidth,
    detailTopY: cardDetailTopY,
    bottomInset: CARD_INSET,
    minCellHeight: CARD_H,
    gap,
  });

  cards.forEach((card, index) => {
    const cardId = `${rootId}__card-${index}`;
    const cardBox = layout.cellBoxes[index]!;
    const detailBox = layout.detailBoxes[index];
    const cardChildren: RenderNodeIr[] = [
      attachSdkOrigin(
        buildRect({
          id: `${cardId}__accent`,
          geometry: { x: 0, y: 0, width: CARD_ACCENT_W, height: cardBox.height },
          style: { fill: cardAccentColor(card.accent) },
          label: `${card.title} accent`,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.cardGrid", context, "accent"),
      ),
      attachSdkOrigin(
        buildText({
          id: `${cardId}__title`,
          text: card.title,
          geometry: { x: CARD_INSET, y: 54, width: CARD_W - CARD_INSET * 2, height: 64.8 },
          style: { fontSize: 48.6, fontFamily: MONO_FONT, fontWeight: "bold", fill: COLOR_INK, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.cardGrid", context, "title"),
      ),
      attachSdkOrigin(
        buildText({
          id: `${cardId}__detail`,
          text: card.detail,
          geometry: {
            x: CARD_INSET,
            y: cardDetailTopY,
            width: cardDetailWidth,
            height: detailBox?.height ?? grownTextHeight(card.detail, cardDetailWidth, 37.8, 118.8),
          },
          style: { fontSize: 37.8, fill: COLOR_MUTED, textAnchor: "start" },
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.cardGrid", context, "detail"),
      ),
    ];

    children.push(
      attachSdkOrigin(
        buildGroup({
          id: cardId,
          capabilityId: "core.group",
          geometry: { x: cardBox.x, y: cardBox.y, width: CARD_W, height: cardBox.height },
          style: { coordinateSpace: "local", fill: COLOR_SURFACE, stroke: COLOR_BORDER, strokeWidth: 2.7 },
          children: cardChildren,
          label: card.title,
          sourceMap: context.sourceMap,
        }),
        makeOrigin("sdk.cardGrid", context, "card"),
      ),
    );
    ports[`card[${index}]`] = { nodeId: cardId };
  });

  const width = layout.width;
  const height = layout.height;

  const root = attachSdkOrigin(
    buildGroup({
      id: rootId,
      capabilityId: "core.group",
      geometry: { x, y, width, height },
      style: { coordinateSpace: "local" },
      children,
      label: "card grid",
      sourceMap: context.sourceMap,
    }),
    makeOrigin("sdk.cardGrid", context, "root"),
  );

  return succeed({
    roots: [root],
    ports: { self: { nodeId: rootId }, ...ports },
    actions: { enter: [rootId], emphasis: [rootId], exit: [rootId] },
  });
};

export const CARD_GRID_DEFINITION: SdkComponentDefinition = {
  descriptor: makeDescriptor("sdk.cardGrid", "core.group", {
    columns: { type: "number", required: false, default: CARD_DEFAULT_COLUMNS },
    cards: { type: "json", required: true },
    gap: { type: "number", required: false, default: CARD_GAP },
    x: { type: "number", required: false, default: 0 },
    y: { type: "number", required: false, default: 0 },
  }),
  factory: cardGridFactory,
  actions: DECK_ACTIONS,
};

// ---------------------------------------------------------------------------
// Deck composite pack: `sdk/registry.ts` appends this to the generic pack.
// ---------------------------------------------------------------------------

/** Deck-port SDK composite component definitions (working factories). */
export const DECK_COMPOSITE_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
  SECTION_DIVIDER_DEFINITION,
  STEP_CHAIN_DEFINITION,
  BIG_STAT_DEFINITION,
  COMPARE_GRID_DEFINITION,
  NUMBERED_SEQUENCE_DEFINITION,
  TIMELINE_AXIS_DEFINITION,
  NODE_TREE_DEFINITION,
  CARD_GRID_DEFINITION,
];
