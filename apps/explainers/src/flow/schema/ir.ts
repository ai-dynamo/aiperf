/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";

import { diagnostic, type Result } from "./diagnostic.js";
import { jsonValueSchema, type JsonValue } from "./json-value.js";
import { layoutPlanSchema, type LayoutPlanIr } from "./layout-plan.js";
import type { SourceRange } from "./source.js";
import {
  semanticModelSchema,
  type SemanticModelIr,
} from "./semantic-model.js";
import {
  flowThemeIrSchema,
  themeRoleReferenceIrSchema,
  type FlowThemeIr,
  type StyleValueIr,
} from "./theme.js";

export type CapabilityRequirement = Readonly<{
  id: string;
  range: string;
}>;

export type GeometryIr = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/** 2D point used by path / polyline / connector geometry. */
export type PointIr = Readonly<{
  x: number;
  y: number;
}>;

/**
 * Connector / line endpoint.
 *
 * Explainer package scenes author either a node reference (`nodeId`) or
 * absolute coordinates (`x`/`y`). SceneRenderer resolves coordinates first.
 *
 * Allowed `anchor` values (when `nodeId` is set): `center`, `n` / `s` / `e` /
 * `w`, `ne` / `nw` / `se` / `sw`, plus aliases `top` / `bottom` / `left` /
 * `right`. Kept as `string` so packages may pass through unknown anchors.
 */
export type ConnectorEndpointIr = Readonly<{
  nodeId?: string | undefined;
  anchor?: string | undefined;
  x?: number | undefined;
  y?: number | undefined;
}>;

/**
 * Relative position for a node's own geometry, resolved at render time
 * against an already-declared sibling's world geometry (document order).
 *
 * `anchor` picks the point on `nodeId` to measure from (same vocabulary as
 * `ConnectorEndpointIr.anchor`; defaults to `center`); `dx`/`dy` offset from
 * that point. Width/height stay as authored on `geometry`. Forward
 * references (a node relative to one declared later in the scene) are not
 * supported — SceneRenderer resolves nodes in document order.
 */
export type RelativePositionIr = Readonly<{
  nodeId: string;
  anchor?: string | undefined;
  dx?: number | undefined;
  dy?: number | undefined;
}>;

export type NodeAccessibilityIr = Readonly<{
  label: string;
  description?: string | undefined;
  decorative?: boolean | undefined;
}>;

/**
 * Compiler-only SDK expansion provenance.
 *
 * Present while the browser compiler assembles and validates scenes; stripped
 * before DeckPackage serialization via `stripSdkOriginsFromScene`.
 */
export type SdkOriginIr = Readonly<{
  componentId: string;
  instanceId: string;
  sourceMap: SourceRange;
  generatedRole: string;
}>;

/**
 * Authored foundation / layout / motion capabilities used by explainer
 * DeckPackage scenes. Open strings remain valid via `capability` /
 * `capabilityId` on nodes; this union documents the known vocabulary.
 */
export type FoundationCapabilityId =
  | "core.text"
  | "core.rect"
  | "core.connector"
  | "core.circle"
  | "core.ellipse"
  | "core.panel"
  | "core.header"
  | "core.arrow"
  | "core.elbow"
  | "core.bracket"
  | "core.callout"
  | "core.chip"
  | "core.note"
  | "core.divider"
  | "core.lane"
  | "core.band"
  | "core.swimlane"
  | "core.stepper"
  | "core.route"
  | "core.fan-out"
  | "core.fan-in"
  | "core.group"
  | "layout.stack"
  | "layout.grid"
  | "layout.pad"
  | "layout.rail"
  | "motion.signal"
  | "motion.pulse";

export type { JsonScalar, JsonValue } from "./json-value.js";

/**
 * Layout style keys for first-class `layout.stack` / `layout.grid` groups
 * (and component-like groups carrying those capabilities):
 *
 * - `layout.stack`: `style.direction` = `"row"` | `"column"`, `style.gap` = number
 * - `layout.grid`: `style.cols` = number, `style.gap` = number
 *
 * Values ride on the existing open `style` record for package compatibility.
 */
export type RenderNodeBaseIr = Readonly<{
  id: string;
  capabilityId?: string | undefined;
  /** Authoring alias for `capabilityId` (e.g. core.text / layout.stack). */
  capability?: FoundationCapabilityId | (string & {}) | undefined;
  geometry: GeometryIr;
  /** Optional render-time x/y override resolved against a sibling node. */
  relativePosition?: RelativePositionIr | undefined;
  style: Readonly<Record<string, StyleValueIr>>;
  accessibility: NodeAccessibilityIr;
  fallback: string;
  sourceMap: SourceRange;
  /** Compiler-only SDK provenance; omitted from serialized DeckPackage scenes. */
  sdkOrigin?: SdkOriginIr | undefined;
  /** SVG path data for connector / arrow / path nodes. */
  path?: string | undefined;
  /** Polyline or control points for path / connector nodes. */
  points?: readonly (PointIr | ConnectorEndpointIr)[] | undefined;
}>;

export type GroupNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "group";
    children: readonly RenderNodeIr[];
  }>;

export type RectNodeIr = RenderNodeBaseIr & Readonly<{ kind: "rect" }>;

export type TextNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "text";
    text: string;
  }>;

/** Orthogonal elbow routing axis for `core.elbow` / connector bends. */
export type ConnectorAxisIr = "x" | "y";

export type ConnectorNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "connector";
    from: ConnectorEndpointIr;
    to: ConnectorEndpointIr;
    /** Optional bend / waypoint for elbow routing. */
    via?: PointIr | undefined;
    /** Preferred first-segment axis for orthogonal elbows. */
    axis?: ConnectorAxisIr | undefined;
  }>;

/** First-class split or merge topology with strict endpoint cardinality. */
export type FanNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "fan";
    capability: "core.fan-out" | "core.fan-in";
    from: ConnectorEndpointIr | readonly ConnectorEndpointIr[];
    to: ConnectorEndpointIr | readonly ConnectorEndpointIr[];
    /** Direction of the shared trunk. */
    axis?: ConnectorAxisIr | undefined;
    /** Optional authored junction point; automatic placement is the default. */
    junction?: PointIr | undefined;
  }>;

export type ComponentNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "component";
    capabilityId: string;
    props: Readonly<Record<string, JsonValue>>;
    semanticModel?: SemanticModelIr | undefined;
    layoutPlan?: LayoutPlanIr | undefined;
    children: readonly RenderNodeIr[];
  }>;

export type RenderNodeIr =
  | GroupNodeIr
  | RectNodeIr
  | TextNodeIr
  | ConnectorNodeIr
  | FanNodeIr
  | ComponentNodeIr;

export type CameraKeyframeIr = Readonly<{
  id: string;
  at: number;
  x: number;
  y: number;
  zoom: number;
  /** Present for cinematic lowers; optional on package/DeckPackage scenes. */
  sourceMap?: SourceRange | undefined;
}>;

/** Logical SVG / canvas bounds for explainer SceneRenderer viewports. */
export type SceneViewportIr = Readonly<{
  width: number;
  height: number;
}>;

/**
 * Well-known timeline cue actions used by explainer diagram playback.
 * Open strings remain accepted for forward compatibility.
 */
export type TimelineCueAction =
  | "enter"
  | "draw"
  | "fade"
  | "exit"
  | "emphasis"
  | "emphasize"
  | "pulse"
  | "reveal"
  | "trace"
  | "stagger"
  | "enter-children"
  | (string & {});

/** Per-cue easing pass-through for SceneRenderer progress mapping. */
export type TimelineCueEasing =
  | "linear"
  | "ease-in"
  | "ease-out"
  | "ease-in-out";

/**
 * Timed diagram cue. Compact stagger uses `action: "stagger"` (or sugar
 * `"enter-children"`) with `targets` + `step`; `target` may be `""` when
 * only `targets[]` identifies the staggered nodes.
 */
export type TimelineCueIr = Readonly<{
  id: string;
  at: number;
  duration: number;
  /** Primary / group id; may be `""` when `targets` is used. */
  target: string;
  action: TimelineCueAction;
  /** Stagger member node ids when `action` is `stagger` / `enter-children`. */
  targets?: readonly string[] | undefined;
  /** Delay between successive stagger targets (ms or scene time units). */
  step?: number | undefined;
  easing?: TimelineCueEasing | undefined;
  sourceMap: SourceRange;
}>;

/** One synchronized spoken-narration and subtitle interval. */
export type NarrativeCueIr = Readonly<{
  id: string;
  startMs: number;
  endMs: number;
  spokenText: string;
  subtitleText: string;
  audioAsset?: string | undefined;
}>;

/** Timed narrative content and its locale-specific voice metadata. */
export type NarrativeTrackIr = Readonly<{
  language: string;
  voice?: string | undefined;
  cues: readonly NarrativeCueIr[];
}>;

export type InteractionIr = Readonly<{
  id: string;
  event: string;
  target: string;
  action: string;
  sourceMap: SourceRange;
}>;

export type ResponsiveVariantIr = Readonly<{
  id: string;
  condition: string;
  roots: readonly RenderNodeIr[];
  sourceMap: SourceRange;
}>;

export type SceneAccessibilityIr = Readonly<{
  label: string;
  readingOrder: readonly string[];
}>;

export type SceneIr = Readonly<{
  id: string;
  title: string;
  /** Human-readable scene summary (SVG `<desc>` / a11y). */
  summary: string;
  /** Optional diagram viewport (defaults to ~700×400 in ExplainerShell). */
  viewport?: SceneViewportIr | undefined;
  roots: readonly RenderNodeIr[];
  camera: readonly CameraKeyframeIr[];
  timeline: readonly TimelineCueIr[];
  narration: string;
  narrativeTrack?: NarrativeTrackIr | undefined;
  interactions: readonly InteractionIr[];
  responsive: readonly ResponsiveVariantIr[];
  accessibility: SceneAccessibilityIr;
  fallback: string;
  sourceMap: SourceRange;
}>;

export type FlowIr = Readonly<{
  irVersion: 2;
  id: string;
  title: string;
  capabilities: readonly CapabilityRequirement[];
  tokens: Readonly<Record<string, string | number | boolean>>;
  themes: readonly FlowThemeIr[];
  defaultTheme?: string | undefined;
  scenes: readonly SceneIr[];
  sourceMap: SourceRange;
}>;

const sourcePositionSchema = z.strictObject({
  offset: z.number().int().nonnegative(),
  line: z.number().int().positive(),
  column: z.number().int().positive(),
});

const sourceRangeSchema = z.strictObject({
  source: z.string().min(1),
  start: sourcePositionSchema,
  end: sourcePositionSchema,
});

const scalarSchema = z.union([z.string(), z.number().finite(), z.boolean()]);
const styleValueSchema = z.union([scalarSchema, themeRoleReferenceIrSchema]);
const styleSchema = z.record(z.string(), styleValueSchema);
const geometrySchema = z.strictObject({
  x: z.number().finite(),
  y: z.number().finite(),
  width: z.number().finite().nonnegative(),
  height: z.number().finite().nonnegative(),
});
const pointIrSchema = z.strictObject({
  x: z.number().finite(),
  y: z.number().finite(),
});
const connectorEndpointSchema = z
  .strictObject({
    nodeId: z.string().min(1).optional(),
    anchor: z.string().min(1).optional(),
    x: z.number().finite().optional(),
    y: z.number().finite().optional(),
  })
  .superRefine((endpoint, context) => {
    const hasNode =
      typeof endpoint.nodeId === "string" && endpoint.nodeId.length > 0;
    const hasPoint =
      typeof endpoint.x === "number" && typeof endpoint.y === "number";
    if (!hasNode && !hasPoint) {
      context.addIssue({
        code: "custom",
        message: "connector endpoint requires nodeId or x/y coordinates",
      });
    }
  });
const polylinePointSchema = z.union([pointIrSchema, connectorEndpointSchema]);
const relativePositionSchema = z.strictObject({
  nodeId: z.string().min(1),
  anchor: z.string().min(1).optional(),
  dx: z.number().finite().optional(),
  dy: z.number().finite().optional(),
});
const nodeAccessibilitySchema = z.strictObject({
  label: z.string().min(1),
  description: z.string().min(1).optional(),
  decorative: z.boolean().optional(),
});
const sdkOriginSchema = z.strictObject({
  componentId: z.string().min(1),
  instanceId: z.string().min(1),
  sourceMap: sourceRangeSchema,
  generatedRole: z.string().min(1),
});
const foundationCapabilitySchema = z.union([
  z.literal("core.text"),
  z.literal("core.rect"),
  z.literal("core.connector"),
  z.literal("core.circle"),
  z.literal("core.ellipse"),
  z.literal("core.panel"),
  z.literal("core.header"),
  z.literal("core.arrow"),
  z.literal("core.elbow"),
  z.literal("core.bracket"),
  z.literal("core.callout"),
  z.literal("core.chip"),
  z.literal("core.note"),
  z.literal("core.divider"),
  z.literal("core.lane"),
  z.literal("core.band"),
  z.literal("core.swimlane"),
  z.literal("core.stepper"),
  z.literal("core.route"),
  z.literal("core.fan-out"),
  z.literal("core.fan-in"),
  z.literal("core.group"),
  z.literal("layout.stack"),
  z.literal("layout.grid"),
  z.literal("layout.pad"),
  z.literal("layout.rail"),
  z.literal("motion.signal"),
  z.literal("motion.pulse"),
  z.string().min(1),
]);

const renderNodeBaseShape = {
  id: z.string().min(1),
  capabilityId: z.string().min(1).optional(),
  capability: foundationCapabilitySchema.optional(),
  geometry: geometrySchema,
  relativePosition: relativePositionSchema.optional(),
  style: styleSchema,
  accessibility: nodeAccessibilitySchema,
  fallback: z.string().min(1),
  sourceMap: sourceRangeSchema,
  sdkOrigin: sdkOriginSchema.optional(),
  path: z.string().optional(),
  points: z.array(polylinePointSchema).optional(),
};
const connectorAxisSchema = z.union([z.literal("x"), z.literal("y")]);
const fanEndpointSideSchema = z.union([
  connectorEndpointSchema,
  z.array(connectorEndpointSchema),
]);

const fanNodeObjectSchema = z.strictObject({
  ...renderNodeBaseShape,
  kind: z.literal("fan"),
  capability: z.union([
    z.literal("core.fan-out"),
    z.literal("core.fan-in"),
  ]),
  from: fanEndpointSideSchema,
  to: fanEndpointSideSchema,
  axis: connectorAxisSchema.optional(),
  junction: pointIrSchema.optional(),
});

function fanCardinalityIssues(node: FanNodeIr): readonly Readonly<{
  path: "from" | "to";
  message: string;
}>[] {
  const issues: Array<{ path: "from" | "to"; message: string }> = [];
  if (node.capability === "core.fan-out") {
    if (Array.isArray(node.from)) {
      issues.push({
        path: "from",
        message: `Fan "${node.id}" fan-out requires exactly one source endpoint`,
      });
    }
    if (!Array.isArray(node.to) || node.to.length < 2) {
      issues.push({
        path: "to",
        message: `Fan "${node.id}" fan-out requires at least two destination endpoints`,
      });
    }
    return issues;
  }
  if (!Array.isArray(node.from) || node.from.length < 2) {
    issues.push({
      path: "from",
      message: `Fan "${node.id}" fan-in requires at least two source endpoints`,
    });
  }
  if (Array.isArray(node.to)) {
    issues.push({
      path: "to",
      message: `Fan "${node.id}" fan-in requires exactly one destination endpoint`,
    });
  }
  return issues;
}

/** Strict schema for a first-class fan split or merge node. */
export const fanNodeSchema: z.ZodType<FanNodeIr> =
  fanNodeObjectSchema.superRefine((node, context) => {
    for (const issue of fanCardinalityIssues(node)) {
      context.addIssue({
        code: "custom",
        path: [issue.path],
        message: issue.message,
      });
    }
  });

/** Timeline cue actions accepted by SceneIr / DeckPackage scenes. */
export const timelineCueActionSchema = z.union([
  z.literal("enter"),
  z.literal("draw"),
  z.literal("fade"),
  z.literal("exit"),
  z.literal("emphasis"),
  z.literal("emphasize"),
  z.literal("pulse"),
  z.literal("reveal"),
  z.literal("trace"),
  z.literal("stagger"),
  z.literal("enter-children"),
  z.string().min(1),
]);

/** Per-cue easing values accepted by SceneIr / DeckPackage scenes. */
export const timelineCueEasingSchema = z.union([
  z.literal("linear"),
  z.literal("ease-in"),
  z.literal("ease-out"),
  z.literal("ease-in-out"),
]);

const renderNodeSchema: z.ZodType<RenderNodeIr> = z.lazy(() =>
  z
    .discriminatedUnion("kind", [
      z.strictObject({
        ...renderNodeBaseShape,
        kind: z.literal("group"),
        children: z.array(renderNodeSchema),
      }),
      z.strictObject({
        ...renderNodeBaseShape,
        kind: z.literal("rect"),
      }),
      z.strictObject({
        ...renderNodeBaseShape,
        kind: z.literal("text"),
        text: z.string(),
      }),
      z.strictObject({
        ...renderNodeBaseShape,
        kind: z.literal("connector"),
        from: connectorEndpointSchema,
        to: connectorEndpointSchema,
        via: pointIrSchema.optional(),
        axis: connectorAxisSchema.optional(),
      }),
      fanNodeObjectSchema,
      z.strictObject({
        ...renderNodeBaseShape,
        kind: z.literal("component"),
        capabilityId: z.string().min(1),
        props: z.record(z.string(), jsonValueSchema),
        semanticModel: semanticModelSchema.optional(),
        layoutPlan: layoutPlanSchema.optional(),
        children: z.array(renderNodeSchema),
      }),
    ])
    .superRefine((node, context) => {
      if (node.kind !== "fan") {
        return;
      }
      for (const issue of fanCardinalityIssues(node)) {
        context.addIssue({
          code: "custom",
          path: [issue.path],
          message: issue.message,
        });
      }
    }),
);

/** Strict schema for any render node in a scene. */
export const sceneNodeSchema: z.ZodType<RenderNodeIr> = renderNodeSchema;

const cameraKeyframeSchema = z.strictObject({
  id: z.string().min(1),
  at: z.number().finite().nonnegative(),
  x: z.number().finite(),
  y: z.number().finite(),
  zoom: z.number().finite().positive(),
  sourceMap: sourceRangeSchema.optional(),
});
const sceneViewportSchema = z.strictObject({
  width: z.number().finite().positive(),
  height: z.number().finite().positive(),
});
const timelineCueSchema = z.strictObject({
  id: z.string().min(1),
  at: z.number().finite().nonnegative(),
  duration: z.number().finite().nonnegative(),
  /** May be empty when stagger `targets` identifies the nodes. */
  target: z.string(),
  action: timelineCueActionSchema,
  targets: z.array(z.string().min(1)).optional(),
  step: z.number().finite().nonnegative().optional(),
  easing: timelineCueEasingSchema.optional(),
  sourceMap: sourceRangeSchema,
});
const timeMsSchema = z
  .number()
  .int()
  .nonnegative()
  .max(Number.MAX_SAFE_INTEGER);

/** Strict schema for one timed narrative cue. */
export const narrativeCueSchema: z.ZodType<NarrativeCueIr> = z
  .strictObject({
    id: z.string().trim().min(1),
    startMs: timeMsSchema,
    endMs: timeMsSchema,
    spokenText: z.string().trim().min(1),
    subtitleText: z.string().trim().min(1),
    audioAsset: z.string().trim().min(1).optional(),
  })
  .superRefine((cue, context) => {
    if (cue.endMs <= cue.startMs) {
      context.addIssue({
        code: "custom",
        path: ["endMs"],
        message: "endMs must be greater than startMs",
      });
    }
  });

/** Strict schema for a non-overlapping timed narrative track. */
export const narrativeTrackSchema: z.ZodType<NarrativeTrackIr> = z
  .strictObject({
    language: z.string().trim().min(1),
    voice: z.string().trim().min(1).optional(),
    cues: z.array(narrativeCueSchema).min(1),
  })
  .superRefine((track, context) => {
    const cueIds = new Set<string>();
    for (const [index, cue] of track.cues.entries()) {
      if (cueIds.has(cue.id)) {
        context.addIssue({
          code: "custom",
          path: ["cues", index, "id"],
          message: "narrative cue ids must be unique",
        });
      }
      cueIds.add(cue.id);
    }

    const chronologicalCues = track.cues
      .map((cue, index) => ({ cue, index }))
      .sort(
        (left, right) =>
          left.cue.startMs - right.cue.startMs ||
          left.cue.endMs - right.cue.endMs ||
          left.cue.id.localeCompare(right.cue.id),
      );
    for (let index = 1; index < chronologicalCues.length; index += 1) {
      const previous = chronologicalCues[index - 1];
      const current = chronologicalCues[index];
      if (
        previous !== undefined &&
        current !== undefined &&
        current.cue.startMs < previous.cue.endMs
      ) {
        context.addIssue({
          code: "custom",
          path: ["cues", current.index, "startMs"],
          message: `narrative cue overlaps "${previous.cue.id}"`,
        });
      }
    }
  });
const interactionSchema = z.strictObject({
  id: z.string().min(1),
  event: z.string().min(1),
  target: z.string().min(1),
  action: z.string().min(1),
  sourceMap: sourceRangeSchema,
});
const responsiveVariantSchema = z.strictObject({
  id: z.string().min(1),
  condition: z.string().min(1),
  roots: z.array(renderNodeSchema),
  sourceMap: sourceRangeSchema,
});
/** Strict Zod schema for a single scene IR document. */
export const sceneIrSchema: z.ZodType<SceneIr> = z.strictObject({
  id: z.string().min(1),
  title: z.string().min(1),
  summary: z.string().min(1),
  viewport: sceneViewportSchema.optional(),
  roots: z.array(renderNodeSchema),
  camera: z.array(cameraKeyframeSchema),
  timeline: z.array(timelineCueSchema),
  narration: z.string(),
  narrativeTrack: narrativeTrackSchema.optional(),
  interactions: z.array(interactionSchema),
  responsive: z.array(responsiveVariantSchema),
  accessibility: z.strictObject({
    label: z.string().min(1),
    readingOrder: z.array(z.string().min(1)),
  }),
  fallback: z.string().min(1),
  sourceMap: sourceRangeSchema,
});

const flowIrSchema = z.strictObject({
  irVersion: z.literal(2),
  id: z.string().min(1),
  title: z.string().min(1),
  capabilities: z.array(
    z.strictObject({
      id: z.string().min(1),
      range: z.string().min(1),
    }),
  ),
  tokens: z.record(z.string(), scalarSchema).default({}),
  themes: z.array(flowThemeIrSchema),
  defaultTheme: z.string().min(1).optional(),
  scenes: z.array(sceneIrSchema),
  sourceMap: sourceRangeSchema,
});

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

function inputRange(input: unknown): SourceRange {
  if (typeof input !== "object" || input === null || !("sourceMap" in input)) {
    return unknownRange;
  }

  const parsed = sourceRangeSchema.safeParse(input.sourceMap);
  return parsed.success ? parsed.data : unknownRange;
}

/** Parses and validates strict version-two Flow IR. */
export function parseFlowIr(input: unknown): FlowIr {
  return flowIrSchema.parse(input);
}

/** Upgrades a version-one Flow IR object to the strict version-two shape. */
export function upgradeFlowIrV1ToV2(input: unknown): unknown {
  if (
    typeof input !== "object" ||
    input === null ||
    Array.isArray(input) ||
    !("irVersion" in input) ||
    input.irVersion !== 1
  ) {
    return input;
  }

  return {
    ...input,
    irVersion: 2,
    themes: [],
  };
}

/** Parses and validates a single scene against the canonical schema. */
export function parseSceneIr(input: unknown): SceneIr {
  return sceneIrSchema.parse(input);
}

/** Parses and validates a standalone timed narrative track. */
export function parseNarrativeTrackIr(input: unknown): NarrativeTrackIr {
  return narrativeTrackSchema.parse(input);
}

/** Validates Flow IR and maps all Zod issues to portable diagnostics. */
export function safeParseFlowIr(input: unknown): Result<FlowIr> {
  const parsed = flowIrSchema.safeParse(input);
  if (parsed.success) {
    return { ok: true, value: parsed.data, diagnostics: [] };
  }

  const range = inputRange(input);
  return {
    ok: false,
    diagnostics: parsed.error.issues.map((issue) => {
      const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
      return diagnostic(
        "IR_INVALID",
        "error",
        `${path}: ${issue.message}`,
        range,
      );
    }),
  };
}
