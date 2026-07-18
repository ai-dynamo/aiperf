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

export type NodeAccessibilityIr = Readonly<{
  label: string;
  description?: string | undefined;
  decorative?: boolean | undefined;
}>;

export type { JsonScalar, JsonValue } from "./json-value.js";

export type RenderNodeBaseIr = Readonly<{
  id: string;
  capabilityId?: string | undefined;
  geometry: GeometryIr;
  style: Readonly<Record<string, StyleValueIr>>;
  accessibility: NodeAccessibilityIr;
  fallback: string;
  sourceMap: SourceRange;
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

export type ConnectorEndpointIr = Readonly<{
  nodeId: string;
  anchor?: string | undefined;
}>;

export type ConnectorNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "connector";
    from: ConnectorEndpointIr;
    to: ConnectorEndpointIr;
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
  | ComponentNodeIr;

export type CameraKeyframeIr = Readonly<{
  id: string;
  at: number;
  x: number;
  y: number;
  zoom: number;
  sourceMap: SourceRange;
}>;

export type TimelineCueIr = Readonly<{
  id: string;
  at: number;
  duration: number;
  target: string;
  action: string;
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
  summary: string;
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
  defaultTheme?: string;
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
const nodeAccessibilitySchema = z.strictObject({
  label: z.string().min(1),
  description: z.string().min(1).optional(),
  decorative: z.boolean().optional(),
});

const renderNodeBaseShape = {
  id: z.string().min(1),
  capabilityId: z.string().min(1).optional(),
  geometry: geometrySchema,
  style: styleSchema,
  accessibility: nodeAccessibilitySchema,
  fallback: z.string().min(1),
  sourceMap: sourceRangeSchema,
};
const connectorEndpointSchema = z.strictObject({
  nodeId: z.string().min(1),
  anchor: z.string().min(1).optional(),
});

const renderNodeSchema: z.ZodType<RenderNodeIr> = z.lazy(() =>
  z.discriminatedUnion("kind", [
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
    }),
    z.strictObject({
      ...renderNodeBaseShape,
      kind: z.literal("component"),
      capabilityId: z.string().min(1),
      props: z.record(z.string(), jsonValueSchema),
      semanticModel: semanticModelSchema.optional(),
      layoutPlan: layoutPlanSchema.optional(),
      children: z.array(renderNodeSchema),
    }),
  ]),
);

const cameraKeyframeSchema = z.strictObject({
  id: z.string().min(1),
  at: z.number().finite().nonnegative(),
  x: z.number().finite(),
  y: z.number().finite(),
  zoom: z.number().finite().positive(),
  sourceMap: sourceRangeSchema,
});
const timelineCueSchema = z.strictObject({
  id: z.string().min(1),
  at: z.number().finite().nonnegative(),
  duration: z.number().finite().nonnegative(),
  target: z.string().min(1),
  action: z.string().min(1),
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
const sceneSchema = z.strictObject({
  id: z.string().min(1),
  title: z.string().min(1),
  summary: z.string().min(1),
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
  scenes: z.array(sceneSchema),
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
  return sceneSchema.parse(input);
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
