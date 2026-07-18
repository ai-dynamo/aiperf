/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";

import { diagnostic, type Result } from "./diagnostic.js";
import type { SourceRange } from "./source.js";

export type LayoutBoundsIr = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

export type LayoutRoutePointIr = Readonly<{
  x: number;
  y: number;
}>;

export type LayoutNodePlanIr = Readonly<{
  nodeId: string;
  bounds: LayoutBoundsIr;
  clip?: boolean | undefined;
  continuation?: boolean | undefined;
}>;

export type LayoutRoutePlanIr = Readonly<{
  edgeId: string;
  points: readonly LayoutRoutePointIr[];
}>;

export type LayoutPlanIr = Readonly<{
  version: 1;
  nodes: readonly LayoutNodePlanIr[];
  routes: readonly LayoutRoutePlanIr[];
}>;

const layoutBoundsSchema = z.strictObject({
  x: z.number().finite(),
  y: z.number().finite(),
  width: z.number().finite().nonnegative(),
  height: z.number().finite().nonnegative(),
});

const layoutNodePlanSchema = z.strictObject({
  nodeId: z.string().min(1),
  bounds: layoutBoundsSchema,
  clip: z.boolean().optional(),
  continuation: z.boolean().optional(),
});

const layoutRoutePointSchema = z.strictObject({
  x: z.number().finite(),
  y: z.number().finite(),
});

const layoutRoutePlanSchema = z.strictObject({
  edgeId: z.string().min(1),
  points: z.array(layoutRoutePointSchema),
});

/** Zod schema for layout plan attachments embedded in Flow IR. */
export const layoutPlanSchema = z.strictObject({
  version: z.literal(1),
  nodes: z.array(layoutNodePlanSchema),
  routes: z.array(layoutRoutePlanSchema),
});

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Parses a strict version-one layout plan. */
export function parseLayoutPlan(input: unknown): LayoutPlanIr {
  return layoutPlanSchema.parse(input);
}

/** Validates a layout plan and maps Zod issues to portable diagnostics. */
export function safeParseLayoutPlan(input: unknown): Result<LayoutPlanIr> {
  const parsed = layoutPlanSchema.safeParse(input);
  if (parsed.success) {
    return { ok: true, value: parsed.data, diagnostics: [] };
  }

  return {
    ok: false,
    diagnostics: parsed.error.issues.map((issue) => {
      const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
      return diagnostic(
        "LAYOUT_PLAN_INVALID",
        "error",
        `${path}: ${issue.message}`,
        unknownRange,
      );
    }),
  };
}

/** Looks up authored bounds for a node id, if present in the plan. */
export function layoutBoundsForNode(
  plan: LayoutPlanIr,
  nodeId: string,
): LayoutBoundsIr | undefined {
  return plan.nodes.find((entry) => entry.nodeId === nodeId)?.bounds;
}
