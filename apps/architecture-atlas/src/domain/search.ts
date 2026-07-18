// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from "zod";

import {
  architectureIdSchema,
  architectureStatusSchema,
  executionModeSchema,
  ownershipSchema,
  type ArchitectureStatus,
  type ExecutionMode,
  type Ownership,
} from "./architecture";
import { audienceSchema } from "./audience";

const csvSchema = <T extends string>(schema: z.ZodType<T>) =>
  z
    .string()
    .optional()
    .refine(
      (value) =>
        value === undefined ||
        value
          .split(",")
          .filter(Boolean)
          .every((item) => schema.safeParse(item).success),
    );

const presentSchema = z
  .union([z.boolean(), z.literal("true"), z.literal("false")])
  .transform((value) => value === true || value === "true")
  .optional();
const modesSchema = csvSchema(executionModeSchema);
const statusesSchema = csvSchema(architectureStatusSchema);
const ownershipSelectionSchema = csvSchema(ownershipSchema);
const layoutSchema = z.enum(["ownership", "lifecycle"]).optional();
const querySchema = z
  .string()
  .max(160)
  .refine((value) => value.trim().length > 0)
  .optional();
const selectedSchema = architectureIdSchema.optional();

export const atlasSearchSchema = z.object({
  audience: audienceSchema.optional(),
  layout: layoutSchema,
  modes: modesSchema,
  ownership: ownershipSelectionSchema,
  statuses: statusesSchema,
  present: presentSchema,
  query: querySchema,
  selected: selectedSchema,
});

export type AtlasSearch = z.infer<typeof atlasSearchSchema>;

export function parseAtlasSearch(search: Record<string, unknown>): AtlasSearch {
  const parsed: AtlasSearch = {};
  const fields = {
    audience: audienceSchema.optional().safeParse(search.audience),
    layout: layoutSchema.safeParse(search.layout),
    modes: modesSchema.safeParse(search.modes),
    ownership: ownershipSelectionSchema.safeParse(search.ownership),
    statuses: statusesSchema.safeParse(search.statuses),
    present: presentSchema.safeParse(search.present),
    query: querySchema.safeParse(search.query),
    selected: selectedSchema.safeParse(search.selected),
  };
  for (const [key, result] of Object.entries(fields)) {
    if (result.success && result.data !== undefined) {
      Object.assign(parsed, { [key]: result.data });
    }
  }
  return parsed;
}

function parseSelection<T extends string>(
  value: string | undefined,
  schema: z.ZodType<T>,
): T[] {
  if (!value) {
    return [];
  }
  return value
    .split(",")
    .map((item) => schema.safeParse(item))
    .flatMap((result) => (result.success ? [result.data] : []));
}

export function parseModes(value: string | undefined): ExecutionMode[] {
  return parseSelection(value, executionModeSchema);
}

export function parseStatuses(
  value: string | undefined,
): ArchitectureStatus[] {
  return parseSelection(value, architectureStatusSchema);
}

export function parseOwnership(value: string | undefined): Ownership[] {
  return parseSelection(value, ownershipSchema);
}

export function encodeSelection<T extends string>(
  values: readonly T[],
): string | undefined {
  return values.length > 0 ? values.join(",") : undefined;
}
