// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from "zod";

import {
  architectureStatusSchema,
  executionModeSchema,
  type ArchitectureStatus,
  type ExecutionMode,
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

export const atlasSearchSchema = z.object({
  audience: audienceSchema.optional(),
  modes: csvSchema(executionModeSchema),
  statuses: csvSchema(architectureStatusSchema),
  present: z
    .union([z.boolean(), z.literal("true"), z.literal("false")])
    .transform((value) => value === true || value === "true")
    .optional(),
});

export type AtlasSearch = z.infer<typeof atlasSearchSchema>;

export function parseAtlasSearch(search: Record<string, unknown>): AtlasSearch {
  const result = atlasSearchSchema.safeParse(search);
  return result.success ? result.data : {};
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

export function encodeSelection<T extends string>(
  values: readonly T[],
): string | undefined {
  return values.length > 0 ? values.join(",") : undefined;
}
