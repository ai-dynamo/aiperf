/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";

import { diagnostic, type Result } from "./diagnostic.js";
import type { SourceRange } from "./source.js";

export type SemanticEntityIr = Readonly<{
  id: string;
  label: string;
  kind?: string | undefined;
}>;

export type SemanticRelationIr = Readonly<{
  id: string;
  from: string;
  to: string;
  kind?: string | undefined;
}>;

export type MorphCorrespondenceKind =
  | "one-to-one"
  | "one-to-many"
  | "many-to-one"
  | "split"
  | "merge"
  | "reorder"
  | "replace"
  | "disappear"
  | "special-insert";

export type MorphCorrespondenceIr = Readonly<{
  id: string;
  sourceIds: readonly string[];
  targetIds: readonly string[];
  kind: MorphCorrespondenceKind;
}>;

export type SemanticModelIr = Readonly<{
  entities: readonly SemanticEntityIr[];
  relations: readonly SemanticRelationIr[];
  morphs: readonly MorphCorrespondenceIr[];
}>;

const morphKindSchema = z.enum([
  "one-to-one",
  "one-to-many",
  "many-to-one",
  "split",
  "merge",
  "reorder",
  "replace",
  "disappear",
  "special-insert",
]);

const semanticEntitySchema = z.strictObject({
  id: z.string().min(1),
  label: z.string().min(1),
  kind: z.string().min(1).optional(),
});

const semanticRelationSchema = z.strictObject({
  id: z.string().min(1),
  from: z.string().min(1),
  to: z.string().min(1),
  kind: z.string().min(1).optional(),
});

const morphCorrespondenceSchema = z.strictObject({
  id: z.string().min(1),
  sourceIds: z.array(z.string().min(1)),
  targetIds: z.array(z.string().min(1)),
  kind: morphKindSchema,
});

/** Zod schema for semantic model attachments embedded in Flow IR. */
export const semanticModelSchema = z.strictObject({
  entities: z.array(semanticEntitySchema),
  relations: z.array(semanticRelationSchema),
  morphs: z.array(morphCorrespondenceSchema),
});

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Parses a strict semantic model attachment. */
export function parseSemanticModel(input: unknown): SemanticModelIr {
  return semanticModelSchema.parse(input);
}

/** Validates a semantic model and maps Zod issues to portable diagnostics. */
export function safeParseSemanticModel(input: unknown): Result<SemanticModelIr> {
  const parsed = semanticModelSchema.safeParse(input);
  if (parsed.success) {
    return { ok: true, value: parsed.data, diagnostics: [] };
  }

  return {
    ok: false,
    diagnostics: parsed.error.issues.map((issue) => {
      const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
      return diagnostic(
        "SEMANTIC_INVALID",
        "error",
        `${path}: ${issue.message}`,
        unknownRange,
      );
    }),
  };
}

/** Returns entity ids in declaration order for stable outline generation. */
export function semanticEntityIds(model: SemanticModelIr): readonly string[] {
  return model.entities.map(({ id }) => id);
}
