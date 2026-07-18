/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";

import { diagnostic, type Result } from "./diagnostic.js";
import { sceneIrSchema, type SceneIr } from "./ir.js";
import type { SourceRange } from "./source.js";

export type DeckHub = Readonly<{
  title: string;
  highlight: string;
  description: string;
}>;

export type DeckGlossaryEntry = Readonly<{
  word: string;
  meaning: string;
}>;

export type DeckTerm = Readonly<{
  word: string;
  meaning: string;
}>;

export type SceneRender = Readonly<{
  kind: "scene";
  scene: SceneIr;
}>;

export type SlidePackage = Readonly<{
  id: string;
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: DeckTerm | undefined;
  points: readonly string[];
  caption: string;
  render?: SceneRender | undefined;
}>;

export type DeckPackage = Readonly<{
  schemaVersion: 1;
  id: string;
  route: string;
  topic: string;
  storagePrefix: string;
  classPrefix: string;
  eyebrowLabel: string;
  startGateTitle: string;
  hub: DeckHub;
  css?: string | undefined;
  finalCard?: SceneRender | undefined;
  slides: readonly SlidePackage[];
  glossary: readonly DeckGlossaryEntry[];
}>;

const deckTermSchema = z.strictObject({
  word: z.string().min(1),
  meaning: z.string().min(1),
});

const sceneRenderSchema = z.strictObject({
  kind: z.literal("scene"),
  scene: sceneIrSchema,
});

/** Strict Zod schema for one explainer slide package. */
export const slidePackageSchema: z.ZodType<SlidePackage> = z.strictObject({
  id: z.string().min(1),
  eyebrow: z.string(),
  title: z.string().min(1),
  lede: z.string(),
  narration: z.string(),
  term: deckTermSchema.optional(),
  points: z.array(z.string()),
  caption: z.string(),
  render: sceneRenderSchema.optional(),
});

/** Strict Zod schema for a flow-backed explainer DeckPackage. */
export const deckPackageSchema: z.ZodType<DeckPackage> = z.strictObject({
  schemaVersion: z.literal(1),
  id: z.string().min(1),
  route: z.string().min(1),
  topic: z.string().min(1),
  storagePrefix: z.string().min(1),
  classPrefix: z.string().min(1),
  eyebrowLabel: z.string().min(1),
  startGateTitle: z.string().min(1),
  hub: z.strictObject({
    title: z.string().min(1),
    highlight: z.string().min(1),
    description: z.string().min(1),
  }),
  css: z.string().optional(),
  finalCard: sceneRenderSchema.optional(),
  slides: z.array(slidePackageSchema),
  glossary: z.array(deckTermSchema),
});

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Parses and validates a strict DeckPackage. */
export function parseDeckPackage(input: unknown): DeckPackage {
  return deckPackageSchema.parse(input);
}

/** Validates a DeckPackage and maps Zod issues to portable diagnostics. */
export function safeParseDeckPackage(input: unknown): Result<DeckPackage> {
  const parsed = deckPackageSchema.safeParse(input);
  if (parsed.success) {
    return { ok: true, value: parsed.data, diagnostics: [] };
  }

  return {
    ok: false,
    diagnostics: parsed.error.issues.map((issue) => {
      const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
      return diagnostic(
        "DECK_PACKAGE_INVALID",
        "error",
        `${path}: ${issue.message}`,
        unknownRange,
      );
    }),
  };
}
