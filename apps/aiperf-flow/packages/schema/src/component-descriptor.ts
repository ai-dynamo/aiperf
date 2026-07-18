/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";

import { diagnostic, type Result } from "./diagnostic.js";
import { jsonValueSchema, type JsonValue } from "./json-value.js";
import type { SourceRange } from "./source.js";

export const COMPONENT_CLASSIFICATIONS = [
  "flow-only",
  "hybrid",
  "leaf",
] as const;

export type ComponentClassification =
  (typeof COMPONENT_CLASSIFICATIONS)[number];

export type ComponentPropDescriptor = Readonly<{
  type: string;
  required: boolean;
  default?: JsonValue | undefined;
}>;

export type ComponentSlotDescriptor = Readonly<{
  accepts: string;
  required: boolean;
}>;

export type ComponentDescriptor = Readonly<{
  id: string;
  symbolExport: string;
  version: string;
  classification: ComponentClassification;
  props: Readonly<Record<string, ComponentPropDescriptor>>;
  slots: Readonly<Record<string, ComponentSlotDescriptor>>;
  events: readonly string[];
  capabilityId: string;
  leafId?: string | undefined;
  deterministic: boolean;
}>;

export type ComponentCatalog = Readonly<{
  components: readonly ComponentDescriptor[];
}>;

const componentPropSchema = z.strictObject({
  type: z.string().min(1),
  required: z.boolean(),
  default: jsonValueSchema.optional(),
});

const componentSlotSchema = z.strictObject({
  accepts: z.string().min(1),
  required: z.boolean(),
});

/** Zod schema for stdlib component descriptor contracts. */
export const componentDescriptorSchema = z.strictObject({
  id: z.string().min(1),
  symbolExport: z.string().min(1),
  version: z.string().min(1),
  classification: z.enum(COMPONENT_CLASSIFICATIONS),
  props: z.record(z.string(), componentPropSchema),
  slots: z.record(z.string(), componentSlotSchema),
  events: z.array(z.string().min(1)),
  capabilityId: z.string().min(1),
  leafId: z.string().min(1).optional(),
  deterministic: z.boolean(),
});

const catalogRange: SourceRange = {
  source: "<component-catalog>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Parses and validates a strict component descriptor. */
export function parseComponentDescriptor(input: unknown): ComponentDescriptor {
  return componentDescriptorSchema.parse(input);
}

/** Validates a component descriptor and maps Zod issues to portable diagnostics. */
export function safeParseComponentDescriptor(
  input: unknown,
): Result<ComponentDescriptor> {
  const parsed = componentDescriptorSchema.safeParse(input);
  if (parsed.success) {
    return { ok: true, value: parsed.data, diagnostics: [] };
  }

  return {
    ok: false,
    diagnostics: parsed.error.issues.map((issue) => {
      const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
      return diagnostic(
        "COMPONENT_INVALID",
        "error",
        `${path}: ${issue.message}`,
        unknownRange,
      );
    }),
  };
}

/** Creates a deterministic catalog or diagnoses duplicate component IDs. */
export function createComponentCatalog(
  descriptors: readonly ComponentDescriptor[],
): Result<ComponentCatalog> {
  const components = [...descriptors].sort(({ id: left }, { id: right }) =>
    left.localeCompare(right),
  );
  const duplicate = components.find(
    ({ id }, index) => index > 0 && components[index - 1]?.id === id,
  );

  if (duplicate !== undefined) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "COMPONENT_DUPLICATE",
          "error",
          `Duplicate component ID "${duplicate.id}".`,
          catalogRange,
          "Use a unique component ID.",
        ),
      ],
    };
  }

  return { ok: true, value: { components }, diagnostics: [] };
}
