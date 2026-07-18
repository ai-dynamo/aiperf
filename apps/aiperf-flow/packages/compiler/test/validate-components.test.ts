/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  ComponentInvocationAst,
  DocumentAst,
} from "@aiperf/flow-language";
import {
  FOUNDATION_CAPABILITIES,
  type ComponentCatalog,
  type ComponentDescriptor,
  type SourceRange,
} from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import type { LinkedDocument } from "../src/link.js";
import { validate } from "../src/validate.js";

function range(): SourceRange {
  return {
    source: "<test>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function invocation(
  name: string,
  props: Readonly<Record<string, string | number | boolean>>,
): ComponentInvocationAst {
  return {
    kind: "component-invocation",
    name,
    props: Object.entries(props).map(([propName, value]) => ({
      kind: "prop-assignment",
      name: propName,
      value: { kind: "literal", value, sourceMap: range() },
      sourceMap: range(),
    })),
    sourceMap: range(),
  };
}

function linkedWith(
  component: ComponentInvocationAst,
  readingOrder = true,
): LinkedDocument {
  const document: DocumentAst = {
    kind: "document",
    id: "demo",
    title: "Demo",
    language: { kind: "language", version: 1, sourceMap: range() },
    requirements: [],
    tokens: [],
    symbols: [],
    scenes: [
      {
        kind: "scene",
        id: "solo",
        title: "Solo",
        summary: {
          kind: "summary",
          text: "Component validation.",
          sourceMap: range(),
        },
        renderDeclarations: [component],
        cameras: [],
        timelines: [],
        interactions: [],
        responsiveVariants: [],
        narration: {
          kind: "narration",
          text: "A sufficiently detailed narration for component validation.",
          sourceMap: range(),
        },
        readingOrder: readingOrder
          ? {
              kind: "reading-order",
              references: [],
              sourceMap: range(),
            }
          : undefined,
        fallback: { kind: "fallback", text: "Component.", sourceMap: range() },
        sourceMap: range(),
      },
    ],
    sourceMap: range(),
  };

  return {
    document,
    tokens: new Map(),
    scenes: new Map([["solo", { nodes: new Map() }]]),
  };
}

const spanMap: ComponentDescriptor = {
  id: "core.span-map",
  symbolExport: "SpanMap",
  version: "1.0.0",
  classification: "hybrid",
  props: {
    id: { type: "string", required: true },
    requireCover: { type: "boolean", required: false },
  },
  slots: {},
  events: [],
  capabilityId: "core.span-map",
  leafId: "leaf.span-interval",
  deterministic: true,
};

const components: ComponentCatalog = { components: [spanMap] };

describe("validate component invocations", () => {
  test.each(["core.span-map", "SpanMap"])(
    "accepts known component %s with valid props",
    (name) => {
      const linked = linkedWith(
        invocation(name, { id: "tokens", requireCover: true }),
      );

      const result = validate(
        linked,
        FOUNDATION_CAPABILITIES,
        false,
        components,
      );

      expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    },
  );

  test("rejects an invocation absent from an authoritative component catalog", () => {
    const result = validate(
      linkedWith(invocation("UnknownWidget", {})),
      FOUNDATION_CAPABILITIES,
      false,
      components,
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "COMPONENT_UNKNOWN",
          severity: "error",
          message: expect.stringContaining("UnknownWidget"),
        }),
      ]),
    );
  });

  test("reports unknown, missing, and type-mismatched component props", () => {
    const result = validate(
      linkedWith(
        invocation("SpanMap", { requireCover: "yes", extra: true }),
      ),
      FOUNDATION_CAPABILITIES,
      false,
      components,
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics.map(({ code }) => code)).toEqual(
      expect.arrayContaining([
        "STRICT_UNKNOWN_PROP",
        "PROP_MISSING_REQUIRED",
        "PROP_TYPE_MISMATCH",
      ]),
    );
  });

  test("retains foundation diagnostics while validating components", () => {
    const result = validate(
      linkedWith(invocation("UnknownWidget", {}), false),
      FOUNDATION_CAPABILITIES,
      false,
      components,
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics.map(({ code }) => code)).toEqual(
      expect.arrayContaining(["COMPONENT_UNKNOWN", "ACCESSIBILITY_REQUIRED"]),
    );
  });

  test("preserves existing behavior when no component catalog is supplied", () => {
    const linked = linkedWith(invocation("UnknownWidget", {}));

    const result = validate(linked, FOUNDATION_CAPABILITIES, false);

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
  });
});
