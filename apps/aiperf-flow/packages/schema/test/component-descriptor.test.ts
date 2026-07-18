/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import {
  createComponentCatalog,
  parseComponentDescriptor,
  safeParseComponentDescriptor,
  type ComponentDescriptor,
} from "../src/component-descriptor.js";

const hybridDescriptor: ComponentDescriptor = {
  id: "core.span-map",
  symbolExport: "SpanMap",
  version: "1.0.0",
  classification: "hybrid",
  props: {
    requireCover: { type: "boolean", required: false, default: true },
    id: { type: "string", required: true },
  },
  slots: {
    "target-view": { accepts: "SemanticEntity", required: true },
    "edge-chrome": { accepts: "TokenRibbon", required: false },
  },
  events: ["on-inspect", "on-focus"],
  capabilityId: "core.span-map",
  leafId: "leaf.span-interval",
  deterministic: true,
};

const flowOnlyDescriptor: ComponentDescriptor = {
  id: "core.semantic-entity",
  symbolExport: "SemanticEntity",
  version: "1.0.0",
  classification: "flow-only",
  props: {
    label: { type: "string", required: true },
  },
  slots: {},
  events: [],
  capabilityId: "core.group",
  deterministic: true,
};

describe("parseComponentDescriptor", () => {
  it("accepts a hybrid descriptor with props, slots, events, and leafId", () => {
    expect(parseComponentDescriptor(hybridDescriptor)).toEqual(hybridDescriptor);
  });

  it("accepts a flow-only descriptor without leafId", () => {
    expect(parseComponentDescriptor(flowOnlyDescriptor)).toEqual(flowOnlyDescriptor);
  });

  it("rejects unknown top-level fields", () => {
    expect(() =>
      parseComponentDescriptor({
        ...flowOnlyDescriptor,
        extra: "field",
      }),
    ).toThrow();
  });

  it("rejects invalid classification values", () => {
    expect(() =>
      parseComponentDescriptor({
        ...flowOnlyDescriptor,
        classification: "runtime",
      }),
    ).toThrow();
  });

  it("rejects unknown prop fields", () => {
    expect(() =>
      parseComponentDescriptor({
        ...flowOnlyDescriptor,
        props: {
          label: { type: "string", required: true, unknown: true },
        },
      }),
    ).toThrow();
  });

  it("rejects missing required descriptor fields", () => {
    const { capabilityId: _capabilityId, ...incomplete } = flowOnlyDescriptor;
    expect(() => parseComponentDescriptor(incomplete)).toThrow();
  });
});

describe("safeParseComponentDescriptor", () => {
  it("returns ok for a valid descriptor", () => {
    const result = safeParseComponentDescriptor(hybridDescriptor);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toEqual(hybridDescriptor);
      expect(result.diagnostics).toEqual([]);
    }
  });

  it("returns diagnostics for invalid input", () => {
    const result = safeParseComponentDescriptor({
      ...flowOnlyDescriptor,
      classification: "invalid",
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics.length).toBeGreaterThan(0);
      expect(result.diagnostics[0]?.code).toBe("COMPONENT_INVALID");
      expect(result.diagnostics[0]?.severity).toBe("error");
    }
  });
});

describe("createComponentCatalog", () => {
  it("returns a sorted catalog for unique component ids", () => {
    const result = createComponentCatalog([
      hybridDescriptor,
      flowOnlyDescriptor,
    ]);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.components.map(({ id }) => id)).toEqual([
        "core.semantic-entity",
        "core.span-map",
      ]);
    }
  });

  it("fails when duplicate component ids are present", () => {
    const duplicate: ComponentDescriptor = {
      ...flowOnlyDescriptor,
      symbolExport: "SemanticEntityClone",
      version: "2.0.0",
    };
    const result = createComponentCatalog([flowOnlyDescriptor, duplicate]);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics).toEqual([
        expect.objectContaining({
          code: "COMPONENT_DUPLICATE",
          severity: "error",
          message: expect.stringContaining("core.semantic-entity"),
        }),
      ]);
    }
  });
});
