/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type {
  ComponentInvocationAst,
  DocumentAst,
  PropAssignmentAst,
  SceneAst,
} from "../language/ast.js";
import type { ArgumentValueAst } from "../language/ast.js";
import {
  createComponentCatalog,
  FOUNDATION_CAPABILITIES,
  type ComponentDescriptor,
} from "../schema/index.js";
import { compileSource } from "./compile-source.js";
import type { LinkedDocument } from "./link.js";
import { lowerExplainerScene } from "./lower-explainer-scene.js";
import { validate } from "./validate.js";

const SOURCE_MAP = {
  source: "validate.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

const NARRATION =
  "This narration is long enough to satisfy the minimum length check.";

function literal(value: string | number | boolean): ArgumentValueAst {
  return { kind: "literal", value, sourceMap: SOURCE_MAP };
}

function prop(name: string, value: ArgumentValueAst): PropAssignmentAst {
  return { kind: "prop-assignment", name, value, sourceMap: SOURCE_MAP };
}

function widgetDescriptor(
  overrides: Partial<ComponentDescriptor> = {},
): ComponentDescriptor {
  return {
    id: "demo.widget",
    symbolExport: "Widget",
    version: "1.0.0",
    classification: "hybrid",
    props: {
      id: { type: "string", required: true },
      label: { type: "string", required: true },
    },
    slots: {},
    events: [],
    capabilityId: "core.group",
    deterministic: true,
    ...overrides,
  };
}

function linkedWithInvocation(
  invocation: ComponentInvocationAst,
): LinkedDocument {
  const scene: SceneAst = {
    kind: "scene",
    id: "scene",
    title: "Scene",
    sourceMap: SOURCE_MAP,
    readingOrder: {
      kind: "reading-order",
      references: ["widget"],
      sourceMap: SOURCE_MAP,
    },
    narration: { kind: "narration", text: NARRATION, sourceMap: SOURCE_MAP },
    renderDeclarations: [invocation],
    cameras: [],
    timelines: [],
    interactions: [],
    responsiveVariants: [],
  };

  const document: DocumentAst = {
    kind: "document",
    id: "doc",
    title: "Doc",
    sourceMap: SOURCE_MAP,
    language: { kind: "language", version: 1, sourceMap: SOURCE_MAP },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [scene],
  };

  return {
    document,
    tokens: new Map(),
    scenes: new Map([["scene", { nodes: new Map() }]]),
    imports: new Map(),
    qualifiedNames: new Map(),
    themes: [],
  };
}

describe("validate component invocations", () => {
  it("emits COMPONENT_UNKNOWN when the catalog is provided and the name is unregistered", () => {
    const catalog = createComponentCatalog([widgetDescriptor()]);
    expect(catalog.ok).toBe(true);
    if (!catalog.ok) {
      return;
    }

    const result = validate(
      linkedWithInvocation({
        kind: "component-invocation",
        name: "MissingThing",
        sourceMap: SOURCE_MAP,
        props: [prop("id", literal("widget"))],
      }),
      FOUNDATION_CAPABILITIES,
      false,
      catalog.value,
    );

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "COMPONENT_UNKNOWN"),
    ).toBe(true);
  });

  it("emits PROP_MISSING_REQUIRED for a registered component missing a required prop", () => {
    const catalog = createComponentCatalog([widgetDescriptor()]);
    expect(catalog.ok).toBe(true);
    if (!catalog.ok) {
      return;
    }

    const result = validate(
      linkedWithInvocation({
        kind: "component-invocation",
        name: "Widget",
        sourceMap: SOURCE_MAP,
        props: [prop("id", literal("widget"))],
      }),
      FOUNDATION_CAPABILITIES,
      false,
      catalog.value,
    );

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "PROP_MISSING_REQUIRED"),
    ).toBe(true);
  });

  it("accepts a registered component with required props satisfied", () => {
    const catalog = createComponentCatalog([widgetDescriptor()]);
    expect(catalog.ok).toBe(true);
    if (!catalog.ok) {
      return;
    }

    const result = validate(
      linkedWithInvocation({
        kind: "component-invocation",
        name: "Widget",
        sourceMap: SOURCE_MAP,
        props: [
          prop("id", literal("widget")),
          prop("label", literal("Ready")),
        ],
      }),
      FOUNDATION_CAPABILITIES,
      false,
      catalog.value,
    );

    expect(result.ok).toBe(true);
  });
});

describe("compileSource wires the component catalog", () => {
  it("reports PROP_MISSING_REQUIRED through compileSource when components are supplied", () => {
    const catalog = createComponentCatalog([widgetDescriptor()]);
    expect(catalog.ok).toBe(true);
    if (!catalog.ok) {
      return;
    }

    const result = compileSource({
      source: `
flow "Doc" as doc {
  language 1
  scene "Scene" as scene {
    Widget(id = "x")
    narrate "${NARRATION}"
    reading-order x
  }
}
`,
      sourceName: "compile-source-components.test.flow",
      capabilities: FOUNDATION_CAPABILITIES,
      strict: false,
      components: catalog.value,
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "PROP_MISSING_REQUIRED"),
    ).toBe(true);
  });
});
