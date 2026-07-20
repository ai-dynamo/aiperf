/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { SymbolDefinitionAst } from "../language/ast.js";
import { createSdkRegistry } from "../sdk/registry.js";
import { expandSdkInvocations } from "./expand-sdk.js";

const SOURCE_RANGE = {
  source: "expand-sdk.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function nativeScene(body: string) {
  return {
    kind: "embedded-scene-source" as const,
    form: "native" as const,
    body,
  };
}

function expand(
  body: string,
  options?: {
    tokens?: ReadonlyMap<string, string>;
    symbols?: readonly SymbolDefinitionAst[];
  },
) {
  return expandSdkInvocations(nativeScene(body), {
    registry: createSdkRegistry(),
    sourceRange: SOURCE_RANGE,
    ...(options?.tokens !== undefined ? { tokens: options.tokens } : {}),
    ...(options?.symbols !== undefined ? { symbols: options.symbols } : {}),
  });
}

/** Legacy macro that expands to a registered SDK chrome component. */
function labeledNoteSymbol(): SymbolDefinitionAst {
  const param = (name: string): SymbolDefinitionAst["params"][number] => ({
    kind: "param",
    name,
    type: { kind: "type-ref", name: "string", sourceMap: SOURCE_RANGE },
    sourceMap: SOURCE_RANGE,
  });
  const idRef = {
    kind: "identifier-reference" as const,
    name: "id",
    sourceMap: SOURCE_RANGE,
  };
  const textRef = {
    kind: "identifier-reference" as const,
    name: "text",
    sourceMap: SOURCE_RANGE,
  };
  const num = (value: number) => ({
    kind: "literal" as const,
    value,
    sourceMap: SOURCE_RANGE,
  });
  return {
    kind: "symbol-definition",
    name: "LabeledNote",
    params: [param("id"), param("text")],
    body: [
      {
        kind: "component-invocation",
        namespace: "sdk",
        name: "Note",
        props: [
          {
            kind: "prop-assignment",
            name: "id",
            value: idRef,
            sourceMap: SOURCE_RANGE,
          },
          {
            kind: "prop-assignment",
            name: "text",
            value: textRef,
            sourceMap: SOURCE_RANGE,
          },
          {
            kind: "prop-assignment",
            name: "x",
            value: num(0),
            sourceMap: SOURCE_RANGE,
          },
          {
            kind: "prop-assignment",
            name: "y",
            value: num(0),
            sourceMap: SOURCE_RANGE,
          },
          {
            kind: "prop-assignment",
            name: "width",
            value: num(120),
            sourceMap: SOURCE_RANGE,
          },
          {
            kind: "prop-assignment",
            name: "height",
            value: num(40),
            sourceMap: SOURCE_RANGE,
          },
        ],
        sourceMap: SOURCE_RANGE,
      },
    ],
    sourceMap: SOURCE_RANGE,
  };
}

describe("expandSdkInvocations unknown @token props", () => {
  it("emits LINK_UNKNOWN_REFERENCE for an unresolved token reference", () => {
    const result = expand(`
      sdk.Note(id = "note", text = token(missingToken), x = 0, y = 0, width = 120, height = 40)
    `);

    expect(result.status).toBe("error");
    if (result.status !== "error") {
      return;
    }
    expect(result.diagnostics.some((d) => d.code === "LINK_UNKNOWN_REFERENCE")).toBe(
      true,
    );
    expect(
      result.diagnostics.some((d) => d.message.includes("missingToken")),
    ).toBe(true);
  });

  it("resolves a declared document token in SDK props", () => {
    const result = expand(
      `
      sdk.Note(id = "note", text = token(greeting), x = 0, y = 0, width = 120, height = 40)
    `,
      { tokens: new Map([["greeting", "Hello"]]) },
    );

    expect(result.status).toBe("ok");
    if (result.status !== "ok") {
      return;
    }
    const note = result.value.render.scene.roots.find((n) => n.id === "note");
    expect(note).toMatchObject({
      props: expect.objectContaining({ text: "Hello" }),
    });
  });
});

describe("expandSdkInvocations symbol / token wrap", () => {
  it("expands a legacy symbol macro that emits an SDK component", () => {
    const result = expand(
      `
      LabeledNote(id = "macro-note", text = "from macro")
      sdk.Chip(id = "chip", label = "keep", x = 200, y = 0, width = 60, height = 24)
    `,
      { symbols: [labeledNoteSymbol()] },
    );

    expect(result.status).toBe("ok");
    if (result.status !== "ok") {
      return;
    }
    const note = result.value.render.scene.roots.find((n) => n.id === "macro-note");
    expect(note).toMatchObject({
      capabilityId: "core.note",
      props: expect.objectContaining({ text: "from macro" }),
    });
    expect(result.value.instanceIndex.has("macro-note")).toBe(true);
  });

  it("resolves document tokens inside symbol macro arguments", () => {
    const result = expand(
      `
      LabeledNote(id = "macro-note", text = token(greeting))
      sdk.Chip(id = "chip", label = "keep", x = 200, y = 0, width = 60, height = 24)
    `,
      {
        symbols: [labeledNoteSymbol()],
        tokens: new Map([["greeting", "tokenized"]]),
      },
    );

    expect(result.status).toBe("ok");
    if (result.status !== "ok") {
      return;
    }
    const note = result.value.render.scene.roots.find((n) => n.id === "macro-note");
    expect(note).toMatchObject({
      capabilityId: "core.note",
      props: expect.objectContaining({ text: "tokenized" }),
    });
  });
});

describe("expandSdkInvocations SDK-attempt parse failures", () => {
  it("surfaces parse diagnostics as error when the body invokes SDK components", () => {
    const result = expand(`
      sdk.Note(id = "note" text = "hi", x = 0, y = 0, width = 120, height = 40)
    `);

    expect(result.status).toBe("error");
    if (result.status !== "error") {
      return;
    }
    expect(result.diagnostics.length).toBeGreaterThan(0);
    expect(
      result.diagnostics.some(
        (d) => d.severity === "error" && d.message.length > 0,
      ),
    ).toBe(true);
  });

  it("returns not-sdk for parse failures that do not target SDK authoring", () => {
    const result = expand(`
      @@@
    `);

    expect(result.status).toBe("not-sdk");
  });
});

describe("expandSdkInvocations stagger targets[]", () => {
  it("expands instance ids in targets[] to enter-bound node ids", () => {
    const result = expand(`
      sdk.Note(id = "a", text = "A", x = 0, y = 0, width = 80, height = 32)
      sdk.Note(id = "b", text = "B", x = 100, y = 0, width = 80, height = 32)
      timeline main {
        at 0 stagger targets [a, b] step 40 duration 100
      }
    `);

    expect(result.status).toBe("ok");
    if (result.status !== "ok") {
      return;
    }
    const stagger = result.value.render.scene.timeline.find(
      (cue) => cue.action === "stagger",
    );
    expect(stagger?.targets).toEqual(["a", "b"]);
  });

  it("expands a stack instance in targets[] through its stagger action bindings", () => {
    const result = expand(`
      sdk.Stack(id = "row", x = 0, y = 0, width = 240, height = 40) {
        children {
          sdk.Note(id = "child-a", text = "A", width = 80, height = 32)
          sdk.Note(id = "child-b", text = "B", width = 80, height = 32)
        }
      }
      timeline main {
        at 0 stagger targets [row] step 40 duration 100
      }
    `);

    expect(result.status).toBe("ok");
    if (result.status !== "ok") {
      return;
    }
    const stagger = result.value.render.scene.timeline.find(
      (cue) => cue.action === "stagger",
    );
    // Stack stagger binds child root ids, not the instance id alone.
    expect(stagger?.targets).toEqual(["child-a", "child-b"]);
  });

  it("emits SDK_TIMELINE_UNKNOWN_TARGET for unknown targets[] members", () => {
    const result = expand(`
      sdk.Note(id = "a", text = "A", x = 0, y = 0, width = 80, height = 32)
      timeline main {
        at 0 stagger targets [a, ghost] step 40 duration 100
      }
    `);

    expect(result.status).toBe("error");
    if (result.status !== "error") {
      return;
    }
    expect(
      result.diagnostics.some(
        (d) =>
          d.code === "SDK_TIMELINE_UNKNOWN_TARGET" && d.message.includes("ghost"),
      ),
    ).toBe(true);
  });
});
