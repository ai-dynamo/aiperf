/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { extractScenes, matchBrace } from "./assert-sdk-authoring.mjs";

describe("assert-sdk-authoring matchBrace comment awareness", () => {
  it("ignores braces inside // line comments", () => {
    const source = '{ // fake } here\n  sdk.card { id: "a" }\n}';
    expect(matchBrace(source, 0)).toBe(source.length);
  });

  it("ignores braces inside /* */ block comments", () => {
    const source = '{ /* stray } */\n  sdk.card { id: "a" }\n}';
    expect(matchBrace(source, 0)).toBe(source.length);
  });

  it("still skips braces inside double-quoted strings", () => {
    const source = '{ label: "has } brace"\n}';
    expect(matchBrace(source, 0)).toBe(source.length);
  });

  it("does not treat // or /* inside strings as comments", () => {
    const source = '{ label: "not // a } comment"\n}';
    expect(matchBrace(source, 0)).toBe(source.length);
  });
});

describe("assert-sdk-authoring extractScenes with commented braces", () => {
  it("extracts the full scene body when comments contain braces", () => {
    const source = `
deck "demo" {
  beat "one" {
    render: @scene {
      // ignore this }
      /* and this } too */
      sdk.card { id: "ok" label: "OK" }
    }
  }
}
`;
    const scenes = extractScenes(source);
    expect(scenes).toHaveLength(1);
    expect(scenes[0].body).toContain('sdk.card { id: "ok" label: "OK" }');
    expect(scenes[0].body).toContain("// ignore this }");
    // Must not end early at the commented brace.
    expect(scenes[0].body.trimEnd()).toMatch(/sdk\.card \{ id: "ok" label: "OK" \}\s*$/);
  });
});
