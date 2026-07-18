/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { FOUNDATION_CAPABILITIES, type FlowIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { canonicalJson, compileSource, packFlow } from "../src/index.js";
import { FOUNDATION_SOURCE } from "./fixture.js";

function compileFoundation(): FlowIr {
  const result = compileSource({
    source: FOUNDATION_SOURCE,
    sourceName: "request-flow.flow",
    capabilities: FOUNDATION_CAPABILITIES,
    strict: false,
  });

  if (!result.ok) {
    throw new Error(
      `Expected the foundation source to compile: ${JSON.stringify(result.diagnostics)}`,
    );
  }

  return result.value;
}

describe("canonicalJson", () => {
  test("sorts object keys and drops undefined entries deterministically", () => {
    const first = canonicalJson({ b: 1, a: 2, c: undefined });
    const second = canonicalJson({ c: undefined, a: 2, b: 1 });

    expect(first).toEqual(second);
    expect(new TextDecoder().decode(first)).toBe('{"a":2,"b":1}');
  });

  test("sorts nested object keys recursively while preserving array order", () => {
    const bytes = canonicalJson({ z: [{ y: 1, x: 2 }], a: 1 });

    expect(new TextDecoder().decode(bytes)).toBe('{"a":1,"z":[{"x":2,"y":1}]}');
  });
});

describe("packFlow", () => {
  test("produces byte-identical packs for repeated compiles of the same source", () => {
    const packedA = packFlow(compileFoundation(), "request-flow.flow");
    const packedB = packFlow(compileFoundation(), "request-flow.flow");

    expect(packedA.manifest).toEqual(packedB.manifest);
    expect(packedA.files).toHaveLength(packedB.files.length);
    for (const [index, fileA] of packedA.files.entries()) {
      const fileB = packedB.files[index];
      expect(fileB?.path).toBe(fileA.path);
      expect(fileB?.hash).toBe(fileA.hash);
      expect(fileB?.mediaType).toBe(fileA.mediaType);
      expect(fileB?.content).toEqual(fileA.content);
    }
  });

  test("includes the manifest, scene chunks, and transcript with sorted paths", () => {
    const packed = packFlow(compileFoundation(), "request-flow.flow");
    const paths = packed.files.map((file) => file.path);

    expect(paths).toEqual([...paths].sort((left, right) => left.localeCompare(right)));
    expect(paths).toContain("flow.manifest.json");
    expect(paths).toContain("transcript.txt");
    expect(paths).toContain("chunks/scene-execution.json");
  });

  test("builds a manifest that references every scene chunk by hash", () => {
    const packed = packFlow(compileFoundation(), "request-flow.flow");

    expect(packed.manifest.formatVersion).toBe(1);
    expect(packed.manifest.id).toBe("request-flow");
    expect(packed.manifest.sourceName).toBe("request-flow.flow");
    expect(packed.manifest.transcriptPath).toBe("transcript.txt");
    expect(packed.manifest.capabilities.map((capability) => capability.id)).toEqual([
      "core.connector",
      "core.rect",
      "core.text",
    ]);
    expect(packed.manifest.scenes).toEqual([
      expect.objectContaining({
        id: "execution",
        chunkPath: "chunks/scene-execution.json",
      }),
    ]);

    const sceneChunk = packed.files.find(
      (file) => file.path === "chunks/scene-execution.json",
    );
    expect(packed.manifest.scenes[0]?.hash).toBe(sceneChunk?.hash);

    const manifestFile = packed.files.find(
      (file) => file.path === "flow.manifest.json",
    );
    expect(manifestFile).toBeDefined();
    expect(JSON.parse(new TextDecoder().decode(manifestFile?.content))).toEqual(
      packed.manifest,
    );
  });

  test("changing the source content changes the resulting hashes", () => {
    const packedA = packFlow(compileFoundation(), "request-flow.flow");

    const mutatedResult = compileSource({
      source: FOUNDATION_SOURCE.replace("#244a35", "#123456"),
      sourceName: "request-flow.flow",
      capabilities: FOUNDATION_CAPABILITIES,
      strict: false,
    });
    if (!mutatedResult.ok) {
      throw new Error("Expected mutated source to compile");
    }
    const packedB = packFlow(mutatedResult.value, "request-flow.flow");

    expect(packedA.manifest.contentHash).not.toBe(packedB.manifest.contentHash);
  });
});
