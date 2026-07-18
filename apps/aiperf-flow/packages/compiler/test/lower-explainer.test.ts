/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { parseDocument } from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import {
  FOUNDATION_CAPABILITIES,
  safeParseDeckPackage,
  type CapabilityRegistryManifest,
  type DeckPackage,
  type Result,
} from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  lowerExplainerToDeckPackage,
  slideIdFromTitle,
  type ExplainerLowerInput,
} from "../src/lower-explainer.js";

function range(): SourceRange {
  return {
    source: "deck.flow",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 1, line: 1, column: 2 },
  };
}

function sampleAst(
  overrides: Partial<ExplainerLowerInput> = {},
): ExplainerLowerInput {
  return {
    kind: "explainer",
    id: "rust-architecture",
    sourceMap: range(),
    metadata: {
      route: "/rust-architecture",
      topic: "architecture",
      storagePrefix: "rust-arch-explainer",
      classPrefix: "rust-arch",
      eyebrowLabel: "RUST ARCHITECTURE",
      startGateTitle: "Rust architecture walkthrough",
      hub: {
        title: "from scratch",
        highlight: "Rust architecture",
        description: "Narrated walkthrough of the native workspace.",
      },
    },
    slides: [
      {
        kind: "slide",
        sourceMap: range(),
        eyebrow: "Product shell",
        title: "One binary is both CLI and engine",
        lede: "AIPerf ships as one native binary.",
        narration: "AIPerf ships as one native aiperf binary.",
        term: { word: "aiperf-cli", meaning: "Native CLI crate" },
        points: ["CLI and engine share one process."],
        caption: "Product shell overview",
        sceneIr: {
          kind: "scene",
          id: "ignored",
          title: "Ignored",
          summary: {
            kind: "summary",
            text: "Should not be lowered yet.",
            sourceMap: range(),
          },
          renderDeclarations: [],
          cameras: [],
          timelines: [],
          interactions: [],
          responsiveVariants: [],
          sourceMap: range(),
        },
      },
    ],
    ...overrides,
  };
}

describe("slideIdFromTitle", () => {
  test("slugifies titles", () => {
    expect(slideIdFromTitle("Product shell", 0)).toBe("product-shell");
  });

  test("falls back to index when title is empty", () => {
    expect(slideIdFromTitle("   ", 2)).toBe("slide-2");
  });
});

describe("lowerExplainerToDeckPackage", () => {
  test("maps metadata and slide text into schemaVersion 1 without render", () => {
    const result = lowerExplainerToDeckPackage(sampleAst());

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.schemaVersion).toBe(1);
    expect(result.value.id).toBe("rust-architecture");
    expect(result.value.route).toBe("/rust-architecture");
    expect(result.value.storagePrefix).toBe("rust-arch-explainer");
    expect(result.value.classPrefix).toBe("rust-arch");
    expect(result.value.hub.highlight).toBe("Rust architecture");
    expect(result.value.slides).toHaveLength(1);
    expect(result.value.slides[0]).toMatchObject({
      id: "one-binary-is-both-cli-and-engine",
      title: "One binary is both CLI and engine",
      narration: "AIPerf ships as one native aiperf binary.",
      term: { word: "aiperf-cli", meaning: "Native CLI crate" },
    });
    expect(result.value.slides[0]?.render).toBeUndefined();
    expect(result.value.glossary).toEqual([
      { word: "aiperf-cli", meaning: "Native CLI crate" },
    ]);
  });

  test("rejects empty narration", () => {
    const result = lowerExplainerToDeckPackage(
      sampleAst({
        slides: [
          {
            kind: "slide",
            sourceMap: range(),
            eyebrow: "Bad",
            title: "Missing narration",
            lede: "Lede",
            narration: "   ",
            points: [],
            caption: "Caption",
          },
        ],
      }),
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(
        result.diagnostics.some((d) => d.code === "EXPLAINER_FIELD_REQUIRED"),
      ).toBe(true);
    }
  });

  test("rejects missing hub fields", () => {
    const ast = sampleAst();
    const result = lowerExplainerToDeckPackage({
      ...ast,
      metadata: {
        ...ast.metadata,
        hub: { title: "", highlight: "x", description: "y" },
      },
    });

    expect(result.ok).toBe(false);
  });

  test("preserves explicit glossary over slide terms", () => {
    const result = lowerExplainerToDeckPackage(
      sampleAst({
        glossary: [{ word: "cell", meaning: "Worker process" }],
      }),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.glossary).toEqual([
        { word: "cell", meaning: "Worker process" },
      ]);
    }
  });
});

const sourceName = "minimal-explainer.flow";
const source = readFileSync(
  path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "fixtures",
    sourceName,
  ),
  "utf8",
);

/** Prefer compileExplainerSource when present; else parse + lowerExplainerDocument. */
async function produceDeckPackage(): Promise<Result<DeckPackage>> {
  const index = await import("../src/index.js");
  const compileExplainerSource = (
    index as {
      compileExplainerSource?: (request: {
        source: string;
        sourceName: string;
        capabilities: CapabilityRegistryManifest;
      }) => Result<DeckPackage>;
    }
  ).compileExplainerSource;

  if (typeof compileExplainerSource === "function") {
    return compileExplainerSource({
      source,
      sourceName,
      capabilities: FOUNDATION_CAPABILITIES,
    });
  }

  const lowerMod = await import("../src/lower-explainer.js");
  const lowerExplainerDocument = (
    lowerMod as {
      lowerExplainerDocument?: (
        doc: unknown,
        capabilities: CapabilityRegistryManifest,
      ) => Result<DeckPackage>;
    }
  ).lowerExplainerDocument;

  if (typeof lowerExplainerDocument !== "function") {
    return {
      ok: false,
      diagnostics: [
        {
          severity: "error",
          code: "EXPLAINER_LOWER_MISSING",
          message:
            "lowerExplainerDocument / compileExplainerSource not available yet",
          range: {
            source: sourceName,
            start: { offset: 0, line: 1, column: 1 },
            end: { offset: 0, line: 1, column: 1 },
          },
        },
      ],
    };
  }

  const parsed = parseDocument(source, sourceName);
  if (!parsed.ok) {
    return parsed;
  }

  return lowerExplainerDocument(parsed.value, FOUNDATION_CAPABILITIES);
}

describe("explainer DeckPackage pipeline", () => {
  test("minimal-explainer.flow produces a valid DeckPackage with scene enter cue", async () => {
    const result = await produceDeckPackage();
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const validated = safeParseDeckPackage(result.value);
    expect(validated.ok, JSON.stringify(validated.diagnostics)).toBe(true);
    if (!validated.ok) {
      return;
    }

    expect(validated.value.schemaVersion).toBe(1);
    expect(validated.value.id).toBe("minimal-explainer");
    expect(validated.value.route).toBe("/minimal-explainer");
    expect(validated.value.slides).toHaveLength(1);

    const slide = validated.value.slides[0]!;
    expect(slide.narration.trim().length).toBeGreaterThan(0);
    expect(slide.render?.kind).toBe("scene");

    const timeline = slide.render?.scene.timeline ?? [];
    expect(timeline.length).toBeGreaterThan(0);
    expect(timeline.some((cue) => cue.action === "enter")).toBe(true);
  });
});
