/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  FOUNDATION_CAPABILITIES,
  type DeckPackage,
  type SceneIr,
  type SourceRange,
} from "@aiperf/flow-schema";
import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  compileExplainerSource,
  type CompileExplainerRequest,
} from "../src/compile-explainer.js";

const sourceMap: SourceRange = {
  source: "deck.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

vi.mock("@aiperf/flow-language", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@aiperf/flow-language")>();
  return {
    ...actual,
    parseDocument: vi.fn(),
  };
});

vi.mock("../src/lower-explainer.js", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../src/lower-explainer.js")>();
  return {
    ...actual,
    lowerExplainerToDeckPackage: vi.fn(actual.lowerExplainerToDeckPackage),
  };
});

import { parseDocument } from "@aiperf/flow-language";

import { lowerExplainerToDeckPackage } from "../src/lower-explainer.js";

const parseDocumentMock = vi.mocked(parseDocument);
const lowerMock = vi.mocked(lowerExplainerToDeckPackage);

const { lowerExplainerToDeckPackage: realLower } = await vi.importActual<
  typeof import("../src/lower-explainer.js")
>("../src/lower-explainer.js");

function request(
  overrides: Partial<CompileExplainerRequest> = {},
): CompileExplainerRequest {
  return {
    source: 'explainer "Rust Architecture" { }',
    sourceName: "deck.flow",
    capabilities: FOUNDATION_CAPABILITIES,
    strict: true,
    ...overrides,
  };
}

function explainerDocument() {
  return {
    kind: "document" as const,
    id: "rust-architecture",
    title: "rust-architecture",
    language: { kind: "language" as const, version: 1 as const, sourceMap },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [],
    sourceMap,
    explainers: [
      {
        kind: "explainer" as const,
        id: "rust-architecture",
        sourceMap,
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
            kind: "slide" as const,
            sourceMap,
            eyebrow: "Product shell",
            title: "One binary is both CLI and engine",
            lede: "AIPerf ships as one native binary.",
            narration: "AIPerf ships as one native aiperf binary.",
            points: ["CLI and engine share one process."],
            caption: "Product shell overview",
            term: { word: "aiperf-cli", meaning: "Native CLI crate" },
          },
        ],
      },
    ],
  };
}

function emptyTimelineScene(): SceneIr {
  return {
    id: "main",
    title: "Main",
    summary: "A diagram slide",
    roots: [
      {
        kind: "rect",
        id: "box",
        geometry: { x: 0, y: 0, width: 100, height: 40 },
        style: {},
        accessibility: { label: "Box" },
        fallback: "Box unavailable",
        sourceMap,
      },
    ],
    camera: [],
    timeline: [],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: { label: "Main scene", readingOrder: ["box"] },
    fallback: "Scene unavailable",
    sourceMap,
  };
}

function basePackage(): DeckPackage {
  return {
    schemaVersion: 1,
    id: "rust-architecture",
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
    slides: [
      {
        id: "product-shell",
        eyebrow: "Product shell",
        title: "One binary is both CLI and engine",
        lede: "AIPerf ships as one native binary.",
        narration: "AIPerf ships as one native aiperf binary.",
        points: ["CLI and engine share one process."],
        caption: "Product shell overview",
        term: { word: "aiperf-cli", meaning: "Native CLI crate" },
      },
    ],
    glossary: [{ word: "aiperf-cli", meaning: "Native CLI crate" }],
  };
}

describe("compileExplainerSource", () => {
  beforeEach(() => {
    parseDocumentMock.mockReset();
    lowerMock.mockReset();
    lowerMock.mockImplementation(realLower);
  });

  test("parses, lowers, validates timelines, and schema-checks a DeckPackage", () => {
    parseDocumentMock.mockReturnValue({
      ok: true,
      value: explainerDocument(),
      diagnostics: [],
    });

    const result = compileExplainerSource(request());

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.schemaVersion).toBe(1);
    expect(result.value.id).toBe("rust-architecture");
    expect(result.value.route).toBe("/rust-architecture");
    expect(result.value.slides).toHaveLength(1);
    expect(result.value.slides[0]).toMatchObject({
      title: "One binary is both CLI and engine",
      narration: "AIPerf ships as one native aiperf binary.",
    });
    expect(result.value.slides[0]?.render).toBeUndefined();
    expect(parseDocumentMock).toHaveBeenCalledWith(
      request().source,
      "deck.flow",
    );
  });

  test("short-circuits on parse failure", () => {
    parseDocumentMock.mockReturnValue({
      ok: false,
      diagnostics: [
        {
          code: "PARSE_ERROR",
          severity: "error",
          message: "boom",
          range: sourceMap,
        },
      ],
    });

    const result = compileExplainerSource(request());

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics[0]?.message).toBe("boom");
    }
    expect(lowerMock).not.toHaveBeenCalled();
  });

  test("rejects documents without an explainer block", () => {
    parseDocumentMock.mockReturnValue({
      ok: true,
      value: {
        ...explainerDocument(),
        explainers: [],
      },
      diagnostics: [],
    });

    const result = compileExplainerSource(request());

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics[0]?.code).toBe("EXPLAINER_REQUIRED");
    }
  });

  test("rejects empty narration during lower", () => {
    const document = explainerDocument();
    document.explainers[0]!.slides = [
      {
        kind: "slide",
        sourceMap,
        eyebrow: "Broken",
        title: "Missing narration",
        lede: "Lede",
        narration: "   ",
        points: [],
        caption: "Caption",
      },
    ];

    parseDocumentMock.mockReturnValue({
      ok: true,
      value: document,
      diagnostics: [],
    });

    const result = compileExplainerSource(request());

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics.some((d) => /narration/.test(d.message))).toBe(
        true,
      );
    }
  });

  test("rejects scene slides with empty timelines after lower", () => {
    parseDocumentMock.mockReturnValue({
      ok: true,
      value: explainerDocument(),
      diagnostics: [],
    });

    const base = basePackage();
    const pkg: DeckPackage = {
      ...base,
      slides: [
        {
          ...base.slides[0]!,
          render: { kind: "scene", scene: emptyTimelineScene() },
        },
      ],
    };

    lowerMock.mockReturnValue({
      ok: true,
      value: pkg,
      diagnostics: [],
    });

    const result = compileExplainerSource(request());

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics).toEqual([
        expect.objectContaining({
          code: "EXPLAINER_TIMELINE_REQUIRED",
          severity: "error",
        }),
      ]);
    }
  });
});
