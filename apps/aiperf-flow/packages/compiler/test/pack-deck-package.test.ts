/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  parseDeckPackage,
  type DeckPackage,
} from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  packDeckPackageToJson,
  writeDeckPackage,
} from "../src/pack-deck-package.js";

const sourceMap = {
  source: "deck.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

const minimalScene = {
  id: "main",
  title: "Main",
  summary: "A diagram slide",
  roots: [
    {
      kind: "rect" as const,
      id: "box",
      geometry: { x: 0, y: 0, width: 100, height: 40 },
      style: {},
      accessibility: { label: "Box" },
      fallback: "Box unavailable",
      sourceMap,
    },
  ],
  camera: [],
  timeline: [
    {
      id: "enter-box",
      at: 0,
      duration: 200,
      target: "box",
      action: "enter" as const,
      sourceMap,
    },
  ],
  narration: "",
  interactions: [],
  responsive: [],
  accessibility: { label: "Main scene", readingOrder: ["box"] },
  fallback: "Scene unavailable",
  sourceMap,
};

function samplePackage(): DeckPackage {
  return parseDeckPackage({
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
        render: { kind: "scene", scene: minimalScene },
      },
    ],
    glossary: [{ word: "aiperf-cli", meaning: "Native CLI crate" }],
  });
}

describe("packDeckPackageToJson", () => {
  test("emits schemaVersion:1 JSON that round-trips through parseDeckPackage", () => {
    const pkg = samplePackage();
    const json = packDeckPackageToJson(pkg);
    const parsed = JSON.parse(json) as unknown;

    expect(parsed).toEqual(
      expect.objectContaining({ schemaVersion: 1, id: "rust-architecture" }),
    );
    expect(parseDeckPackage(parsed).id).toBe(pkg.id);
    expect(parseDeckPackage(parsed).slides).toHaveLength(1);
  });

  test("is deterministic for the same package", () => {
    const pkg = samplePackage();
    expect(packDeckPackageToJson(pkg)).toBe(packDeckPackageToJson(pkg));
  });
});

describe("writeDeckPackage", () => {
  test("writes schemaVersion:1 JSON to the given path", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pack-deck-package-"));
    const path = join(dir, "rust-architecture.package.json");

    try {
      const pkg = samplePackage();
      await writeDeckPackage(path, pkg);

      const onDisk = await readFile(path, "utf8");
      expect(onDisk).toBe(packDeckPackageToJson(pkg));
      expect(JSON.parse(onDisk)).toEqual(
        expect.objectContaining({ schemaVersion: 1 }),
      );
      expect(parseDeckPackage(JSON.parse(onDisk)).id).toBe(pkg.id);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
