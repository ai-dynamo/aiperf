/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { DIAGRAM_SDK_COMPONENTS } from "./diagram/catalog.js";
import { GENERIC_CATALOG_COMPONENTS } from "./generic/catalog.js";

function readDeck(name: string): string {
  const path = [
    resolve(process.cwd(), "decks-flow", name),
    resolve(process.cwd(), "apps/explainers/decks-flow", name),
  ].find(existsSync);
  if (path === undefined) {
    throw new Error(`cannot locate catalog deck ${name}`);
  }
  return readFileSync(path, "utf8");
}

function slideBodies(source: string): Map<string, string> {
  const starts = [...source.matchAll(/^\s*slide "([^"]+)" \{/gm)];
  return new Map(
    starts.map((match, index) => [
      match[1],
      source.slice(
        match.index,
        starts[index + 1]?.index ?? source.length,
      ),
    ]),
  );
}

function invocationFor(id: string): string {
  const name = id.slice("sdk.".length);
  return `sdk.${name[0].toUpperCase()}${name.slice(1)}(`;
}

describe("SDK catalog decks", () => {
  it.each([
    {
      name: "generic",
      source: readDeck("sdk-generic-catalog.flow"),
      components: GENERIC_CATALOG_COMPONENTS,
      slideCount: 63,
    },
    {
      name: "diagram",
      source: readDeck("sdk-diagram-catalog.flow"),
      components: DIAGRAM_SDK_COMPONENTS,
      slideCount: 49,
    },
  ])("$name deck gives every registered primitive one focused slide", ({
    source,
    components,
    slideCount,
  }) => {
    const slides = slideBodies(source);

    expect(slides.size).toBe(slideCount);
    for (const component of components) {
      const id = component.descriptor.id;
      const body = slides.get(id);
      expect(body, `missing focused slide ${id}`).toBeDefined();
      expect(body).toContain(invocationFor(id));
    }
  });
});
