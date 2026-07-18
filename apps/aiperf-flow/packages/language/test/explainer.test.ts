/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  parseExplainerBlock,
  type TokenStream,
} from "../src/grammar/explainer.js";
import { parseDocument } from "../src/parser.js";

/** Mock TokenStream implementation for testing. */
class MockTokenStream implements TokenStream {
  private tokens: string[] = [];
  private position = 0;

  constructor(input: string) {
    const tokens: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < input.length; i++) {
      const char = input[i];

      if (char === '"') {
        inQuotes = !inQuotes;
        current += char;
      } else if (!inQuotes && /\s/.test(char)) {
        if (current) {
          tokens.push(current);
          current = "";
        }
      } else if (!inQuotes && /[{}\[\]:,@.]/.test(char)) {
        if (current) {
          tokens.push(current);
          current = "";
        }
        tokens.push(char);
      } else {
        current += char;
      }
    }

    if (current) {
      tokens.push(current);
    }

    this.tokens = tokens.filter((t) => t && t !== "");
  }

  expect(keyword: string): void {
    const current = this.tokens[this.position];
    if (current?.replace(/"/g, "") !== keyword) {
      throw new Error(
        `Expected "${keyword}" but got "${current}" at position ${this.position}`,
      );
    }
    this.position++;
  }

  expectString(): string {
    const current = this.tokens[this.position];
    if (!current) {
      throw new Error(`Expected string but reached end of input`);
    }
    this.position++;
    return current.replace(/"/g, "");
  }

  expectIdentifier(): string {
    const current = this.tokens[this.position];
    if (!current || current.match(/[{}\[\]:,@"]/)) {
      throw new Error(
        `Expected identifier but got "${current}" at position ${this.position}`,
      );
    }
    this.position++;
    return current;
  }

  match(keyword: string): boolean {
    const current = this.tokens[this.position];
    return current?.replace(/"/g, "") === keyword;
  }

  advance(): void {
    this.position++;
  }

  peekImage(): string | undefined {
    return this.tokens[this.position];
  }
}

const FULL_EXPLAINER = `
  explainer "Rust Architecture" {
    id: "rust-architecture"
    route: "/rust-architecture"
    topic: "architecture"
    storagePrefix: "rust-arch-explainer"
    classPrefix: "rust-arch"
    eyebrowLabel: "RUST ARCHITECTURE"
    startGateTitle: "Rust architecture walkthrough"
    hub: {
      title: "from scratch"
      highlight: "Rust architecture"
      description: "Narrated walkthrough of the native workspace."
    }

    slide "Product shell" {
      eyebrow: "Product shell"
      title: "One binary is both CLI and engine"
      lede: "The product entry point."
      narration: "AIPerf ships as one native aiperf binary."
      term: { word: "aiperf-cli", meaning: "CLI and engine crate" }
      points: ["One binary", "Native profile"]
      caption: "Product shell overview"
      render: @scene {
        roots: [
          {
            id: "box"
            capability: "core.rect"
            layout: { x: 10, y: 20, width: 100, height: 40 }
            style: { fill: "#244a35" }
            accessibility: { label: "Box" }
          }
        ]
        timeline: [
          {
            id: "enter-box"
            at: 0
            duration: 400
            target: "box"
            action: "enter"
          }
        ]
      }
    }
  }
`;

describe("Explainer Parser", () => {
  it("parses id from the id field", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.type).toBe("explainer");
    expect(ast.id).toBe("rust-architecture");
  });

  it("parses route", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.route).toBe("/rust-architecture");
  });

  it("parses topic", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.topic).toBe("architecture");
  });

  it("parses storagePrefix", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.storagePrefix).toBe("rust-arch-explainer");
  });

  it("parses classPrefix", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.classPrefix).toBe("rust-arch");
  });

  it("parses eyebrowLabel", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.eyebrowLabel).toBe("RUST ARCHITECTURE");
  });

  it("parses startGateTitle", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.startGateTitle).toBe("Rust architecture walkthrough");
  });

  it("parses hub title, highlight, and description", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.metadata.hub).toEqual({
      title: "from scratch",
      highlight: "Rust architecture",
      description: "Narrated walkthrough of the native workspace.",
    });
  });

  it("parses slide text fields", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    expect(ast.slides).toHaveLength(1);
    const slide = ast.slides[0];
    expect(slide.eyebrow).toBe("Product shell");
    expect(slide.title).toBe("One binary is both CLI and engine");
    expect(slide.lede).toBe("The product entry point.");
    expect(slide.narration).toBe("AIPerf ships as one native aiperf binary.");
    expect(slide.term).toEqual({
      word: "aiperf-cli",
      meaning: "CLI and engine crate",
    });
    expect(slide.points).toEqual(["One binary", "Native profile"]);
    expect(slide.caption).toBe("Product shell overview");
  });

  it("parses render:@scene into sceneIr", () => {
    const ast = parseExplainerBlock(new MockTokenStream(FULL_EXPLAINER));
    const sceneIr = ast.slides[0].sceneIr as {
      kind: string;
      id: string;
      roots: unknown[];
      timeline: unknown[];
      renderDeclarations: Array<{ kind: string; id: string }>;
      timelines: Array<{ cues: Array<{ target: string; action: string }> }>;
    };
    expect(sceneIr.kind).toBe("scene");
    expect(sceneIr.id).toBe("embedded-scene");
    expect(sceneIr.roots).toHaveLength(1);
    expect(sceneIr.timeline).toHaveLength(1);
    expect(sceneIr.renderDeclarations[0]).toMatchObject({
      kind: "rect",
      id: "box",
    });
    expect(sceneIr.timelines[0]?.cues[0]).toMatchObject({
      target: "box",
      action: "reveal",
    });
  });

  it("falls back to header string when id field is absent", () => {
    const code = `
      explainer "legacy-deck-id" {
        route: "/legacy"
        topic: "intro"
        storagePrefix: "legacy-store"
        classPrefix: "legacy"
        eyebrowLabel: "Legacy"
        startGateTitle: "Start"
        hub: {
          title: "hub"
          highlight: "hi"
          description: "desc"
        }

        slide "Only" {
          eyebrow: "E"
          title: "T"
          lede: "L"
          narration: "Narration text."
          points: []
          caption: "C"
        }
      }
    `;
    const ast = parseExplainerBlock(new MockTokenStream(code));
    expect(ast.id).toBe("legacy-deck-id");
  });

  it("rejects missing narration", () => {
    const code = `
      explainer "bad" {
        id: "bad"
        route: "/bad"
        topic: "intro"
        storagePrefix: "bad-store"
        classPrefix: "bad"
        eyebrowLabel: "Bad"
        startGateTitle: "?"
        hub: {
          title: "t"
          highlight: "h"
          description: "d"
        }

        slide "No Narration" {
          eyebrow: "Bad"
          title: "Broken"
          lede: "Missing"
          points: []
          caption: ""
        }
      }
    `;

    expect(() => parseExplainerBlock(new MockTokenStream(code))).toThrow(
      /narration.*required/i,
    );
  });

  it("rejects missing hub", () => {
    const code = `
      explainer "no-hub" {
        id: "no-hub"
        route: "/no-hub"
        topic: "intro"
        storagePrefix: "x"
        classPrefix: "x"
        eyebrowLabel: "X"
        startGateTitle: "X"

        slide "S" {
          eyebrow: "E"
          title: "T"
          lede: "L"
          narration: "N"
          points: []
          caption: "C"
        }
      }
    `;

    expect(() => parseExplainerBlock(new MockTokenStream(code))).toThrow(
      /hub.*required/i,
    );
  });

  it("rejects missing storagePrefix", () => {
    const code = `
      explainer "missing-prefix" {
        id: "missing-prefix"
        route: "/missing"
        topic: "intro"
        classPrefix: "x"
        eyebrowLabel: "X"
        startGateTitle: "X"
        hub: {
          title: "t"
          highlight: "h"
          description: "d"
        }

        slide "S" {
          eyebrow: "E"
          title: "T"
          lede: "L"
          narration: "N"
          points: []
          caption: "C"
        }
      }
    `;

    expect(() => parseExplainerBlock(new MockTokenStream(code))).toThrow(
      /storagePrefix.*required/i,
    );
  });
});

describe("parseDocument explainer", () => {
  it("parses an explainer-only .flow file into document.explainers", () => {
    const source = `
explainer "test-deck" {
  route: "/test"
  topic: "intro"
  storagePrefix: "test-store"
  classPrefix: "test"
  eyebrowLabel: "Test"
  startGateTitle: "Go?"
  hub: {
    title: "from scratch"
    highlight: "Test"
    description: "Test deck."
  }

  slide "First Slide" {
    eyebrow: "Intro"
    title: "Welcome"
    lede: "Getting started"
    narration: "This is the first slide."
    points: ["Point 1", "Point 2"]
    caption: "Test slide"
  }
}
`;

    const result = parseDocument(source, "test-deck.flow");
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) return;

    expect(result.value.explainers).toHaveLength(1);
    const explainer = result.value.explainers![0]!;
    expect(explainer.kind).toBe("explainer");
    expect(explainer.id).toBe("test-deck");
    expect(explainer.metadata.route).toBe("/test");
    expect(explainer.metadata.topic).toBe("intro");
    expect(explainer.metadata.eyebrowLabel).toBe("Test");
    expect(explainer.metadata.startGateTitle).toBe("Go?");
    expect(explainer.slides).toHaveLength(1);
    expect(explainer.slides[0]!.kind).toBe("slide");
    expect(explainer.slides[0]!.narration).toBe("This is the first slide.");
    expect(explainer.slides[0]!.points).toEqual(["Point 1", "Point 2"]);
    expect(result.value.scenes).toEqual([]);
  });

  it("rejects explainer-only file with missing narration", () => {
    const source = `
explainer "bad" {
  route: "/bad"
  topic: "intro"
  eyebrowLabel: "Bad"
  startGateTitle: "?"

  slide "No Narration" {
    eyebrow: "Bad"
    title: "Broken"
    lede: "Missing"
    points: []
    caption: ""
  }
}
`;

    const result = parseDocument(source, "bad.flow");
    expect(result.ok).toBe(false);
  });
});
