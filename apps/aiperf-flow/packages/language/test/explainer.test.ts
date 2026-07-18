/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  parseExplainerBlock,
  type TokenStream,
} from "../src/grammar/explainer.js";

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
      } else if (!inQuotes && /[{}\[\]:,@]/.test(char)) {
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
        roots: []
        timeline: []
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
    expect(ast.slides[0].sceneIr).toEqual({ type: "scene" });
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
