/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  parseExplainerBlock,
  type TokenStream,
} from "../src/grammar/explainer.js";

// Mock TokenStream implementation for testing
class MockTokenStream implements TokenStream {
  private tokens: string[] = [];
  private position = 0;

  constructor(input: string) {
    // Tokenize while preserving quoted strings
    const tokens: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < input.length; i++) {
      const char = input[i];
      const nextChar = input[i + 1];

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

describe("Explainer Parser", () => {
  it("parses basic explainer block", () => {
    const code = `
      explainer "test-deck" {
        route: "/test"
        topic: "intro"
        eyebrowLabel: "Test"
        startGateTitle: "Go?"

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

    const tokens = new MockTokenStream(code);
    const ast = parseExplainerBlock(tokens);

    expect(ast.type).toBe("explainer");
    expect(ast.id).toBe("test-deck");
    expect(ast.metadata.route).toBe("/test");
    expect(ast.metadata.topic).toBe("intro");
    expect(ast.slides).toHaveLength(1);
    expect(ast.slides[0].narration).toBe("This is the first slide.");
  });

  it("parses slide with term definition", () => {
    const code = `
      explainer "test-term" {
        route: "/test"
        topic: "intro"
        eyebrowLabel: "Test"
        startGateTitle: "Go?"

        slide "With Term" {
          eyebrow: "Def"
          title: "Term Slide"
          lede: "Learning"
          narration: "Here we learn a term."
          term: { word: "Concept", meaning: "An idea" }
          points: []
          caption: "Glossary"
        }
      }
    `;

    const tokens = new MockTokenStream(code);
    const ast = parseExplainerBlock(tokens);

    expect(ast.slides[0].term).toEqual({ word: "Concept", meaning: "An idea" });
  });

  it("rejects missing narration", () => {
    const code = `
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

    const tokens = new MockTokenStream(code);
    expect(() => parseExplainerBlock(tokens)).toThrow(
      /narration.*required/i,
    );
  });
});
