/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { CHAPTERS, PAGES, pageById, pagesForChapter } from "./catalog.js";

describe("mock-server-architecture catalog", () => {
  it("carries exactly the audited 64 pages with unique ids", () => {
    expect(PAGES).toHaveLength(64);
    expect(new Set(PAGES.map((p) => p.id)).size).toBe(64);
  });

  it("has all ten chapters and every chapter is non-empty", () => {
    expect(CHAPTERS).toHaveLength(10);
    for (const chapter of CHAPTERS) {
      expect(pagesForChapter(chapter.id).length).toBeGreaterThan(0);
    }
  });

  it("preserves load-bearing source/proof paths and invariants verbatim", () => {
    const journey = pageById("request-journey");
    expect(journey.source).toBe("rust/mock-server/src/handlers.rs");
    expect(journey.proof).toBe("rust/e2e/tests/test_tuned_raw_timing.rs");
    expect(journey.invariant).toBe(
      "A request crosses parsing, token budgeting, latency, streaming, and accounting in one server process.",
    );
    expect(pageById("riva-nlp-boundary").status).toBe("boundary");
    expect(pageById("process-boundary").proof).toBe("rust/e2e/tests/test_chat_endpoint.rs");
  });

  it("built pages have a rust/ source and every proof is a rust/ path or Cargo.toml", () => {
    for (const entry of PAGES) {
      if (entry.status === "built") {
        expect(entry.source.startsWith("rust/")).toBe(true);
      }
      expect(entry.proof.startsWith("rust/") || entry.proof === "Cargo.toml").toBe(true);
    }
  });
});
