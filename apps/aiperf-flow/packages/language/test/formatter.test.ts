/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import { formatDocument, parseDocument } from "../src/index.js";

function withoutSourceMaps(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(withoutSourceMaps);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => key !== "sourceMap")
        .map(([key, entry]) => [key, withoutSourceMaps(entry)]),
    );
  }
  return value;
}

describe("formatDocument", () => {
  test("canonicalizes valid source and is semantically idempotent", () => {
    const messySource = `flow "A \\"quoted\\" flow" as demo{language 1
require core.connector "^1.0.0" token accent="#fff"
scene "Scene" as scene{/* retained positions after comments */
summary "Summary" rect b{fallback "B" description "Bee" role "img" label "B" fill token(accent) height 20 width 30 y 2 x 1}
rect a{x 3 y 4 width 50 height 60 fill "#000" label "A" role "img" description "Aye" fallback "A"}
connector edge{fallback "A to B" stroke token(accent) label "edge" to b from a}
camera main{at 0 frame a,b zoom 1} timeline primary{at 0 trace edge duration 10}
interaction inspect{on select a do inspect a}
responsive compact when width<720{set b.y=30 set b.x=10}
narrate "Narration" reading-order a,b,edge fallback "Fallback"}}
`;
    const parsed = parseDocument(messySource, "messy.flow");
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const formatted = formatDocument(parsed.value);
    expect(formatted).toBe(`flow "A \\"quoted\\" flow" as demo {
  language 1

  require core.connector "^1.0.0"

  token accent = "#fff"

  scene "Scene" as scene {
    summary "Summary"

    rect b {
      x 1
      y 2
      width 30
      height 20
      fill token(accent)
      label "B"
      role "img"
      description "Bee"
      fallback "B"
    }

    rect a {
      x 3
      y 4
      width 50
      height 60
      fill "#000"
      label "A"
      role "img"
      description "Aye"
      fallback "A"
    }

    connector edge {
      from a
      to b
      label "edge"
      stroke token(accent)
      fallback "A to B"
    }

    camera main {
      at 0 frame a,b zoom 1
    }

    timeline primary {
      at 0 trace edge duration 10
    }

    interaction inspect {
      on select a
      do inspect a
    }

    responsive compact when width < 720 {
      set b.y = 30
      set b.x = 10
    }

    narrate "Narration"
    reading-order a,b,edge
    fallback "Fallback"
  }
}
`);

    const reparsed = parseDocument(formatted, "formatted.flow");
    expect(reparsed.ok, JSON.stringify(reparsed.diagnostics)).toBe(true);
    if (!reparsed.ok) {
      return;
    }

    expect(withoutSourceMaps(reparsed.value)).toEqual(
      withoutSourceMaps(parsed.value),
    );
    expect(formatDocument(reparsed.value)).toBe(formatted);
  });
});
