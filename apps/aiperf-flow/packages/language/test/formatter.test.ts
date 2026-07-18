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

  test("round-trips themes in canonical document order", () => {
    const source = `flow "Lab" as lab {
  language 1
  use theme lab_chalk
  scene "Main" as main {
    summary "Lab scene"
    rect router {
      x 0
      y 0
      width 10
      height 10
      fill theme(surface.raised)
      label "Router"
      role "group"
      description "Router"
      fallback "Router"
    }
    reading-order router
    fallback "Lab fallback"
  }
  theme lab_chalk extends systems_chalk {
    color accent.control = "#78dce8"
    duration motion.draw = 420ms
    font font.body = ["Nunito Sans", "Segoe UI", "sans-serif"]
  }
  require core.rect "^1.0.0"
}
`;
    const parsed = parseDocument(source, "theme.flow");
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const formatted = formatDocument(parsed.value);
    expect(formatted).toBe(`flow "Lab" as lab {
  language 1

  require core.rect "^1.0.0"

  theme lab_chalk extends systems_chalk {
    color accent.control = "#78dce8"
    duration motion.draw = 420ms
    font font.body = ["Nunito Sans", "Segoe UI", "sans-serif"]
  }

  use theme lab_chalk

  scene "Main" as main {
    summary "Lab scene"

    rect router {
      x 0
      y 0
      width 10
      height 10
      fill theme(surface.raised)
      label "Router"
      role "group"
      description "Router"
      fallback "Router"
    }

    reading-order router
    fallback "Lab fallback"
  }
}
`);

    const reparsed = parseDocument(formatted, "formatted-theme.flow");
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
