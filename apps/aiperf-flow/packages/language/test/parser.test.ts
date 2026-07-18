/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import { parseDocument } from "../src/index.js";

export const FOUNDATION_SOURCE = `flow "Request flow" as request-flow {
  language 1
  require core.rect "^1.0.0"
  require core.text "^1.0.0"
  require core.connector "^1.0.0"
  token accent = "#7aa2f7"

  scene "Execution boundary" as execution {
    summary "The CLI starts a runtime that dispatches work."

    rect cli {
      x 40
      y 100
      width 160
      height 72
      fill token(accent)
      label "CLI"
      role "img"
      description "Command-line process"
      fallback "CLI"
    }

    rect runtime {
      x 300
      y 100
      width 180
      height 72
      fill "#244a35"
      label "Runtime"
      role "img"
      description "Execution runtime"
      fallback "Runtime"
    }

    connector spawn {
      from cli
      to runtime
      label "spawn --execute"
      stroke token(accent)
      fallback "CLI starts Runtime"
    }

    camera main {
      at 0 frame cli,runtime zoom 1
      at 2000 frame runtime zoom 1.4
    }

    timeline primary {
      at 0 reveal cli duration 400
      at 800 trace spawn duration 1200
      at 2200 reveal runtime duration 400
    }

    interaction inspect-runtime {
      on select runtime
      do inspect runtime
    }

    responsive compact when width < 720 {
      set runtime.x = 40
      set runtime.y = 240
    }

    narrate "The CLI starts a fresh runtime and dispatches work."
    reading-order cli,runtime,spawn
    fallback "CLI starts Runtime."
  }
}
`;

describe("parseDocument", () => {
  test("parses the complete foundation grammar with source ranges", () => {
    const result = parseDocument(FOUNDATION_SOURCE, "request-flow.flow");

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const document = result.value;
    expect(document.id).toBe("request-flow");
    expect(document.tokens[0]?.value).toMatchObject({
      kind: "literal",
      value: "#7aa2f7",
    });
    expect(document.sourceMap).toEqual({
      source: "request-flow.flow",
      start: { offset: 0, line: 1, column: 1 },
      end: {
        offset: FOUNDATION_SOURCE.length,
        line: FOUNDATION_SOURCE.split("\n").length,
        column: 1,
      },
    });

    const scene = document.scenes[0];
    expect(scene?.id).toBe("execution");
    expect(scene?.renderDeclarations.map(({ kind, id }) => [kind, id])).toEqual([
      ["rect", "cli"],
      ["rect", "runtime"],
      ["connector", "spawn"],
    ]);
    expect(scene?.cameras[0]?.keyframes).toEqual([
      expect.objectContaining({
        time: 0,
        targets: expect.objectContaining({
          kind: "reference-list",
          references: ["cli", "runtime"],
        }),
        zoom: 1,
      }),
      expect.objectContaining({
        time: 2000,
        targets: expect.objectContaining({
          kind: "reference-list",
          references: ["runtime"],
        }),
        zoom: 1.4,
      }),
    ]);
    expect(scene?.timelines[0]?.cues).toEqual([
      expect.objectContaining({
        time: 0,
        action: "reveal",
        target: "cli",
        duration: 400,
      }),
      expect.objectContaining({
        time: 800,
        action: "trace",
        target: "spawn",
        duration: 1200,
      }),
      expect.objectContaining({
        time: 2200,
        action: "reveal",
        target: "runtime",
        duration: 400,
      }),
    ]);
    expect(scene?.interactions).toEqual([
      expect.objectContaining({
        id: "inspect-runtime",
        event: expect.objectContaining({ name: "select", target: "runtime" }),
        action: expect.objectContaining({ name: "inspect", target: "runtime" }),
      }),
    ]);
    expect(scene?.responsiveVariants[0]).toEqual(
      expect.objectContaining({
        id: "compact",
        condition: expect.objectContaining({
          property: "width",
          operator: "<",
          value: 720,
        }),
        overrides: [
          expect.objectContaining({ target: "runtime", property: "x", value: 40 }),
          expect.objectContaining({ target: "runtime", property: "y", value: 240 }),
        ],
      }),
    );
    expect(scene?.narration?.text).toBe(
      "The CLI starts a fresh runtime and dispatches work.",
    );
    expect(scene?.readingOrder?.references).toEqual(["cli", "runtime", "spawn"]);
    expect(scene?.sourceMap.start).toEqual({
      offset: FOUNDATION_SOURCE.indexOf('scene "Execution boundary"'),
      line: 8,
      column: 3,
    });
    expect(scene?.sourceMap.end.line).toBe(67);
  });

  test("returns recovered parser diagnostics with source locations", () => {
    const result = parseDocument(
      `flow "Broken" as broken {
  language nope
  require
  scene "Missing" as missing {
    fallback "still recover"
  }
}
`,
      "broken.flow",
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics.length).toBeGreaterThanOrEqual(2);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "PARSE_UNEXPECTED_TOKEN",
          range: expect.objectContaining({
            source: "broken.flow",
            start: expect.objectContaining({ line: 2 }),
          }),
        }),
        expect.objectContaining({
          code: "PARSE_UNEXPECTED_TOKEN",
          range: expect.objectContaining({ source: "broken.flow" }),
        }),
      ]),
    );
  });

  test("reports invalid characters from the lexer", () => {
    const result = parseDocument(
      'flow "Broken" as broken { language 1 ~ }',
      "invalid.flow",
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "LEX_INVALID_CHARACTER",
        range: expect.objectContaining({ source: "invalid.flow" }),
      }),
    );
  });
});
