// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { formatDocument } from "../src/formatter.js";
import { parseDocument } from "../src/parser.js";

const source = `
flow "Lab" as lab {
  language 1

  theme lab_chalk extends systems_chalk {
    color accent.control = "#78dce8"
    color accent.execution = "#ffd866"
    number stroke.standard = 2
    duration motion.draw = 420ms
    font font.body = ["Nunito Sans", "Segoe UI", "sans-serif"]
    enum stroke.cap = "round"
  }

  use theme lab_chalk

  require core.rect "^1.0.0"

  scene "Main" as main {
    summary "s"
    rect router {
      x 0
      y 0
      width 10
      height 10
      fill theme(surface.raised)
      stroke theme(accent.control)
      label "Router"
      role "group"
      description "router"
      fallback "router"
    }
    reading-order router
    fallback "f"
  }
}
`;

describe("theme grammar", () => {
  test("parses theme declaration, use theme, and theme(role) style refs", () => {
    const result = parseDocument(source, "lab.flow");
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) return;
    const doc = result.value;
    expect(doc.themes).toHaveLength(1);
    expect(doc.themes[0]?.id).toBe("lab_chalk");
    expect(doc.themes[0]?.extends).toBe("systems_chalk");
    expect(doc.themes[0]?.assignments.map((a) => a.role)).toEqual([
      "accent.control",
      "accent.execution",
      "stroke.standard",
      "motion.draw",
      "font.body",
      "stroke.cap",
    ]);
    expect(doc.useTheme?.themeId).toBe("lab_chalk");
    const rect = doc.scenes[0]?.renderDeclarations[0];
    expect(rect?.kind).toBe("rect");
    if (rect?.kind !== "rect") return;
    expect(rect.fill).toEqual(
      expect.objectContaining({
        kind: "theme-role-reference",
        role: "surface.raised",
      }),
    );
    expect(rect.stroke).toEqual(
      expect.objectContaining({
        kind: "theme-role-reference",
        role: "accent.control",
      }),
    );
  });

  test("rejects theme(role) with empty role", () => {
    const result = parseDocument(
      `flow "X" as x { language 1
        scene "S" as s {
          summary "s"
          rect a { x 0 y 0 width 1 height 1 fill theme() label "a" role "g" description "a" fallback "a" }
          reading-order a
          fallback "f"
        }
      }`,
      "bad.flow",
    );
    expect(result.ok).toBe(false);
  });

  test("formats typed theme syntax", () => {
    const result = parseDocument(source, "lab.flow");
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) return;

    const formatted = formatDocument(result.value);

    expect(formatted).toContain(
      `theme lab_chalk extends systems_chalk {
    color accent.control = "#78dce8"
    color accent.execution = "#ffd866"
    number stroke.standard = 2
    duration motion.draw = 420ms
    font font.body = ["Nunito Sans", "Segoe UI", "sans-serif"]
    enum stroke.cap = "round"
  }`,
    );
    expect(formatted).toContain("use theme lab_chalk");
    expect(formatted).toContain("fill theme(surface.raised)");
    expect(formatted).toContain("stroke theme(accent.control)");
  });
});
