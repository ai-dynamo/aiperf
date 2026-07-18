// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

function source(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

describe("immersive layout contract", () => {
  test("keeps causal metadata semantic without expanding the visual path", () => {
    const css = source("../../src/theme.css");

    expect(css).toContain(".aiperf-flow__causal-path-label");
    expect(css).toContain(".aiperf-flow__causal-beat-description");
    expect(css).toContain(".aiperf-flow__causal-path li");
    expect(css).toContain(".aiperf-flow__causal-beats");
    expect(css).not.toContain("display: contents");
  });

  test("keeps canvas bitmap sizing independent from viewport min-height", () => {
    const app = source("../../src/app.tsx");
    const css = source("../../src/theme.css");

    expect(app).not.toContain('className="aiperf-flow__canvas aiperf-flow__stage"');
    expect(app).toContain("aiperf-flow__canvas-host");
    expect(css).toContain(".aiperf-flow__canvas-host");
    expect(css).toMatch(/\.aiperf-flow__canvas[\s\S]*min-height:\s*0;/);
  });

  test("reserves the bottom HUD stack below side panels", () => {
    const css = source("../../src/theme.css");

    expect(css).toContain("--flow-hud-stack-clearance");
    expect(css).toMatch(
      /\.aiperf-flow__semantic-twin[\s\S]*var\(--flow-hud-stack-clearance\)/,
    );
    expect(css).toMatch(
      /@media \(width <= 600px\)[\s\S]*\.aiperf-flow__semantic-twin[\s\S]*position:\s*absolute;/,
    );
  });

  test("shows the transcript on request instead of over the transport", () => {
    expect(source("../../src/theme.css")).toContain(
      ".aiperf-flow__transcript:target",
    );
  });

  test("does not mount a duplicate narration highlight", () => {
    expect(source("../../src/app.tsx")).not.toContain(
      'className="aiperf-flow__narration-highlight"',
    );
  });

  test("keeps live announcements out of the layout grid", () => {
    expect(source("../../src/theme.css")).toContain(
      ".aiperf-flow__live-region",
    );
  });

  test("keeps the command constellation free of glass effects", () => {
    const command = source(
      "../../src/immersive/command-constellation.tsx",
    );

    expect(command).not.toContain("backdropFilter");
    expect(command).not.toContain("boxShadow");
  });
});
