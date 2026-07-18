// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");
}

describe("preview layout ownership", () => {
  test("the preview styles only its host chrome", () => {
    expect(source("./styles.css")).not.toContain(".aiperf-flow__");
  });

  test("the preview loads runtime styles before host chrome overrides", () => {
    const entrypoint = source("./main.tsx");
    const runtimeTheme = entrypoint.indexOf(
      '../packages/runtime/src/theme.css',
    );
    const previewTheme = entrypoint.indexOf("./styles.css");

    expect(runtimeTheme).toBeGreaterThanOrEqual(0);
    expect(previewTheme).toBeGreaterThan(runtimeTheme);
  });

  test("the preview does not mount duplicate runtime controls", () => {
    expect(source("./App.tsx")).not.toContain("preview-canvas-tools");
  });

  test("the preview does not reimplement narrative cue projection", () => {
    expect(source("./App.tsx")).not.toContain("function sceneNarrativeCues");
  });
});
