/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, test } from "vitest";

import { checkCommand, formatCommand } from "../src/commands.js";

const MINIMAL_FLOW = `flow "Demo" as demo {
  language 1

  require core.rect "^1.0.0"

  scene "Scene" as scene {
    summary "A scene"

    rect box {
      x 0
      y 0
      width 10
      height 10
      fill "#000"
      label "Box"
      role "img"
      description "A box"
      fallback "Box"
    }

    narrate "Narration long enough for accessibility checks."
    reading-order box
    fallback "Fallback"
  }
}
`;

describe("formatCommand", () => {
  test("rewrites noncanonical source and --check fails until formatted", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aiperf-flow-cli-"));
    const path = join(dir, "demo.flow");
    await writeFile(path, MINIMAL_FLOW.replaceAll("\n  ", "\n"), "utf8");

    const checkBefore = await formatCommand({ paths: [path], check: true });
    expect(checkBefore.exitCode).toBe(1);

    const format = await formatCommand({ paths: [path], check: false });
    expect(format.exitCode).toBe(0);

    const checkAfter = await formatCommand({ paths: [path], check: true });
    expect(checkAfter.exitCode).toBe(0);

    const written = await readFile(path, "utf8");
    expect(written).toContain('flow "Demo" as demo {');
  });
});

describe("checkCommand", () => {
  test("compiles a valid foundation source", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aiperf-flow-cli-"));
    const path = join(dir, "demo.flow");
    await writeFile(path, MINIMAL_FLOW, "utf8");

    const result = await checkCommand({
      paths: [path],
      strict: false,
      json: false,
    });

    expect(result.exitCode, result.stderr).toBe(0);
    expect(result.stdout).toBe("ok\n");
  });

  test("emits JSON diagnostics for invalid source", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aiperf-flow-cli-"));
    const path = join(dir, "bad.flow");
    await writeFile(path, "not a flow document\n", "utf8");

    const result = await checkCommand({
      paths: [path],
      strict: false,
      json: true,
    });

    expect(result.exitCode).toBe(1);
    const diagnostics = JSON.parse(result.stdout) as unknown[];
    expect(diagnostics.length).toBeGreaterThan(0);
  });
});
