/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { compileSource, packFlow } from "@aiperf/flow-compiler";
import { FOUNDATION_CAPABILITIES } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { buildCommand } from "../src/commands.js";

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

async function createSource(): Promise<{
  directory: string;
  sourcePath: string;
  outDir: string;
}> {
  const directory = await mkdtemp(join(tmpdir(), "aiperf-flow-build-"));
  const sourcePath = join(directory, "demo.flow");
  await writeFile(sourcePath, MINIMAL_FLOW, "utf8");
  return { directory, sourcePath, outDir: join(directory, "site") };
}

describe("buildCommand", () => {
  test("writes every packed file byte-for-byte", async () => {
    const { sourcePath, outDir } = await createSource();

    const result = await buildCommand({
      path: sourcePath,
      outDir,
      strict: false,
      clean: false,
    });

    expect(result.exitCode, result.stderr).toBe(0);
    const compiled = compileSource({
      source: MINIMAL_FLOW,
      sourceName: sourcePath,
      capabilities: FOUNDATION_CAPABILITIES,
      strict: false,
    });
    if (!compiled.ok) {
      throw new Error("test fixture did not compile");
    }
    const packed = packFlow(compiled.value, sourcePath);
    expect(result.stdout).toContain(`Built ${packed.files.length} file(s)`);
    expect(result.stdout).toContain(packed.manifest.id);
    expect(result.stdout).toContain(packed.manifest.contentHash);
    for (const file of packed.files) {
      expect(await readFile(join(outDir, file.path))).toEqual(
        Buffer.from(file.content),
      );
    }
  });

  test("leaves no output when compilation fails", async () => {
    const { sourcePath, outDir } = await createSource();
    await writeFile(sourcePath, "not a flow document\n", "utf8");

    const result = await buildCommand({
      path: sourcePath,
      outDir,
      strict: false,
      clean: false,
    });

    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain(sourcePath);
    await expect(lstat(outDir)).rejects.toMatchObject({ code: "ENOENT" });
  });

  test("rejects a nonempty output directory without --clean", async () => {
    const { sourcePath, outDir } = await createSource();
    await mkdir(outDir);
    await writeFile(join(outDir, "occupied"), "occupied", "utf8");

    const result = await buildCommand({
      path: sourcePath,
      outDir,
      strict: false,
      clean: false,
    });

    expect(result.exitCode).toBe(2);
    expect(result.stderr).toContain("--clean");
    expect(await readFile(join(outDir, "occupied"), "utf8")).toBe("occupied");
  });

  test("--clean replaces an existing output tree deterministically", async () => {
    const { sourcePath, outDir } = await createSource();
    const first = await buildCommand({
      path: sourcePath,
      outDir,
      strict: false,
      clean: false,
    });
    expect(first.exitCode, first.stderr).toBe(0);
    const firstManifest = await readFile(join(outDir, "flow.manifest.json"));
    await writeFile(join(outDir, "obsolete.txt"), "obsolete", "utf8");

    const second = await buildCommand({
      path: sourcePath,
      outDir,
      strict: false,
      clean: true,
    });

    expect(second.exitCode, second.stderr).toBe(0);
    expect(await readdir(outDir)).not.toContain("obsolete.txt");
    expect(await readFile(join(outDir, "flow.manifest.json"))).toEqual(
      firstManifest,
    );
  });

  test("--clean rejects a symlink output without touching its target", async () => {
    const { directory, sourcePath, outDir } = await createSource();
    const target = join(directory, "target");
    await writeFile(target, "protected", "utf8");
    await symlink(target, outDir);

    const result = await buildCommand({
      path: sourcePath,
      outDir,
      strict: false,
      clean: true,
    });

    expect(result.exitCode).toBe(2);
    expect(result.stderr).toContain("symbolic link");
    expect(await readFile(target, "utf8")).toBe("protected");
  });
});
