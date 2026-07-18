#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Compile `apps/explainers/decks-flow/*.flow` into DeckPackage JSON artifacts
 * under `apps/explainers/src/decks-generated/*.package.json` via the real
 * `@aiperf/flow-compiler` API (`compileExplainerSource` + `writeDeckPackage`).
 *
 * Requires workspace packages to be built first (`npm run flow:build`).
 *
 * Run from `apps/aiperf-flow`:
 *   node scripts/build-explainer-packages.mjs
 *   npm run build:explainer-packages
 */

import { mkdir, readdir, readFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as compiler from "@aiperf/flow-compiler";
import {
  FOUNDATION_CAPABILITIES,
  hasErrors,
} from "@aiperf/flow-schema";

const { compileExplainerSource, writeDeckPackage } = compiler;

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "../../..");
const decksFlowDir = join(repoRoot, "apps/explainers/decks-flow");
const outDir = join(repoRoot, "apps/explainers/src/decks-generated");

function formatDiagnostic(diagnostic) {
  const { range } = diagnostic;
  const location = `${range.source}:${range.start.line}:${range.start.column}`;
  const repair =
    diagnostic.repair === undefined ? "" : ` (${diagnostic.repair})`;
  return `${location}: ${diagnostic.severity} ${diagnostic.code}: ${diagnostic.message}${repair}`;
}

function printDiagnostics(diagnostics) {
  for (const diagnostic of diagnostics) {
    console.error(formatDiagnostic(diagnostic));
  }
}

async function listFlowSources(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".flow"))
    .map((entry) => entry.name)
    .sort();
}

async function compileOne(flowName) {
  if (typeof compileExplainerSource !== "function") {
    throw new Error(
      "@aiperf/flow-compiler must export compileExplainerSource (build packages after that API lands)",
    );
  }
  if (typeof writeDeckPackage !== "function") {
    throw new Error(
      "@aiperf/flow-compiler must export writeDeckPackage (build packages after that API lands)",
    );
  }

  const sourcePath = join(decksFlowDir, flowName);
  const source = await readFile(sourcePath, "utf8");
  const compiled = compileExplainerSource({
    source,
    sourceName: sourcePath,
    capabilities: FOUNDATION_CAPABILITIES,
    strict: true,
  });

  if (!compiled.ok || hasErrors(compiled.diagnostics)) {
    printDiagnostics(compiled.diagnostics);
    throw new Error(`failed to compile ${sourcePath}`);
  }

  if (compiled.diagnostics.length > 0) {
    printDiagnostics(compiled.diagnostics);
  }

  const pkg = compiled.value;
  const outPath = join(outDir, `${pkg.id}.package.json`);
  await writeDeckPackage(outPath, pkg);
  console.log(`Wrote ${outPath} (from ${basename(sourcePath)})`);
  return pkg;
}

async function main() {
  await mkdir(decksFlowDir, { recursive: true });
  await mkdir(outDir, { recursive: true });

  const flowNames = await listFlowSources(decksFlowDir);
  if (flowNames.length === 0) {
    console.log(`No .flow sources in ${decksFlowDir}; nothing to compile.`);
    return;
  }

  const packages = [];
  const failures = [];
  for (const flowName of flowNames) {
    try {
      packages.push(await compileOne(flowName));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push({ flowName, message });
      console.error(`FAILED ${flowName}: ${message}`);
    }
  }

  if (packages.length > 0 && typeof compiler.validateExplainerSet === "function") {
    const setResult = compiler.validateExplainerSet(packages);
    if (!setResult.ok || hasErrors(setResult.diagnostics)) {
      printDiagnostics(setResult.diagnostics);
      throw new Error("explainer set validation failed (duplicate id/route)");
    }
  }

  console.log(
    `Compiled ${packages.length}/${flowNames.length} explainer package(s) into ${outDir}`,
  );
  if (failures.length > 0) {
    console.error(
      `build-explainer-packages: ${failures.length} deck(s) failed:\n` +
        failures.map((f) => `  - ${f.flowName}: ${f.message}`).join("\n"),
    );
    process.exitCode = 1;
  }
}

const isDirectRun =
  process.argv[1] !== undefined &&
  pathToFileURL(resolve(process.argv[1])).href === import.meta.url;

if (isDirectRun) {
  main().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`build-explainer-packages: ${message}`);
    process.exitCode = 1;
  });
}
