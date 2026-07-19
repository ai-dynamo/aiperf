/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Compile `apps/explainers/decks-flow/*.flow` into DeckPackage JSON artifacts
 * under `apps/explainers/src/decks-generated/*.package.json` using ONLY the
 * local, browser-safe Flow toolchain under `apps/explainers/src/flow`.
 *
 * This is the build-time mirror of the runtime compile path in
 * `apps/explainers/src/core/load-deck-flows.ts`: the same capability
 * manifest (`FOUNDATION_CAPABILITIES`), the same strict
 * `compileExplainerSource` + `validateExplainerSet` pipeline, and the same
 * deterministic serialization (`packDeckPackageToJson`). Stale
 * `*.package.json` files whose id no longer has a matching `.flow` source
 * are removed.
 *
 * Self-contained: imports no external Flow workspace package.
 *
 * Runs under `vite-node` (a `vitest` dependency already available under
 * `apps/explainers/node_modules/.bin`), which transpiles this TypeScript
 * module on the fly with the same resolution Vite uses for `.test.ts` files
 * that import `../src/flow/**`. Run from `apps/explainers`:
 *
 *   npx vite-node scripts/build-explainer-packages.ts
 *   npm run build:explainer-packages
 */

import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compileExplainerSource } from "../src/flow/compiler/compile-explainer.js";
import { packDeckPackageToJson } from "../src/flow/compiler/serialization.js";
import { validateExplainerSet } from "../src/flow/compiler/validate-explainer-set.js";
import { formatDiagnostic } from "../src/flow/diagnostics.js";
import {
  FOUNDATION_CAPABILITIES,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
} from "../src/flow/schema/index.js";

const FLOW_EXTENSION = ".flow";
const PACKAGE_EXTENSION = ".package.json";

const __dirname = dirname(fileURLToPath(import.meta.url));
const explainersRoot = resolve(__dirname, "..");
const decksFlowDir = join(explainersRoot, "decks-flow");
const outDir = join(explainersRoot, "src/decks-generated");

type CompiledDeck = Readonly<{
  flowName: string;
  sourcePath: string;
  pkg: DeckPackage;
}>;

type CompileFailure = Readonly<{ flowName: string; message: string }>;

function printDiagnostics(diagnostics: readonly Diagnostic[]): void {
  for (const entry of diagnostics) {
    console.error(formatDiagnostic(entry));
  }
}

function flowIdFromName(flowName: string): string {
  return flowName.slice(0, -FLOW_EXTENSION.length);
}

async function listFlowSources(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(FLOW_EXTENSION))
    .map((entry) => entry.name)
    .sort();
}

/**
 * Compiles one `.flow` source with the strict foundation-capability policy.
 * Throws (with diagnostics already printed) on compile errors or on an
 * id/filename mismatch, so a misnamed deck fails loudly instead of silently
 * writing under a name that the registry glob would never resolve.
 */
async function compileOne(flowName: string): Promise<CompiledDeck> {
  const sourcePath = join(decksFlowDir, flowName);
  const source = await readFile(sourcePath, "utf8");
  const compiled = compileExplainerSource({
    source,
    sourceName: sourcePath,
    capabilities: FOUNDATION_CAPABILITIES,
    strict: true,
    strictSdkAuthoring: true,
  });

  if (!compiled.ok || hasErrors(compiled.diagnostics)) {
    printDiagnostics(compiled.diagnostics);
    throw new Error(`failed to compile ${sourcePath}`);
  }
  if (compiled.diagnostics.length > 0) {
    printDiagnostics(compiled.diagnostics);
  }

  const expectedId = flowIdFromName(flowName);
  if (compiled.value.id !== expectedId) {
    throw new Error(
      `deck id mismatch at ${sourcePath}: filename id "${expectedId}" vs compiled id "${compiled.value.id}"`,
    );
  }

  return { flowName, sourcePath, pkg: compiled.value };
}

/**
 * Deletes `*.package.json` files under `outDir` whose id has no matching
 * `.flow` source in `sourceIds`. Scoped to source removal/rename only: a
 * deck that exists on disk but failed to compile this run keeps its
 * previously generated package rather than losing it to a transient error.
 */
async function removeStalePackages(sourceIds: ReadonlySet<string>): Promise<void> {
  const entries = await readdir(outDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(PACKAGE_EXTENSION)) {
      continue;
    }
    const id = entry.name.slice(0, -PACKAGE_EXTENSION.length);
    if (sourceIds.has(id)) {
      continue;
    }
    const stalePath = join(outDir, entry.name);
    await rm(stalePath);
    console.log(`Removed stale ${stalePath}`);
  }
}

async function main(): Promise<void> {
  await mkdir(decksFlowDir, { recursive: true });
  await mkdir(outDir, { recursive: true });

  const flowNames = await listFlowSources(decksFlowDir);
  if (flowNames.length === 0) {
    console.log(`No .flow sources in ${decksFlowDir}; nothing to compile.`);
    return;
  }

  const compiledDecks: CompiledDeck[] = [];
  const failures: CompileFailure[] = [];
  for (const flowName of flowNames) {
    try {
      compiledDecks.push(await compileOne(flowName));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push({ flowName, message });
      console.error(`FAILED ${flowName}: ${message}`);
    }
  }

  if (compiledDecks.length > 0) {
    const setResult = validateExplainerSet(
      compiledDecks.map((deck) => deck.pkg),
      { sourcePaths: compiledDecks.map((deck) => deck.sourcePath) },
    );
    if (!setResult.ok || hasErrors(setResult.diagnostics)) {
      printDiagnostics(setResult.diagnostics);
      throw new Error("explainer set validation failed (duplicate id/route)");
    }

    for (const deck of compiledDecks) {
      const outPath = join(outDir, `${deck.pkg.id}${PACKAGE_EXTENSION}`);
      await writeFile(outPath, packDeckPackageToJson(deck.pkg), "utf8");
      console.log(`Wrote ${outPath} (from ${deck.flowName})`);
    }
  }

  await removeStalePackages(new Set(flowNames.map(flowIdFromName)));

  console.log(
    `Compiled ${compiledDecks.length}/${flowNames.length} explainer package(s) into ${outDir}`,
  );
  if (failures.length > 0) {
    console.error(
      `build-explainer-packages: ${failures.length} deck(s) failed:\n` +
        failures.map((f) => `  - ${f.flowName}: ${f.message}`).join("\n"),
    );
    process.exitCode = 1;
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`build-explainer-packages: ${message}`);
  process.exitCode = 1;
});
