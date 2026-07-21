#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Compiles every `decks-flow/*.flow` source once, server-side, and writes:
 *
 *  - `src/core/deck-manifest.generated.json` — a tiny {id, route, hub, ...}
 *    manifest per deck, committed/regenerated so the Hub screen and route
 *    table never have to compile a single .flow body just to list decks.
 *  - `public/decks/<id>.json` — the full compiled DeckPackage per deck, a
 *    production-only artifact. The live app's `load-deck-flows.ts` fetches
 *    these in prod builds instead of compiling .flow in the browser; dev
 *    ignores them and compiles the live source lazily per route.
 *
 * Run via `npm run generate:manifest` (manifest only, fast, runs before dev)
 * or `npm run precompile:decks` (manifest + full packages, runs before build).
 * Pass `--full` to also emit the `public/decks/*.json` artifacts.
 *
 *   npx vite-node scripts/build-deck-artifacts.mjs [--full]
 */

import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compileExplainerSource } from "../src/flow/compiler/compile-explainer.js";
import { validateExplainerSet } from "../src/flow/compiler/validate-explainer-set.js";
import { formatDiagnostic } from "../src/flow/diagnostics.js";
import { FOUNDATION_CAPABILITIES, hasErrors } from "../src/flow/schema/index.js";

const FLOW_EXTENSION = ".flow";
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DECKS_DIR = join(ROOT, "decks-flow");
const MANIFEST_PATH = join(ROOT, "src/core/deck-manifest.generated.json");
const PACKAGES_DIR = join(ROOT, "public/decks");

const full = process.argv.includes("--full");

function printDiagnostics(diagnostics) {
  for (const entry of diagnostics) {
    console.error(formatDiagnostic(entry));
  }
}

async function main() {
  const entries = await readdir(DECKS_DIR, { withFileTypes: true });
  const flowNames = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(FLOW_EXTENSION))
    .map((entry) => entry.name)
    .sort();

  const packages = [];
  const sourcePaths = [];
  let failed = false;

  for (const flowName of flowNames) {
    const sourcePath = join(DECKS_DIR, flowName);
    const source = await readFile(sourcePath, "utf8");
    const compiled = compileExplainerSource({
      source,
      sourceName: sourcePath,
      capabilities: FOUNDATION_CAPABILITIES,
      strict: true,
      strictSdkAuthoring: true,
    });

    if (!compiled.ok || hasErrors(compiled.diagnostics)) {
      console.error(`FAILED ${flowName}`);
      printDiagnostics(compiled.diagnostics);
      failed = true;
      continue;
    }

    const expectedId = flowName.slice(0, -FLOW_EXTENSION.length);
    if (compiled.value.id !== expectedId) {
      console.error(
        `deck id mismatch at ${sourcePath}: filename id "${expectedId}" vs compiled id "${compiled.value.id}"`,
      );
      failed = true;
      continue;
    }

    packages.push(compiled.value);
    sourcePaths.push(sourcePath);
  }

  if (failed) {
    process.exitCode = 1;
    return;
  }

  const setResult = validateExplainerSet(packages, { sourcePaths });
  if (!setResult.ok || hasErrors(setResult.diagnostics)) {
    printDiagnostics(setResult.diagnostics);
    process.exitCode = 1;
    return;
  }

  const manifest = packages
    .map((pkg) => ({
      id: pkg.id,
      route: pkg.route,
      topic: pkg.topic,
      eyebrowLabel: pkg.eyebrowLabel,
      hub: pkg.hub,
      slideCount: pkg.slides.length,
    }))
    .sort((a, b) => a.id.localeCompare(b.id));

  await writeFile(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`wrote ${manifest.length} deck(s) to ${MANIFEST_PATH}`);

  if (full) {
    await mkdir(PACKAGES_DIR, { recursive: true });
    for (const pkg of packages) {
      await writeFile(join(PACKAGES_DIR, `${pkg.id}.json`), JSON.stringify(pkg));
    }
    console.log(`wrote ${packages.length} precompiled deck package(s) to ${PACKAGES_DIR}`);
  }
}

main().catch((error) => {
  console.error(String(error?.stack ?? error));
  process.exitCode = 1;
});
