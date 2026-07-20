/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Compiles every `apps/explainers/decks-flow/*.flow` source in-memory with
 * the exact same strict `compileExplainerSource` + `validateExplainerSet`
 * pipeline the live app uses (`src/core/load-deck-flows.ts`), and prints the
 * resulting `DeckPackage[]` as JSON on stdout. No artifacts are written to
 * disk — there is no generated-package build step; `.flow` is the only
 * source of truth. Runs under `vite-node`:
 *
 *   npx vite-node scripts/compile-decks.ts
 */

import { readdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compileExplainerSource } from "../src/flow/compiler/compile-explainer.js";
import { validateExplainerSet } from "../src/flow/compiler/validate-explainer-set.js";
import { formatDiagnostic } from "../src/flow/diagnostics.js";
import { resolveScene } from "../src/core/diagram/resolution/resolve-scene.js";
import { resolvedSceneSnapshot } from "../src/core/diagram/resolution/serialize.js";
import type { ResolvedSceneSnapshot } from "../src/core/diagram/resolution/types.js";
import type { DeckPackage } from "../src/flow/schema/deck-package.js";
import {
  FOUNDATION_CAPABILITIES,
  hasErrors,
  type Diagnostic,
} from "../src/flow/schema/index.js";

const FLOW_EXTENSION = ".flow";

const __dirname = dirname(fileURLToPath(import.meta.url));
const explainersRoot = resolve(__dirname, "..");
const decksFlowDir = join(explainersRoot, "decks-flow");

type VerifierBundle = Readonly<{
  packages: readonly DeckPackage[];
  resolvedScenes: readonly Readonly<{
    deckId: string;
    slideId: string;
    snapshot: ResolvedSceneSnapshot;
  }>[];
}>;

function printDiagnostics(diagnostics: readonly Diagnostic[]): void {
  for (const entry of diagnostics) {
    console.error(formatDiagnostic(entry));
  }
}

async function main(): Promise<void> {
  const entries = await readdir(decksFlowDir, { withFileTypes: true });
  const flowNames = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(FLOW_EXTENSION))
    .map((entry) => entry.name)
    .sort();

  const packages = [];
  const sourcePaths: string[] = [];
  let failed = false;

  for (const flowName of flowNames) {
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

  const resolvedScenes = packages.flatMap((packageValue) => {
    const slides = packageValue.slides.flatMap((slide) => {
      const scene = slide.render?.scene;
      if (scene === undefined) return [];
      return [
        {
          deckId: packageValue.id,
          slideId: slide.id,
          snapshot: resolvedSceneSnapshot(resolveScene(scene)),
        },
      ];
    });
    const finalScene = packageValue.finalCard?.scene;
    return finalScene === undefined
      ? slides
      : [
          ...slides,
          {
            deckId: packageValue.id,
            slideId: "__final-card",
            snapshot: resolvedSceneSnapshot(resolveScene(finalScene)),
          },
        ];
  });
  const bundle: VerifierBundle = { packages, resolvedScenes };
  process.stdout.write(JSON.stringify(bundle));
}

main().catch((error) => {
  console.error(String(error?.stack ?? error));
  process.exitCode = 1;
});
