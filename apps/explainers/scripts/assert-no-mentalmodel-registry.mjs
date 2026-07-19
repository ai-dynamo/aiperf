#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Assert that `apps/explainers/src/core/deck-registry.ts` does not reach any
 * deck `MentalModel.tsx` via static relative imports (transitive module graph).
 *
 * Registry is packages-only: one `.flow` per deck compiles to DeckPackage under
 * `decks-generated/`; `packageToDeckDefinition` mounts `SceneRenderer`. MentalModel
 * `.tsx` files may remain on disk but must not be reachable from the registry
 * import graph. Always hard-fails on hits.
 *
 * Usage:
 *   node scripts/assert-no-mentalmodel-registry.mjs
 *   npm run assert:no-mentalmodel-registry
 *   make assert-no-mentalmodel-registry
 *   make assert-explainer-packages
 */

import { readFileSync, existsSync, statSync } from "node:fs";
import { dirname, join, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const REGISTRY = resolve(ROOT, "src/core/deck-registry.ts");
const EXTENSIONS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"];

const IMPORT_RE =
  /(?:import|export)\s+(?:type\s+)?(?:[^"'`;]+?\s+from\s+)?["']([^"']+)["']/g;

function isFile(path) {
  try {
    return existsSync(path) && statSync(path).isFile();
  } catch {
    return false;
  }
}

function resolveImport(fromFile, spec) {
  if (!spec.startsWith(".")) return null;

  const base = resolve(dirname(fromFile), spec);
  const candidates = [
    base,
    ...EXTENSIONS.map((ext) => base + ext),
    ...EXTENSIONS.map((ext) => join(base, `index${ext}`)),
  ];

  for (const candidate of candidates) {
    if (isFile(candidate)) return candidate;
  }
  return null;
}

function collectImports(source) {
  const specs = [];
  for (const match of source.matchAll(IMPORT_RE)) {
    specs.push(match[1]);
  }
  return specs;
}

function isMentalModelFile(filePath) {
  return /(^|[/\\])MentalModel\.tsx$/.test(filePath);
}

function walkRegistryGraph(entry) {
  const queue = [entry];
  const visited = new Set();
  const mentalModels = [];

  while (queue.length > 0) {
    const file = queue.shift();
    if (visited.has(file)) continue;
    visited.add(file);

    if (isMentalModelFile(file) && file !== entry) {
      mentalModels.push(file);
      continue;
    }

    let source;
    try {
      source = readFileSync(file, "utf8");
    } catch (err) {
      console.error(`Failed to read ${relative(ROOT, file)}: ${err.message}`);
      process.exit(1);
    }

    for (const spec of collectImports(source)) {
      const resolved = resolveImport(file, spec);
      if (resolved && !visited.has(resolved)) {
        queue.push(resolved);
      }
    }
  }

  return mentalModels;
}

if (!existsSync(REGISTRY)) {
  console.error(`Missing registry file: ${relative(ROOT, REGISTRY)}`);
  process.exit(1);
}

const hits = walkRegistryGraph(REGISTRY);
if (hits.length > 0) {
  console.error(
    [
      "deck-registry.ts still imports MentalModel.tsx via the registry path:",
      ...hits.sort().map((hit) => `  - ${relative(ROOT, hit)}`),
    ].join("\n"),
  );
  process.exit(1);
}

console.log(
  "OK: deck-registry.ts does not import any MentalModel.tsx on the registry path.",
);
