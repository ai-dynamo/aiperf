/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { compileExplainerSource } from "../flow/compiler/compile-explainer.js";
import { validateExplainerSet } from "../flow/compiler/validate-explainer-set.js";
import { formatDiagnostic } from "../flow/diagnostics.js";
import {
  FOUNDATION_CAPABILITIES,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
} from "../flow/schema/index.js";
import { packageToDeckDefinition } from "./package-adapter";
import type { DeckDefinition } from "./types";

/**
 * Lazy chunk loaders — one per `decks-flow/*.flow` file. Vite code-splits
 * each into its own chunk; nothing here is fetched or compiled until a
 * specific deck is requested by id (see `loadDeckPackageById`), so visiting
 * one deck's route never pulls in or compiles the other fifteen.
 */
const flowSources = import.meta.glob("../../decks-flow/*.flow", {
  query: "?raw",
  import: "default",
}) as Record<string, () => Promise<string>>;

const FLOW_PATH_RE = /(?:^|\/)([^/]+)\.flow$/;

function compilationError(
  path: string,
  diagnostics: readonly Diagnostic[],
): Error {
  const details = diagnostics.map(formatDiagnostic).join("\n");
  return new Error(
    `Failed to compile live Flow source "${path}":${details ? `\n${details}` : " compiler returned no diagnostics"}`,
  );
}

function flowIdFromPath(path: string): string {
  const id = FLOW_PATH_RE.exec(path)?.[1];
  if (id === undefined) {
    throw new Error(
      `Invalid live Flow source path "${path}"; expected a filename ending in "<deck-id>.flow"`,
    );
  }
  return id;
}

function pathForId(id: string): [path: string, loader: () => Promise<string>] {
  const entry = Object.entries(flowSources).find(
    ([path]) => flowIdFromPath(path) === id,
  );
  if (entry === undefined) {
    throw new Error(
      `Missing live Flow source for "${id}" under decks-flow ` +
        `(expected ${id}.flow compiled via compileExplainerSource)`,
    );
  }
  return entry;
}

function compileOne(path: string, source: string): DeckPackage {
  const result = compileExplainerSource({
    source,
    sourceName: path,
    capabilities: FOUNDATION_CAPABILITIES,
    strict: true,
    strictSdkAuthoring: true,
  });
  if (!result.ok || hasErrors(result.diagnostics)) {
    throw compilationError(path, result.diagnostics);
  }
  const filenameId = flowIdFromPath(path);
  if (result.value.id !== filenameId) {
    throw new Error(
      `Live Flow id mismatch at "${path}": filename id "${filenameId}" vs compiled id "${result.value.id}"`,
    );
  }
  return result.value;
}

/** Precompiled-package fetch, prod-only (see `scripts/build-deck-artifacts.mjs --full`). */
async function fetchPrecompiledPackage(id: string): Promise<DeckPackage | undefined> {
  if (!import.meta.env.PROD) {
    return undefined;
  }
  try {
    const base = import.meta.env.BASE_URL;
    const response = await fetch(`${base}decks/${id}.json`);
    if (!response.ok) {
      return undefined;
    }
    return (await response.json()) as DeckPackage;
  } catch {
    return undefined;
  }
}

const packageCache = new Map<string, Promise<DeckPackage>>();

/**
 * Resolve one deck's compiled package, compiling (or fetching a precompiled
 * artifact for) only that deck. This is the hot path every deck route uses:
 * navigating to `/steppable-replay-engine` never touches the other decks.
 */
export function loadDeckPackageById(id: string): Promise<DeckPackage> {
  const cached = packageCache.get(id);
  if (cached !== undefined) {
    return cached;
  }
  const promise = (async () => {
    const precompiled = await fetchPrecompiledPackage(id);
    if (precompiled !== undefined) {
      return precompiled;
    }
    const [path, loader] = pathForId(id);
    const source = await loader();
    return compileOne(path, source);
  })();
  packageCache.set(id, promise);
  return promise;
}

/** Resolve one deck from its live Flow source, adapted for `ExplainerShell`. */
export async function loadDeckFlowById(id: string): Promise<DeckDefinition> {
  const pkg = await loadDeckPackageById(id);
  return packageToDeckDefinition(pkg);
}

let cachedAllPackages: readonly DeckPackage[] | undefined;

/**
 * Compiles and cross-validates every live Flow package. Only used by the
 * opt-in dev-diagnostics pass (`main.tsx`, dynamically imported, non-blocking)
 * — never on the app's route-rendering hot path.
 */
export async function loadDeckPackages(): Promise<readonly DeckPackage[]> {
  if (cachedAllPackages !== undefined) {
    return cachedAllPackages;
  }

  const packages: DeckPackage[] = [];
  const sourcePaths: string[] = [];

  for (const path of Object.keys(flowSources).sort()) {
    const source = await flowSources[path]!();
    packages.push(compileOne(path, source));
    sourcePaths.push(path);
  }

  const validated = validateExplainerSet(packages, { sourcePaths });
  if (!validated.ok || hasErrors(validated.diagnostics)) {
    throw compilationError("<live Flow package set>", validated.diagnostics);
  }

  cachedAllPackages = validated.value;
  return cachedAllPackages;
}
