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

const flowSources = import.meta.glob("../../decks-flow/*.flow", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

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

let cachedDeckPackages: readonly DeckPackage[] | undefined;

/** Compiles and cross-validates the live Flow package set once per module. */
export function loadDeckPackages(): readonly DeckPackage[] {
  if (cachedDeckPackages !== undefined) {
    return cachedDeckPackages;
  }

  const packages: DeckPackage[] = [];
  const sourcePaths: string[] = [];

  for (const path of Object.keys(flowSources).sort()) {
    const source = flowSources[path];
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

    packages.push(result.value);
    sourcePaths.push(path);
  }

  const validated = validateExplainerSet(packages, { sourcePaths });
  if (!validated.ok || hasErrors(validated.diagnostics)) {
    throw compilationError("<live Flow package set>", validated.diagnostics);
  }

  cachedDeckPackages = validated.value;
  return cachedDeckPackages;
}

/** Returns the compiled package cache without compiling Flow sources. */
export function compiledDeckPackages(): readonly DeckPackage[] {
  if (cachedDeckPackages === undefined) {
    throw new Error(
      "Live Flow packages must be loaded before accessing the compiled package cache",
    );
  }
  return cachedDeckPackages;
}

/** Adapts the cached live Flow package set into deck definitions. */
export function loadDeckFlows(): DeckDefinition[] {
  return loadDeckPackages().map(packageToDeckDefinition);
}

/** Resolve one deck from the cached, validated live Flow package set. */
export function loadDeckFlowById(id: string): DeckDefinition | undefined {
  const pkg = loadDeckPackages().find((entry) => entry.id === id);
  return pkg === undefined ? undefined : packageToDeckDefinition(pkg);
}
