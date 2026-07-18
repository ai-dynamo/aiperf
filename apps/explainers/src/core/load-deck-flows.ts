/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  compileExplainerSource,
  FOUNDATION_CAPABILITIES,
  hasErrors,
  type Diagnostic,
} from "../flow";
import { packageToDeckDefinition } from "./package-adapter";
import type { DeckDefinition } from "./types";

const flowSources = import.meta.glob("../../decks-flow/*.flow", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const FLOW_PATH_RE = /(?:^|\/)([^/]+)\.flow$/;

function formatDiagnostic(diagnostic: Diagnostic): string {
  const { source, start } = diagnostic.range;
  const repair =
    diagnostic.repair === undefined ? "" : ` (${diagnostic.repair})`;
  return `${source}:${start.line}:${start.column}: ${diagnostic.severity} ${diagnostic.code}: ${diagnostic.message}${repair}`;
}

function compilationError(path: string, diagnostics: readonly Diagnostic[]): Error {
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

/** Eagerly compile all raw Flow sources and adapt them into deck definitions. */
export function loadDeckFlows(): DeckDefinition[] {
  const decks: DeckDefinition[] = [];
  const sourceById = new Map<string, string>();

  for (const path of Object.keys(flowSources).sort()) {
    const source = flowSources[path];
    const result = compileExplainerSource({
      source,
      sourceName: path,
      capabilities: FOUNDATION_CAPABILITIES,
      strict: true,
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

    const duplicatePath = sourceById.get(result.value.id);
    if (duplicatePath !== undefined) {
      throw new Error(
        `Duplicate live Flow deck id "${result.value.id}" in "${duplicatePath}" and "${path}"`,
      );
    }
    sourceById.set(result.value.id, path);
    decks.push(packageToDeckDefinition(result.value));
  }

  return decks;
}

/** Compile and resolve one live Flow deck by id. */
export function loadDeckFlowById(id: string): DeckDefinition | undefined {
  return loadDeckFlows().find((deck) => deck.id === id);
}
