/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckDefinition } from "./types";

// Eager static imports — Vite/Vitest always resolve these; glob alone can miss
// newly generated JSON under decks-generated until a full restart.
import cellularAlgorithmsPackage from "../decks-generated/cellular-algorithms.package.json";
import cellularInternalsPackage from "../decks-generated/cellular-internals.package.json";
import dynosimPackage from "../decks-generated/dynosim.package.json";
import rustArchitectureAtlasPackage from "../decks-generated/rust-architecture-atlas.package.json";
import rustArchitecturePackage from "../decks-generated/rust-architecture.package.json";
import segmentPoolsPackage from "../decks-generated/segment-pools.package.json";
import slurmVeloPackage from "../decks-generated/slurm-velo.package.json";
import veloDeepDivePackage from "../decks-generated/velo-deep-dive.package.json";

/** Minimal DeckPackage shape required before handing off to the adapter. */
export type DeckPackageModule = Readonly<{
  schemaVersion: 1;
  id: string;
  route: string;
}>;

/** Adapter seam: convert a validated DeckPackage into a legacy DeckDefinition. */
export type PackageToDeckDefinition<TPkg extends DeckPackageModule = DeckPackageModule> = (
  pkg: TPkg,
) => DeckDefinition;

/** One discovered generated package module, keyed later by deck id. */
export type GeneratedPackageEntry = Readonly<{
  path: string;
  module: unknown;
}>;

/** Filename pattern for compiler-emitted DeckPackage artifacts. */
const PACKAGE_PATH_RE = /(?:^|\/)([^/]+)\.package\.(json|ts|js)$/;

/**
 * Canonical eager import map for every registered DeckPackage JSON artifact.
 * Keys use the same relative paths Vite would emit from `import.meta.glob`.
 */
const EAGER_PACKAGE_JSON_MODULES: Record<string, unknown> = {
  "../decks-generated/cellular-algorithms.package.json": cellularAlgorithmsPackage,
  "../decks-generated/cellular-internals.package.json": cellularInternalsPackage,
  "../decks-generated/dynosim.package.json": dynosimPackage,
  "../decks-generated/rust-architecture-atlas.package.json": rustArchitectureAtlasPackage,
  "../decks-generated/rust-architecture.package.json": rustArchitecturePackage,
  "../decks-generated/segment-pools.package.json": segmentPoolsPackage,
  "../decks-generated/slurm-velo.package.json": slurmVeloPackage,
  "../decks-generated/velo-deep-dive.package.json": veloDeepDivePackage,
};

/**
 * Extract the deck id from a generated package path such as
 * `../decks-generated/rust-architecture.package.json`.
 */
export function packageIdFromPath(path: string): string | undefined {
  const match = PACKAGE_PATH_RE.exec(path);
  return match?.[1];
}

/** Prefer JSON artifacts over TS/JS when the same deck id appears twice. */
function packageExtensionPriority(path: string): number {
  if (path.endsWith(".package.json")) return 0;
  if (path.endsWith(".package.ts")) return 1;
  if (path.endsWith(".package.js")) return 2;
  return 3;
}

function unwrapModuleExport(mod: unknown): unknown {
  if (mod === null || typeof mod !== "object") {
    return mod;
  }

  const record = mod as Record<string, unknown>;
  if ("default" in record && isDeckPackageModule(record.default)) {
    return record.default;
  }

  for (const value of Object.values(record)) {
    if (isDeckPackageModule(value)) {
      return value;
    }
  }

  if ("default" in record) {
    return record.default;
  }

  return mod;
}

/**
 * Eager Vite globs plus the canonical JSON import map.
 * Supports `.package.json` (build output) and `.package.ts` / `.package.js`
 * golden modules. The eager map guarantees all eight registered decks are
 * discoverable even when glob alone returns empty.
 */
export function generatedPackageModules(): Record<string, unknown> {
  // Vite requires string literals here (not shared constants). Recursive
  // globs pick up nested or newly added artifacts beyond the eager map.
  const jsonModules = import.meta.glob("../decks-generated/**/*.package.json", {
    eager: true,
  }) as Record<string, unknown>;
  const tsModules = import.meta.glob("../decks-generated/**/*.package.ts", {
    eager: true,
  }) as Record<string, unknown>;
  const jsModules = import.meta.glob("../decks-generated/**/*.package.js", {
    eager: true,
  }) as Record<string, unknown>;
  // Eager map last so known packages always win over flaky/empty globs.
  return {
    ...jsModules,
    ...tsModules,
    ...jsonModules,
    ...EAGER_PACKAGE_JSON_MODULES,
  };
}

/**
 * Build an id → module import map from a Vite (or test) module record.
 * When both JSON and TS exist for the same id, JSON wins.
 */
export function indexGeneratedPackagesById(
  modules: Readonly<Record<string, unknown>>,
): Map<string, GeneratedPackageEntry> {
  const byId = new Map<string, GeneratedPackageEntry>();

  for (const path of Object.keys(modules).sort()) {
    const id = packageIdFromPath(path);
    if (id === undefined) {
      continue;
    }
    const existing = byId.get(id);
    if (
      existing !== undefined &&
      packageExtensionPriority(existing.path) <= packageExtensionPriority(path)
    ) {
      continue;
    }
    byId.set(id, { path, module: modules[path] });
  }

  return byId;
}

/** Returns true when `value` looks like a schemaVersion-1 DeckPackage. */
export function isDeckPackageModule(value: unknown): value is DeckPackageModule {
  if (value === null || typeof value !== "object") {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    candidate.schemaVersion === 1 &&
    typeof candidate.id === "string" &&
    candidate.id.length > 0 &&
    typeof candidate.route === "string" &&
    candidate.route.length > 0
  );
}

/**
 * Convert a map of Vite (or test) JSON/TS modules into DeckDefinitions via
 * `packageToDeckDefinition`.
 */
export function deckDefinitionsFromPackageModules<TPkg extends DeckPackageModule>(
  modules: Readonly<Record<string, unknown>>,
  packageToDeckDefinition: PackageToDeckDefinition<TPkg>,
): DeckDefinition[] {
  const decks: DeckDefinition[] = [];
  const byId = indexGeneratedPackagesById(modules);

  for (const id of [...byId.keys()].sort()) {
    const entry = byId.get(id);
    if (entry === undefined) {
      continue;
    }
    const raw = unwrapModuleExport(entry.module);
    if (!isDeckPackageModule(raw)) {
      throw new Error(`Invalid DeckPackage module at ${entry.path}`);
    }
    if (raw.id !== id) {
      throw new Error(
        `DeckPackage id mismatch at ${entry.path}: filename id "${id}" vs package id "${raw.id}"`,
      );
    }
    decks.push(packageToDeckDefinition(raw as TPkg));
  }

  return decks;
}

/**
 * Resolve a single generated package module by deck id from an import map.
 * Returns `undefined` when no matching `{id}.package.{json,ts,js}` exists.
 */
export function resolveGeneratedPackageById(
  id: string,
  modules: Readonly<Record<string, unknown>>,
): GeneratedPackageEntry | undefined {
  return indexGeneratedPackagesById(modules).get(id);
}

/**
 * Eagerly load all decks-generated `*.package.{json,ts,js}` modules (recursive)
 * and adapt them through `packageToDeckDefinition`.
 *
 * Discovery prefers the eager JSON import map, then recursive Vite globs.
 */
export function loadDeckPackages<TPkg extends DeckPackageModule>(
  packageToDeckDefinition: PackageToDeckDefinition<TPkg>,
  modules: Readonly<Record<string, unknown>> = generatedPackageModules(),
): DeckDefinition[] {
  return deckDefinitionsFromPackageModules(modules, packageToDeckDefinition);
}

/**
 * Load a single generated package by deck id and adapt it.
 * Returns `undefined` when the module is absent (legacy dual-load fallback).
 */
export function loadDeckPackageById<TPkg extends DeckPackageModule>(
  id: string,
  packageToDeckDefinition: PackageToDeckDefinition<TPkg>,
  modules: Readonly<Record<string, unknown>> = generatedPackageModules(),
): DeckDefinition | undefined {
  const entry = resolveGeneratedPackageById(id, modules);
  if (entry === undefined) {
    return undefined;
  }
  return deckDefinitionsFromPackageModules(
    { [entry.path]: entry.module },
    packageToDeckDefinition,
  )[0];
}

/** True when at least one generated DeckPackage artifact is discoverable. */
export function hasGeneratedDeckPackages(
  modules: Readonly<Record<string, unknown>> = generatedPackageModules(),
): boolean {
  return indexGeneratedPackagesById(modules).size > 0;
}
