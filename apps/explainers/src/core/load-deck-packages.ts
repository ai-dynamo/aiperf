/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckDefinition } from "./types";

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

function unwrapModuleExport(mod: unknown): unknown {
  if (mod !== null && typeof mod === "object" && "default" in mod) {
    return (mod as { default: unknown }).default;
  }
  return mod;
}

/** Eager Vite glob of compiler-emitted DeckPackage JSON modules. */
function generatedPackageModules(): Record<string, unknown> {
  // Vite requires a string literal here (not a shared constant).
  return import.meta.glob("../decks-generated/*.package.json", {
    eager: true,
  }) as Record<string, unknown>;
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
 * Convert a map of Vite (or test) JSON modules into DeckDefinitions via
 * `packageToDeckDefinition`.
 */
export function deckDefinitionsFromPackageModules<TPkg extends DeckPackageModule>(
  modules: Readonly<Record<string, unknown>>,
  packageToDeckDefinition: PackageToDeckDefinition<TPkg>,
): DeckDefinition[] {
  const decks: DeckDefinition[] = [];

  for (const path of Object.keys(modules).sort()) {
    const raw = unwrapModuleExport(modules[path]);
    if (!isDeckPackageModule(raw)) {
      throw new Error(`Invalid DeckPackage module at ${path}`);
    }
    decks.push(packageToDeckDefinition(raw as TPkg));
  }

  return decks;
}

/**
 * Eagerly load all `decks-generated/*.package.json` modules and adapt them
 * through `packageToDeckDefinition`.
 *
 * When no generated packages exist yet, returns an empty list so the registry
 * can keep a legacy fallback path.
 */
export function loadDeckPackages<TPkg extends DeckPackageModule>(
  packageToDeckDefinition: PackageToDeckDefinition<TPkg>,
): DeckDefinition[] {
  return deckDefinitionsFromPackageModules(
    generatedPackageModules(),
    packageToDeckDefinition,
  );
}

/**
 * Load a single generated package by deck id (`{id}.package.json`) and adapt it.
 * Returns `undefined` when the module is absent.
 */
export function loadDeckPackageById<TPkg extends DeckPackageModule>(
  id: string,
  packageToDeckDefinition: PackageToDeckDefinition<TPkg>,
): DeckDefinition | undefined {
  const modules = generatedPackageModules();
  const suffix = `/${id}.package.json`;
  const match = Object.entries(modules).find(([path]) => path.endsWith(suffix));
  if (!match) {
    return undefined;
  }
  const [path, mod] = match;
  return deckDefinitionsFromPackageModules({ [path]: mod }, packageToDeckDefinition)[0];
}
