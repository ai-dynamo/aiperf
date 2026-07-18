// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime singleton theme registry for evaluation-time theme resolution.

import type { FlowThemeIr } from "@aiperf/flow-schema";
import {
  FrozenThemeRegistry,
  ThemeRegistry,
  type ResolvedTheme,
} from "./registry.js";
import { SYSTEMS_CHALK } from "./systems-chalk.js";

/** Create a bootstrap theme registry with bundled themes. */
function createBootstrapRegistry(): ThemeRegistry {
  const registry = new ThemeRegistry();
  registry.registerBundled([SYSTEMS_CHALK]);
  return registry;
}

/** Singleton runtime theme registry state. */
let runtimeRegistry: {
  mutable: ThemeRegistry;
  frozen: FrozenThemeRegistry | null;
  activeThemeId: string | null;
} = {
  mutable: createBootstrapRegistry(),
  frozen: null,
  activeThemeId: null,
};

/**
 * Register a theme at runtime before resolution begins.
 * Must be called before `resolveTheme()` or `getActiveTheme()`.
 *
 * @throws Error if the registry is already frozen
 * @throws DuplicateThemeIdError if the theme ID is already registered
 * @throws ReservedThemeIdError if the theme ID is reserved
 */
export function registerTheme(theme: FlowThemeIr): void {
  if (runtimeRegistry.frozen !== null) {
    throw new Error(
      "Cannot register theme after registry is frozen for resolution",
    );
  }
  runtimeRegistry.mutable.registerDocumentThemes([theme]);
}

/**
 * Register multiple themes at runtime before resolution begins.
 * Must be called before `resolveTheme()` or `getActiveTheme()`.
 *
 * @throws Error if the registry is already frozen
 * @throws DuplicateThemeIdError if any theme ID is already registered
 * @throws ReservedThemeIdError if any theme ID is reserved
 */
export function registerThemes(themes: readonly FlowThemeIr[]): void {
  if (runtimeRegistry.frozen !== null) {
    throw new Error(
      "Cannot register themes after registry is frozen for resolution",
    );
  }
  runtimeRegistry.mutable.registerDocumentThemes(themes);
}

/**
 * Freeze the runtime theme registry and prepare for theme resolution.
 * This is called automatically on first resolution if not already frozen.
 * Can be called explicitly to control freezing timing.
 */
export function freezeThemeRegistry(): FrozenThemeRegistry {
  if (runtimeRegistry.frozen === null) {
    runtimeRegistry.frozen = runtimeRegistry.mutable.freeze();
  }
  return runtimeRegistry.frozen;
}

/**
 * Resolve a theme ID to a fully resolved, validated theme with inheritance.
 * Automatically freezes the registry on first call if not already frozen.
 *
 * @param id - The theme ID to resolve
 * @returns The resolved theme with all inherited values
 * @throws UnknownThemeIdError if the theme ID is not registered
 * @throws ThemeInheritanceCycleError if the inheritance chain contains a cycle
 * @throws IncompleteThemeError if the resolved theme is missing required roles
 * @throws ThemeContrastError if contrast requirements are not met
 */
export function resolveTheme(id: string): ResolvedTheme {
  const frozen = freezeThemeRegistry();
  return frozen.resolve(id);
}

/**
 * Set the active theme ID for evaluation-time resolution.
 *
 * @param id - The theme ID to set as active, or null to clear
 */
export function setActiveThemeId(id: string | null): void {
  runtimeRegistry.activeThemeId = id;
}

/**
 * Get the currently active theme with resolved values.
 * Returns null if no active theme is set.
 *
 * @returns The resolved active theme or null
 * @throws UnknownThemeIdError if the active theme ID is not registered
 * @throws ThemeInheritanceCycleError if the inheritance chain contains a cycle
 * @throws IncompleteThemeError if the resolved theme is missing required roles
 * @throws ThemeContrastError if contrast requirements are not met
 */
export function getActiveTheme(): ResolvedTheme | null {
  if (runtimeRegistry.activeThemeId === null) {
    return null;
  }
  return resolveTheme(runtimeRegistry.activeThemeId);
}

/**
 * Get the ID of the currently active theme.
 * Returns null if no active theme is set.
 */
export function getActiveThemeId(): string | null {
  return runtimeRegistry.activeThemeId;
}

/**
 * Get all registered theme IDs (sorted).
 * Automatically freezes the registry if not already frozen.
 */
export function getRegisteredThemeIds(): readonly string[] {
  const frozen = freezeThemeRegistry();
  return frozen.ids();
}

/**
 * Check if a theme ID is registered.
 * Automatically freezes the registry if not already frozen.
 */
export function hasTheme(id: string): boolean {
  const frozen = freezeThemeRegistry();
  return frozen.has(id);
}

/**
 * Reset the runtime theme registry to initial state.
 * Useful for testing or resetting theme state.
 * After reset, you can register new themes and call freezeThemeRegistry() again.
 */
export function resetThemeRegistry(): void {
  runtimeRegistry = {
    mutable: createBootstrapRegistry(),
    frozen: null,
    activeThemeId: null,
  };
}
