// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Theme role → display instruction mapping and color value contracts.
//!
//! Defines the closed contract by which a resolved theme (from the theme registry)
//! applies visual and typographic styling to display-list rendering contexts.
//! Theme values flow through this interface to produce deterministic, evaluator-neutral
//! display instructions.

import type { ThemeRole, ThemeValueIr } from "@aiperf/flow-schema";

import type { ResolvedTheme } from "../theme/index.js";

/**
 * A single display property mapping — from one theme role to the display
 * property kind it controls and the extraction path for its value.
 *
 * Contracts establish which theme role can supply values to which categories
 * of visual properties (color, typography, motion, geometry), and which roles
 * are optional, required, or dependent on other roles.
 */
export type ThemeRoleContract = Readonly<{
  /** The theme role identifier, e.g., "ink.primary" or "motion.draw". */
  role: ThemeRole;

  /** The kind of value this role must provide: color, font, number, duration, enum. */
  valueKind: "color" | "font" | "number" | "duration" | "enum";

  /** Category of display property: foreground, background, accent, typography, spacing, motion, or structure. */
  category:
    | "foreground"
    | "background"
    | "accent"
    | "typography"
    | "spacing"
    | "motion"
    | "structure";

  /** Whether every resolved theme must provide this role. */
  required: boolean;

  /** Human-readable description of the role's purpose in the display contract. */
  description: string;

  /** Roles this role depends on (e.g., contrast pairs). Empty if none. */
  dependencies: readonly ThemeRole[];
}>;

/**
 * Display instruction extracted from a single theme value.
 *
 * A display instruction is the evaluated result of applying one theme value
 * to a logical display context (rendering path, text, shape, motion, etc.).
 * Instructions are serialization-ready and backend-neutral; they do not
 * reference host objects or runtime state.
 */
export type DisplayInstruction = Readonly<{
  /** The theme role this instruction derived from. */
  role: ThemeRole;

  /** The kind of value, determining the instruction structure. */
  kind: "color" | "font" | "number" | "duration" | "enum";

  /** The typed value extracted from the theme. */
  value: string | number | string[] | readonly string[];

  /** Category of display property this instruction controls. */
  category:
    | "foreground"
    | "background"
    | "accent"
    | "typography"
    | "spacing"
    | "motion"
    | "structure";

  /** Timestamp (in ms) when this instruction was applied, for lifecycle tracking. */
  appliedAtMs: number;
}>;

/**
 * Semantic role → display instruction mapping for one evaluated scene.
 *
 * Maps the semantic role of an entity (e.g., "request", "trace", "token")
 * to a set of applicable display instructions drawn from the active theme.
 * A scene's theme application context, including role mappings and display
 * instruction chains, is frozen at evaluation time.
 */
export type ThemeDisplayMapping = Readonly<{
  /** The ID of the resolved theme this mapping came from. */
  themeId: string;

  /**
   * Maps semantic entity role to the display instructions that apply to it.
   * Each semantic role can draw from multiple theme roles (e.g., both
   * "ink.primary" for text and "accent.control" for highlights).
   */
  semanticRoleToInstructions: Readonly<
    Record<string, readonly DisplayInstruction[]>
  >;

  /**
   * Lifecycle marker: which phase of the scene evaluation applied this mapping.
   * Valid values: "bootstrap", "resolve", "apply", "finalize".
   */
  appliedPhase: "bootstrap" | "resolve" | "apply" | "finalize";
}>;

/**
 * Color value contract specifying ranges, formats, and accessibility requirements.
 *
 * All colors in a resolved theme must be valid hex RGB or RGBA values.
 * Contrast ratios between foreground/background pairs are validated at theme
 * resolution time and frozen thereafter.
 */
export type ColorValueContract = Readonly<{
  /** Regex pattern for valid hex colors: #RRGGBB or #RRGGBBAA. */
  format: RegExp;

  /** Minimum WCAG contrast ratio required between specified foreground/background role pairs. */
  minContrastRatio: number;

  /**
   * List of (foreground, background) role pairs that must meet minContrastRatio.
   * Enforced at theme resolution, not at display-instruction time.
   */
  requiredContrastPairs: readonly Readonly<{
    foreground: ThemeRole;
    background: ThemeRole;
  }>[];
}>;

/**
 * Theme application lifecycle: the phases through which a theme is resolved,
 * validated, and applied to display-list rendering.
 *
 * Phases must complete in order and are not re-entrant:
 *
 * 1. **bootstrap**: Theme registry is created and bundled themes are registered.
 *    No document themes yet; no display instructions yet.
 *
 * 2. **resolve**: Document themes are registered and inherited chains are resolved.
 *    Theme values are validated and frozen. Contrast ratios are checked.
 *    At end of phase: the active theme ID is selected via selectActiveThemeId.
 *
 * 3. **apply**: Display instructions are generated from the resolved theme.
 *    Instructions are mapped to semantic roles via themeDisplayMapping.
 *    Role → instruction mappings are frozen and cached.
 *
 * 4. **finalize**: Scene rendering uses cached instructions. No further theme
 *    changes are accepted. Display lists are built using instruction values.
 *
 * All theme-related state after "finalize" is read-only. Re-applying a theme
 * requires starting a new scene evaluation cycle from "bootstrap".
 */
export type ThemeApplicationLifecycle = Readonly<{
  /** Current phase of the lifecycle. */
  phase: "bootstrap" | "resolve" | "apply" | "finalize";

  /** Timestamp (ms) when this phase began. */
  startedAtMs: number;

  /** Timestamp (ms) when this phase completed. Null if phase is current. */
  endedAtMs: number | null;

  /** ID of the resolved theme active in this phase. Null until "resolve" completes. */
  activeThemeId: string | null;

  /** Display mappings frozen in "apply" phase. Null until "apply" completes. */
  displayMapping: ThemeDisplayMapping | null;
}>;

/**
 * Theme role inventory: a closed, immutable set of valid role identifiers
 * and their contracts within the display-list system.
 *
 * Used by evaluators and renderers to validate that a resolved theme provides
 * all required roles and that display instructions are well-formed.
 */
export class ThemeRoleInventory {
  readonly #contracts: ReadonlyMap<ThemeRole, ThemeRoleContract>;
  readonly #rolesByCategory: Readonly<
    Record<string, readonly ThemeRole[] | undefined>
  >;

  /**
   * Constructs a role inventory from a closed set of contracts.
   * All theme roles that may appear in ResolvedTheme.values must be
   * registered here.
   */
  constructor(contracts: readonly ThemeRoleContract[]) {
    this.#contracts = new Map(contracts.map((c) => [c.role, c]));

    const byCategory = new Map<string, ThemeRole[]>();
    for (const contract of contracts) {
      const list = byCategory.get(contract.category) ?? [];
      list.push(contract.role);
      byCategory.set(contract.category, list);
    }

    const frozen = Object.fromEntries(
      [...byCategory.entries()].map(([cat, roles]) => [cat, Object.freeze(roles)]),
    );
    this.#rolesByCategory = Object.freeze(frozen);
  }

  /**
   * Retrieves the contract for a theme role, or undefined if the role
   * is not in the inventory.
   */
  contract(role: ThemeRole): ThemeRoleContract | undefined {
    return this.#contracts.get(role);
  }

  /**
   * Lists all roles in this inventory.
   */
  allRoles(): readonly ThemeRole[] {
    return Object.freeze([...this.#contracts.keys()]);
  }

  /**
   * Lists all roles in a specific category, or undefined if the category
   * is not recognized.
   */
  rolesByCategory(
    category: string,
  ): readonly ThemeRole[] | undefined {
    return this.#rolesByCategory[category];
  }

  /**
   * Lists all required roles in this inventory.
   */
  requiredRoles(): readonly ThemeRole[] {
    return this.allRoles().filter((role) => this.#contracts.get(role)?.required);
  }

  /**
   * Validates that a resolved theme provides all required roles
   * and that each role is registered in this inventory.
   *
   * Throws TypeError if a required role is absent or an unknown role is present.
   */
  validateThemeCompleteness(theme: ResolvedTheme): void {
    const missing = this.requiredRoles().filter((role) => theme.values[role] === undefined);
    if (missing.length > 0) {
      throw new TypeError(
        `Theme "${theme.id}" is missing required roles: ${missing.join(", ")}`,
      );
    }

    const unknown = Object.keys(theme.values).filter(
      (role) => !this.#contracts.has(role as ThemeRole),
    );
    if (unknown.length > 0) {
      throw new TypeError(
        `Theme "${theme.id}" contains unknown roles: ${unknown.join(", ")}`,
      );
    }
  }
}

/**
 * Extracts one DisplayInstruction from a theme value at a point in time.
 *
 * This is a pure function: given a theme role, its value, and the current
 * evaluation time, it produces a serializable instruction. No side effects.
 *
 * Throws TypeError if the role is not recognized or the value kind is invalid.
 */
export function displayInstructionFromThemeValue(
  role: ThemeRole,
  value: ThemeValueIr,
  category: string,
  atMs: number,
): DisplayInstruction {
  let instructionValue: string | number | string[] | readonly string[];

  switch (value.kind) {
    case "color":
      instructionValue = value.value;
      break;
    case "font":
      instructionValue = Object.freeze([...value.value]);
      break;
    case "number":
      instructionValue = value.value;
      break;
    case "duration":
      instructionValue = value.valueMs;
      break;
    case "enum":
      instructionValue = value.value;
      break;
    default:
      throw new TypeError(
        `Unknown theme value kind: ${(value as { kind: string }).kind}`,
      );
  }

  return Object.freeze({
    role,
    kind: value.kind,
    value: instructionValue,
    category: category as
      | "foreground"
      | "background"
      | "accent"
      | "typography"
      | "spacing"
      | "motion"
      | "structure",
    appliedAtMs: atMs,
  });
}

/**
 * Builds a theme-to-display-instruction mapping for a semantic role.
 *
 * Given a resolved theme and a semantic role name, returns the set of
 * display instructions that apply to that role. A semantic role may map
 * to zero or more theme roles (e.g., "request" may use "ink.primary" text
 * and "accent.control" highlight).
 *
 * The mapping is deterministic: same inputs always produce the same output,
 * and output is frozen and ready for serialization.
 */
export function buildThemeDisplayMapping(
  theme: ResolvedTheme,
  roleMapping: ReadonlyMap<string, readonly ThemeRole[]>,
  phase: "bootstrap" | "resolve" | "apply" | "finalize",
  atMs: number,
): ThemeDisplayMapping {
  const semanticRoleToInstructions: Record<
    string,
    readonly DisplayInstruction[]
  > = {};

  for (const [semanticRole, themeRoles] of roleMapping.entries()) {
    const instructions: DisplayInstruction[] = [];
    for (const themeRole of themeRoles) {
      const value = theme.values[themeRole];
      if (value !== undefined) {
        // Infer category from role name prefix convention.
        const rolePrefix = themeRole.split(".")[0] ?? "unknown";
        const category =
          rolePrefix === "ink"
            ? "foreground"
            : rolePrefix === "surface"
              ? "background"
              : rolePrefix === "accent"
                ? "accent"
                : rolePrefix === "font" || rolePrefix === "weight" || rolePrefix === "size"
                  ? "typography"
                  : rolePrefix === "stroke"
                    ? "structure"
                    : rolePrefix === "motion"
                      ? "motion"
                      : "structure";

        instructions.push(displayInstructionFromThemeValue(themeRole, value, category, atMs));
      }
    }
    semanticRoleToInstructions[semanticRole] = Object.freeze(instructions);
  }

  return Object.freeze({
    themeId: theme.id,
    semanticRoleToInstructions: Object.freeze(semanticRoleToInstructions),
    appliedPhase: phase,
  });
}

/**
 * Validates that a display instruction's value is compatible with its kind.
 *
 * This is a runtime guard: it ensures that serialized instructions can be
 * safely used by rendering backends without type errors.
 *
 * Throws TypeError if value does not match kind.
 */
export function validateDisplayInstruction(instruction: DisplayInstruction): void {
  const { kind, value } = instruction;

  switch (kind) {
    case "color":
      if (typeof value !== "string" || !/^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$/.test(value)) {
        throw new TypeError(
          `Display instruction role "${instruction.role}" kind "color" requires hex #RRGGBB or #RRGGBBAA, got ${value}`,
        );
      }
      break;

    case "font":
      if (!Array.isArray(value)) {
        throw new TypeError(
          `Display instruction role "${instruction.role}" kind "font" requires array, got ${typeof value}`,
        );
      }
      if (!value.every((v) => typeof v === "string" && v.length > 0)) {
        throw new TypeError(
          `Display instruction role "${instruction.role}" kind "font" requires non-empty string array`,
        );
      }
      break;

    case "number":
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
        throw new TypeError(
          `Display instruction role "${instruction.role}" kind "number" requires finite non-negative number, got ${value}`,
        );
      }
      break;

    case "duration":
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
        throw new TypeError(
          `Display instruction role "${instruction.role}" kind "duration" requires non-negative integer ms, got ${value}`,
        );
      }
      break;

    case "enum":
      if (typeof value !== "string") {
        throw new TypeError(
          `Display instruction role "${instruction.role}" kind "enum" requires string value, got ${typeof value}`,
        );
      }
      break;

    default:
      throw new TypeError(`Unknown instruction kind: ${kind}`);
  }
}
