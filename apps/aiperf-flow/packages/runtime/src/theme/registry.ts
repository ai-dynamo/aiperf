// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional theme registration and deterministic inheritance resolution.

import {
  THEME_ROLES,
  type FlowThemeIr,
  type ThemeRole,
  type ThemeValueIr,
  themeRoleEnumValues,
  themeRoleKind,
} from "@aiperf/flow-schema";

import { validateThemeContrast } from "./contrast.js";
import {
  DuplicateThemeIdError,
  IncompleteThemeError,
  InvalidThemeValueError,
  ReservedThemeIdError,
  ThemeInheritanceCycleError,
  ThemeRoleKindError,
  UnknownThemeIdError,
  UnknownThemeRoleError,
  cloneTheme,
  deepFreeze,
  type ResolvedTheme,
} from "./types.js";

/** Internal sentinel parent accepted only for bundled root themes. */
export const BUNDLED_ROOT_BASE = "__bundled_root__" as const;

const THEME_ROLE_SET = new Set<string>(THEME_ROLES);
const HEX_COLOR = /^#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;

function validateThemeValue(
  themeId: string,
  role: ThemeRole,
  value: ThemeValueIr,
): void {
  const expectedKind = themeRoleKind(role);
  if (value.kind !== expectedKind) {
    throw new ThemeRoleKindError(
      `Theme "${themeId}" role "${role}" requires kind "${expectedKind}", received "${value.kind}"`,
    );
  }

  let valid: boolean;
  switch (value.kind) {
    case "color":
      valid = HEX_COLOR.test(value.value);
      break;
    case "font":
      valid =
        value.value.length > 0 &&
        value.value.every((family) => family.length > 0);
      break;
    case "number":
      valid = Number.isFinite(value.value) && value.value >= 0;
      if (role.startsWith("weight.")) {
        valid =
          valid &&
          Number.isInteger(value.value) &&
          value.value >= 100 &&
          value.value <= 900;
      } else if (role.startsWith("size.")) {
        valid = valid && value.value > 0;
      }
      break;
    case "duration":
      valid =
        Number.isFinite(value.valueMs) &&
        Number.isInteger(value.valueMs) &&
        value.valueMs >= 0;
      break;
    case "enum": {
      const allowed = themeRoleEnumValues(role);
      valid = allowed?.includes(value.value) ?? false;
      break;
    }
  }

  if (!valid) {
    throw new InvalidThemeValueError(
      `Theme "${themeId}" has an invalid value for role "${role}"`,
    );
  }
}

function validateOverrides(theme: FlowThemeIr): void {
  for (const [rawRole, value] of Object.entries(theme.values)) {
    if (!THEME_ROLE_SET.has(rawRole)) {
      throw new UnknownThemeRoleError(
        `Theme "${theme.id}" contains unknown role "${rawRole}"`,
      );
    }
    validateThemeValue(theme.id, rawRole as ThemeRole, value);
  }
}

/** Immutable theme lookup with cached inheritance resolution. */
export class FrozenThemeRegistry {
  readonly #themes: ReadonlyMap<string, FlowThemeIr>;
  readonly #ids: readonly string[];
  readonly #resolved = new Map<string, ResolvedTheme>();

  constructor(themes: ReadonlyMap<string, FlowThemeIr>) {
    this.#themes = new Map(themes);
    this.#ids = Object.freeze(
      [...this.#themes.keys()].sort((left, right) => left.localeCompare(right)),
    );
    Object.freeze(this);
  }

  ids(): readonly string[] {
    return this.#ids;
  }

  has(id: string): boolean {
    return this.#themes.has(id);
  }

  resolve(id: string): ResolvedTheme {
    return this.#resolved.get(id) ?? this.#resolve(id, []);
  }

  #resolve(id: string, path: readonly string[]): ResolvedTheme {
    const cached = this.#resolved.get(id);
    if (cached !== undefined) {
      return cached;
    }

    const cycleStart = path.indexOf(id);
    if (cycleStart !== -1) {
      const cycle = [...path.slice(cycleStart), id];
      throw new ThemeInheritanceCycleError(
        `Theme inheritance cycle: ${cycle.join(" -> ")}`,
      );
    }

    const theme = this.#themes.get(id);
    if (theme === undefined) {
      throw new UnknownThemeIdError(`Unknown theme ID "${id}"`);
    }
    validateOverrides(theme);

    const values: Partial<Record<ThemeRole, ThemeValueIr>> =
      theme.extends === BUNDLED_ROOT_BASE
        ? {}
        : { ...this.#resolve(theme.extends, [...path, id]).values };
    Object.assign(values, theme.values);

    const missing = THEME_ROLES.filter((role) => values[role] === undefined);
    if (missing.length > 0) {
      throw new IncompleteThemeError(
        `Theme "${id}" is missing required roles: ${missing.join(", ")}`,
      );
    }

    for (const role of THEME_ROLES) {
      validateThemeValue(theme.id, role, values[role]!);
    }
    const resolved = deepFreeze({
      id,
      values: values as Record<ThemeRole, ThemeValueIr>,
    });
    validateThemeContrast(resolved);
    this.#resolved.set(id, resolved);
    return resolved;
  }
}

/** Transactional builder for bundled and document theme definitions. */
export class ThemeRegistry {
  readonly #themes = new Map<string, FlowThemeIr>();
  readonly #bundledIds = new Set<string>();
  #frozen = false;

  registerBundled(themes: readonly FlowThemeIr[]): void {
    this.#register(themes, true);
  }

  registerDocumentThemes(themes: readonly FlowThemeIr[]): void {
    this.#register(themes, false);
  }

  freeze(): FrozenThemeRegistry {
    this.#frozen = true;
    return new FrozenThemeRegistry(this.#themes);
  }

  #register(themes: readonly FlowThemeIr[], bundled: boolean): void {
    if (this.#frozen) {
      throw new Error("Theme registry is frozen.");
    }

    const pendingIds = new Set<string>();
    for (const theme of themes) {
      if (
        !bundled &&
        (theme.id === BUNDLED_ROOT_BASE || this.#bundledIds.has(theme.id))
      ) {
        throw new ReservedThemeIdError(`Theme ID "${theme.id}" is reserved`);
      }
      if (!bundled && theme.extends === BUNDLED_ROOT_BASE) {
        throw new ReservedThemeIdError(
          `Theme "${theme.id}" cannot extend reserved bundled root "${BUNDLED_ROOT_BASE}"`,
        );
      }
      if (this.#themes.has(theme.id) || pendingIds.has(theme.id)) {
        throw new DuplicateThemeIdError(`Duplicate theme ID "${theme.id}"`);
      }
      pendingIds.add(theme.id);
    }

    const copies = themes.map(cloneTheme);
    for (const theme of copies) {
      this.#themes.set(theme.id, theme);
      if (bundled) {
        this.#bundledIds.add(theme.id);
      }
    }
  }
}

/** Applies host, document, then legacy active-theme precedence. */
export function selectActiveThemeId(input: Readonly<{
  overrideId?: string;
  documentDefault?: string;
  legacyId?: string;
}>): string | undefined {
  return input.overrideId ?? input.documentDefault ?? input.legacyId;
}
