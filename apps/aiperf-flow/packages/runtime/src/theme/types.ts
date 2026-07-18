// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime theme result and error contracts.

import type {
  FlowThemeIr,
  ThemeRole,
  ThemeValueIr,
} from "@aiperf/flow-schema";

/** A fully inherited, validated runtime theme. */
export type ResolvedTheme = Readonly<{
  id: string;
  values: Readonly<Record<ThemeRole, ThemeValueIr>>;
}>;

class NamedThemeError extends Error {
  constructor(name: string, message: string) {
    super(message);
    this.name = name;
  }
}

/** Raised when a theme ID is registered more than once. */
export class DuplicateThemeIdError extends NamedThemeError {
  constructor(message: string) {
    super("DuplicateThemeIdError", message);
  }
}

/** Raised when authored content uses a runtime-reserved theme ID. */
export class ReservedThemeIdError extends NamedThemeError {
  constructor(message: string) {
    super("ReservedThemeIdError", message);
  }
}

/** Raised when a requested theme or parent theme is absent. */
export class UnknownThemeIdError extends NamedThemeError {
  constructor(message: string) {
    super("UnknownThemeIdError", message);
  }
}

/** Raised when a theme inheritance chain contains a cycle. */
export class ThemeInheritanceCycleError extends NamedThemeError {
  constructor(message: string) {
    super("ThemeInheritanceCycleError", message);
  }
}

/** Raised when a resolved theme does not define every required role. */
export class IncompleteThemeError extends NamedThemeError {
  constructor(message: string) {
    super("IncompleteThemeError", message);
  }
}

/** Raised when a required foreground/background pair has insufficient contrast. */
export class ThemeContrastError extends NamedThemeError {
  constructor(message: string) {
    super("ThemeContrastError", message);
  }
}

/** Raised when a role receives the wrong theme value discriminant. */
export class ThemeRoleKindError extends NamedThemeError {
  constructor(message: string) {
    super("ThemeRoleKindError", message);
  }
}

/** Raised when untrusted IR contains a role outside the closed vocabulary. */
export class UnknownThemeRoleError extends NamedThemeError {
  constructor(message: string) {
    super("UnknownThemeRoleError", message);
  }
}

/** Raised when untrusted IR contains an invalid value payload. */
export class InvalidThemeValueError extends NamedThemeError {
  constructor(message: string) {
    super("InvalidThemeValueError", message);
  }
}

export function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) {
      deepFreeze(child);
    }
    Object.freeze(value);
  }
  return value;
}

export function cloneTheme(theme: FlowThemeIr): FlowThemeIr {
  const values = Object.fromEntries(
    Object.entries(theme.values).map(([role, value]) => [
      role,
      value.kind === "font"
        ? { ...value, value: [...value.value] }
        : { ...value },
    ]),
  ) as Partial<Record<ThemeRole, ThemeValueIr>>;

  return deepFreeze({
    id: theme.id,
    extends: theme.extends,
    values,
    sourceMap: {
      source: theme.sourceMap.source,
      start: { ...theme.sourceMap.start },
      end: { ...theme.sourceMap.end },
    },
  });
}
