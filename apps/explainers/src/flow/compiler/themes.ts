/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Theme collection and authoring validation for linked `.flow` documents.

import type {
  ThemeAssignmentAst,
  ThemeDeclarationAst,
  UseThemeAst,
} from "../language/index.js";
import {
  diagnostic,
  hasErrors,
  themeRoleEnumValues,
  themeRoleKind,
  THEME_ROLES,
  type Diagnostic,
  type Result,
  type ThemeRole,
} from "../schema/index.js";

export const BUNDLED_THEME_IDS = ["systems_chalk"] as const;

export type ThemeValidationInput = Readonly<{
  themes: readonly ThemeDeclarationAst[];
  useTheme?: UseThemeAst;
  bundledThemeIds?: readonly string[];
}>;

export type ThemeValidationOutput = Readonly<{
  themes: readonly ThemeDeclarationAst[];
  defaultTheme?: string;
}>;

const HEX_COLOR = /^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;
const THEME_ROLE_SET = new Set<string>(THEME_ROLES);

function isThemeRole(role: string): role is ThemeRole {
  return THEME_ROLE_SET.has(role);
}

function receivedValue(assignment: ThemeAssignmentAst): string {
  const { value } = assignment;
  if (value.kind === "theme-font-literal") {
    return `[${value.families.map((family) => `"${family}"`).join(", ")}]`;
  }
  return String(value.value);
}

function invalidValueDetail(
  assignment: ThemeAssignmentAst,
  role: ThemeRole,
): string | undefined {
  const { value } = assignment;
  switch (assignment.valueKind) {
    case "color": {
      const literal = value.kind === "literal" ? value.value : undefined;
      if (typeof literal === "string" && HEX_COLOR.test(literal)) {
        return undefined;
      }
      return `an invalid color value "${receivedValue(assignment)}"; expected #RRGGBB or #RRGGBBAA`;
    }
    case "number": {
      const literal = value.kind === "literal" ? value.value : undefined;
      if (typeof literal === "number" && Number.isFinite(literal)) {
        return undefined;
      }
      return `a non-finite number value "${receivedValue(assignment)}"`;
    }
    case "duration": {
      const literal = value.kind === "literal" ? value.value : undefined;
      if (typeof literal === "number" && Number.isFinite(literal)) {
        return undefined;
      }
      return `a non-finite duration value "${receivedValue(assignment)}"`;
    }
    case "enum": {
      const literal = value.kind === "literal" ? value.value : undefined;
      const allowed = themeRoleEnumValues(role) ?? [];
      if (typeof literal === "string" && allowed.includes(literal)) {
        return undefined;
      }
      return `an invalid enum value "${receivedValue(assignment)}"; expected one of: ${allowed.join(", ")}`;
    }
    case "font": {
      const families = value.kind === "theme-font-literal" ? value.families : [];
      if (
        families.length > 0 &&
        families.every((family) => family.trim().length > 0)
      ) {
        return undefined;
      }
      return "an empty font stack";
    }
  }
}

function assignmentDiagnostics(theme: ThemeDeclarationAst): readonly Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  for (const assignment of theme.assignments) {
    if (!isThemeRole(assignment.role)) {
      diagnostics.push(
        diagnostic(
          "THEME_UNKNOWN_ROLE",
          "error",
          `Theme "${theme.id}" assigns unknown role "${assignment.role}".`,
          assignment.sourceMap,
          "Use a role from the closed theme-role vocabulary.",
        ),
      );
      continue;
    }

    const expected = themeRoleKind(assignment.role);
    if (assignment.valueKind !== expected) {
      diagnostics.push(
        diagnostic(
          "THEME_ROLE_KIND_MISMATCH",
          "error",
          `Theme "${theme.id}" role "${assignment.role}" expects a ${expected} value but received a ${assignment.valueKind} value "${receivedValue(assignment)}".`,
          assignment.sourceMap,
          `Assign "${assignment.role}" with a ${expected} value.`,
        ),
      );
      continue;
    }

    const detail = invalidValueDetail(assignment, assignment.role);
    if (detail !== undefined) {
      diagnostics.push(
        diagnostic(
          "THEME_INVALID_VALUE",
          "error",
          `Theme "${theme.id}" role "${assignment.role}" has ${detail}.`,
          assignment.sourceMap,
          "Provide a value that matches the role's typed vocabulary.",
        ),
      );
    }
  }
  return diagnostics;
}

function cycleDiagnostics(
  authored: ReadonlyMap<string, ThemeDeclarationAst>,
): readonly Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  for (const [id, theme] of authored) {
    const visited = new Set<string>([id]);
    let current = theme.extends;
    let cyclic = false;
    while (authored.has(current)) {
      if (visited.has(current)) {
        cyclic = current === id;
        break;
      }
      visited.add(current);
      current = authored.get(current)?.extends ?? "";
    }
    if (cyclic) {
      diagnostics.push(
        diagnostic(
          "THEME_INHERITANCE_CYCLE",
          "error",
          `Theme inheritance cycle involving "${id}" through "${theme.extends}".`,
          theme.sourceMap,
          "Break the cycle so every custom theme resolves to a bundled base.",
        ),
      );
    }
  }
  return diagnostics;
}

/** Validates authored themes and resolves the selected default theme. */
export function validateThemes(
  input: ThemeValidationInput,
): Result<ThemeValidationOutput> {
  const themes = input.themes ?? [];
  const bundled = new Set(input.bundledThemeIds ?? BUNDLED_THEME_IDS);
  const diagnostics: Diagnostic[] = [];

  const authored = new Map<string, ThemeDeclarationAst>();
  for (const theme of themes) {
    if (bundled.has(theme.id)) {
      diagnostics.push(
        diagnostic(
          "THEME_RESERVED_ID",
          "error",
          `Theme id "${theme.id}" is reserved by a bundled theme.`,
          theme.sourceMap,
          "Choose a theme id that a bundled theme does not already use.",
        ),
      );
    } else if (authored.has(theme.id)) {
      diagnostics.push(
        diagnostic(
          "THEME_DUPLICATE_ID",
          "error",
          `Duplicate theme id "${theme.id}".`,
          theme.sourceMap,
          "Rename this theme or remove the earlier declaration with the same id.",
        ),
      );
    } else {
      authored.set(theme.id, theme);
    }
  }

  const known = new Set<string>([...bundled, ...authored.keys()]);
  for (const theme of themes) {
    if (!known.has(theme.extends)) {
      diagnostics.push(
        diagnostic(
          "THEME_UNKNOWN_BASE",
          "error",
          `Theme "${theme.id}" extends unknown base theme "${theme.extends}".`,
          theme.sourceMap,
          "Extend a bundled theme or an authored theme declared in this document.",
        ),
      );
    }
    diagnostics.push(...assignmentDiagnostics(theme));
  }

  diagnostics.push(...cycleDiagnostics(authored));

  if (input.useTheme !== undefined && !known.has(input.useTheme.themeId)) {
    diagnostics.push(
      diagnostic(
        "THEME_UNKNOWN_DEFAULT",
        "error",
        `Selected default theme "${input.useTheme.themeId}" is not defined.`,
        input.useTheme.sourceMap,
        "Select a bundled theme or an authored theme declared in this document.",
      ),
    );
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return {
    ok: true,
    value:
      input.useTheme === undefined
        ? { themes }
        : { themes, defaultTheme: input.useTheme.themeId },
    diagnostics,
  };
}
