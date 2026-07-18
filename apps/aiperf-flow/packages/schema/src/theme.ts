/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Closed theme-role vocabulary and typed theme value IR contracts.
 *
 * Interactive accent contrast pairs use a 3.0 floor (WCAG non-text / UI
 * component boundary), not the 4.5 normal-text floor.
 */

import { z } from "zod";

import type { SourceRange } from "./source.js";

export const THEME_ROLES = [
  "surface.canvas",
  "surface.panel",
  "surface.raised",
  "surface.control",
  "ink.primary",
  "ink.muted",
  "ink.inverse",
  "line.structural",
  "line.guide",
  "accent.control",
  "accent.execution",
  "accent.compute",
  "accent.attention",
  "accent.success",
  "accent.danger",
  "accent.focus",
  "font.display",
  "font.body",
  "font.data",
  "weight.regular",
  "weight.label",
  "weight.emphasis",
  "size.caption",
  "size.body",
  "size.label",
  "size.title",
  "stroke.hairline",
  "stroke.standard",
  "stroke.emphasis",
  "stroke.cap",
  "stroke.join",
  "motion.draw",
  "motion.enter",
  "motion.emphasis",
  "motion.stagger",
  "motion.easing",
] as const;

export type ThemeRole = (typeof THEME_ROLES)[number];

export type ThemeValueIr =
  | Readonly<{ kind: "color"; value: string }>
  | Readonly<{ kind: "font"; value: readonly string[] }>
  | Readonly<{ kind: "number"; value: number }>
  | Readonly<{ kind: "duration"; valueMs: number }>
  | Readonly<{ kind: "enum"; value: string }>;

export type ThemeRoleReferenceIr = Readonly<{
  kind: "theme-role";
  role: ThemeRole;
}>;

export type FlowThemeIr = Readonly<{
  id: string;
  extends: string;
  values: Readonly<Partial<Record<ThemeRole, ThemeValueIr>>>;
  sourceMap: SourceRange;
}>;

export type StyleScalarIr = string | number | boolean;
export type StyleValueIr = StyleScalarIr | ThemeRoleReferenceIr;

export type RequiredContrastPair = Readonly<{
  foreground: ThemeRole;
  background: ThemeRole;
  minRatio: number;
}>;

/** Hex colors must be `#RRGGBB` or `#RRGGBBAA`. */
const hexColorSchema = z
  .string()
  .regex(/^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/);

const themeEnumValueSchema = z.enum([
  "butt",
  "round",
  "square",
  "bevel",
  "miter",
  "linear",
  "ease_in",
  "ease_out",
  "ease_in_out",
]);

export const themeRoleSchema: z.ZodType<ThemeRole> = z.enum(THEME_ROLES);

export const themeValueIrSchema: z.ZodType<ThemeValueIr> = z.discriminatedUnion(
  "kind",
  [
    z.strictObject({
      kind: z.literal("color"),
      value: hexColorSchema,
    }),
    z.strictObject({
      kind: z.literal("font"),
      value: z.array(z.string().min(1)).min(1),
    }),
    z.strictObject({
      kind: z.literal("number"),
      value: z.number().finite().nonnegative(),
    }),
    z.strictObject({
      kind: z.literal("duration"),
      valueMs: z.number().int().nonnegative(),
    }),
    z.strictObject({
      kind: z.literal("enum"),
      value: themeEnumValueSchema,
    }),
  ],
);

export const themeRoleReferenceIrSchema: z.ZodType<ThemeRoleReferenceIr> =
  z.strictObject({
    kind: z.literal("theme-role"),
    role: themeRoleSchema,
  });

const sourcePositionSchema = z.strictObject({
  offset: z.number().int().nonnegative(),
  line: z.number().int().positive(),
  column: z.number().int().positive(),
});

const enumValuesByRole = {
  "stroke.cap": ["butt", "round", "square"],
  "stroke.join": ["bevel", "round", "miter"],
  "motion.easing": ["linear", "ease_in", "ease_out", "ease_in_out"],
} as const satisfies Partial<Record<ThemeRole, readonly string[]>>;

function expectedKind(role: ThemeRole): ThemeValueIr["kind"] {
  if (
    role.startsWith("surface.") ||
    role.startsWith("ink.") ||
    role.startsWith("line.") ||
    role.startsWith("accent.")
  ) {
    return "color";
  }
  if (role.startsWith("font.")) {
    return "font";
  }
  if (role === "stroke.cap" || role === "stroke.join" || role === "motion.easing") {
    return "enum";
  }
  if (role.startsWith("motion.")) {
    return "duration";
  }
  return "number";
}

export const flowThemeIrSchema: z.ZodType<FlowThemeIr> = z
  .strictObject({
    id: z.string().min(1),
    extends: z.string().min(1),
    values: z.partialRecord(themeRoleSchema, themeValueIrSchema),
    sourceMap: z.strictObject({
      source: z.string().min(1),
      start: sourcePositionSchema,
      end: sourcePositionSchema,
    }),
  })
  .superRefine((theme, context) => {
    for (const [role, value] of Object.entries(theme.values) as [
      ThemeRole,
      ThemeValueIr,
    ][]) {
      const kind = expectedKind(role);
      if (value.kind !== kind) {
        context.addIssue({
          code: "custom",
          path: ["values", role],
          message: `${role} requires a ${kind} theme value`,
        });
        continue;
      }

      if (
        value.kind === "number" &&
        role.startsWith("weight.") &&
        (!Number.isInteger(value.value) ||
          value.value < 100 ||
          value.value > 900)
      ) {
        context.addIssue({
          code: "custom",
          path: ["values", role, "value"],
          message: `${role} must be an integer from 100 to 900`,
        });
      } else if (
        value.kind === "number" &&
        role.startsWith("size.") &&
        value.value <= 0
      ) {
        context.addIssue({
          code: "custom",
          path: ["values", role, "value"],
          message: `${role} must be positive`,
        });
      } else if (value.kind === "enum") {
        const allowed = enumValuesByRole[role as keyof typeof enumValuesByRole];
        if (allowed !== undefined && !allowed.includes(value.value as never)) {
          context.addIssue({
            code: "custom",
            path: ["values", role, "value"],
            message: `${role} must be one of: ${allowed.join(", ")}`,
          });
        }
      }
    }
  });

/**
 * WCAG AA pairs checked at runtime theme registration.
 * Text pairs use 4.5; non-text / UI component pairs use 3.0.
 */
export const REQUIRED_CONTRAST_PAIRS: readonly RequiredContrastPair[] = [
  { foreground: "ink.primary", background: "surface.canvas", minRatio: 4.5 },
  { foreground: "ink.primary", background: "surface.panel", minRatio: 4.5 },
  { foreground: "ink.primary", background: "surface.raised", minRatio: 4.5 },
  { foreground: "ink.primary", background: "surface.control", minRatio: 4.5 },
  { foreground: "ink.muted", background: "surface.canvas", minRatio: 4.5 },
  { foreground: "ink.muted", background: "surface.panel", minRatio: 4.5 },
  { foreground: "accent.control", background: "surface.control", minRatio: 3.0 },
  {
    foreground: "accent.execution",
    background: "surface.control",
    minRatio: 3.0,
  },
  { foreground: "accent.compute", background: "surface.control", minRatio: 3.0 },
  {
    foreground: "accent.attention",
    background: "surface.control",
    minRatio: 3.0,
  },
  { foreground: "accent.success", background: "surface.control", minRatio: 3.0 },
  { foreground: "accent.danger", background: "surface.control", minRatio: 3.0 },
  { foreground: "accent.focus", background: "surface.canvas", minRatio: 3.0 },
  { foreground: "accent.focus", background: "surface.panel", minRatio: 3.0 },
  { foreground: "accent.focus", background: "surface.control", minRatio: 3.0 },
];

/** Parses a closed theme role or throws for unknown identifiers. */
export function parseThemeRole(input: string): ThemeRole {
  const parsed = themeRoleSchema.safeParse(input);
  if (!parsed.success) {
    throw new Error(`Unknown theme role: ${input}`);
  }
  return parsed.data;
}
