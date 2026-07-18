// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional runtime theme registration and deterministic resolution.

import {
  REQUIRED_CONTRAST_PAIRS,
  THEME_ROLES,
  type FlowThemeIr,
  type ThemeRole,
  type ThemeValueIr,
  themeRoleEnumValues,
  themeRoleKind,
} from "@aiperf/flow-schema";

/** Sentinel parent used only by complete bundled root themes. */
export const BUNDLED_ROOT_BASE = "__bundled_root__";

/** A fully inherited, validated runtime theme. */
export type ResolvedTheme = Readonly<{
  id: string;
  values: Readonly<Record<ThemeRole, ThemeValueIr>>;
}>;

export class DuplicateThemeIdError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DuplicateThemeIdError";
  }
}

export class ReservedThemeIdError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReservedThemeIdError";
  }
}

export class UnknownThemeIdError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnknownThemeIdError";
  }
}

export class ThemeInheritanceCycleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ThemeInheritanceCycleError";
  }
}

export class UnknownThemeRoleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnknownThemeRoleError";
  }
}

export class ThemeRoleKindError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ThemeRoleKindError";
  }
}

export class InvalidThemeValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidThemeValueError";
  }
}

export class IncompleteThemeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IncompleteThemeError";
  }
}

export class ThemeContrastError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ThemeContrastError";
  }
}

const THEME_ROLE_SET = new Set<string>(THEME_ROLES);
const HEX_COLOR = /^#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) {
      deepFreeze(child);
    }
    Object.freeze(value);
  }
  return value;
}

function cloneTheme(theme: FlowThemeIr): FlowThemeIr {
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

  let valid = true;
  switch (value.kind) {
    case "color":
      valid = HEX_COLOR.test(value.value);
      break;
    case "font":
      valid =
        Array.isArray(value.value) &&
        value.value.length > 0 &&
        value.value.every((family) => typeof family === "string" && family.length > 0);
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
        Number.isInteger(value.valueMs) &&
        Number.isFinite(value.valueMs) &&
        value.valueMs >= 0;
      break;
    case "enum": {
      const allowed = themeRoleEnumValues(role);
      valid = allowed !== undefined && allowed.includes(value.value);
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

type Rgb = readonly [number, number, number];

function parseRgb(color: string): Rgb {
  return [
    Number.parseInt(color.slice(1, 3), 16),
    Number.parseInt(color.slice(3, 5), 16),
    Number.parseInt(color.slice(5, 7), 16),
  ];
}

function relativeLuminance(color: string): number {
  const channels = parseRgb(color).map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return (
    0.2126 * channels[0]! + 0.7152 * channels[1]! + 0.0722 * channels[2]!
  );
}

function contrastRatio(foreground: string, background: string): number {
  const lighter = Math.max(
    relativeLuminance(foreground),
    relativeLuminance(background),
  );
  const darker = Math.min(
    relativeLuminance(foreground),
    relativeLuminance(background),
  );
  return (lighter + 0.05) / (darker + 0.05);
}

function validateResolvedTheme(theme: ResolvedTheme): void {
  for (const pair of REQUIRED_CONTRAST_PAIRS) {
    const foreground = theme.values[pair.foreground];
    const background = theme.values[pair.background];
    if (foreground.kind !== "color" || background.kind !== "color") {
      throw new ThemeRoleKindError(
        `Theme "${theme.id}" contrast roles must contain color values`,
      );
    }
    const ratio = contrastRatio(foreground.value, background.value);
    if (ratio + Number.EPSILON < pair.minRatio) {
      throw new ThemeContrastError(
        `Theme "${theme.id}" contrast for "${pair.foreground}" on "${pair.background}" is ${ratio.toFixed(2)}; requires ${pair.minRatio.toFixed(1)}`,
      );
    }
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

  has(id: string): boolean {
    return this.#themes.has(id);
  }

  ids(): readonly string[] {
    return this.#ids;
  }

  resolve(id: string): ResolvedTheme {
    const cached = this.#resolved.get(id);
    if (cached !== undefined) {
      return cached;
    }
    return this.#resolve(id, []);
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

    const resolved = deepFreeze({
      id,
      values: values as Record<ThemeRole, ThemeValueIr>,
    });
    validateResolvedTheme(resolved);
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

const bundledSourceMap = {
  source: "runtime:systems_chalk",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
} as const;

/** Complete bundled Systems Chalk root theme. */
export const SYSTEMS_CHALK: FlowThemeIr = deepFreeze({
  id: "systems_chalk",
  extends: BUNDLED_ROOT_BASE,
  sourceMap: bundledSourceMap,
  values: {
    "surface.canvas": { kind: "color", value: "#232526" },
    "surface.panel": { kind: "color", value: "#292C2D" },
    "surface.raised": { kind: "color", value: "#303334" },
    "surface.control": { kind: "color", value: "#383C3E" },
    "ink.primary": { kind: "color", value: "#F1F3F2" },
    "ink.muted": { kind: "color", value: "#AEB4B5" },
    "ink.inverse": { kind: "color", value: "#232526" },
    "line.structural": { kind: "color", value: "#D7DADA" },
    "line.guide": { kind: "color", value: "#777D80" },
    "accent.control": { kind: "color", value: "#71D8D0" },
    "accent.execution": { kind: "color", value: "#69C8BA" },
    "accent.compute": { kind: "color", value: "#77B8DE" },
    "accent.attention": { kind: "color", value: "#F0CF58" },
    "accent.success": { kind: "color", value: "#7DCE82" },
    "accent.danger": { kind: "color", value: "#F07972" },
    "accent.focus": { kind: "color", value: "#9BDBF5" },
    "font.display": {
      kind: "font",
      value: ["Nunito Sans", "Segoe UI", "sans-serif"],
    },
    "font.body": {
      kind: "font",
      value: ["Nunito Sans", "Segoe UI", "sans-serif"],
    },
    "font.data": {
      kind: "font",
      value: ["IBM Plex Mono", "Cascadia Code", "monospace"],
    },
    "weight.regular": { kind: "number", value: 400 },
    "weight.label": { kind: "number", value: 500 },
    "weight.emphasis": { kind: "number", value: 600 },
    "size.caption": { kind: "number", value: 11 },
    "size.body": { kind: "number", value: 13 },
    "size.label": { kind: "number", value: 12 },
    "size.title": { kind: "number", value: 18 },
    "stroke.hairline": { kind: "number", value: 1 },
    "stroke.standard": { kind: "number", value: 2 },
    "stroke.emphasis": { kind: "number", value: 3 },
    "stroke.cap": { kind: "enum", value: "round" },
    "stroke.join": { kind: "enum", value: "round" },
    "motion.draw": { kind: "duration", valueMs: 420 },
    "motion.enter": { kind: "duration", valueMs: 240 },
    "motion.emphasis": { kind: "duration", valueMs: 180 },
    "motion.stagger": { kind: "duration", valueMs: 60 },
    "motion.easing": { kind: "enum", value: "ease_out" },
  },
});

/** Creates the standard registry with all bundled themes registered. */
export function createBootstrapThemeRegistry(): ThemeRegistry {
  const registry = new ThemeRegistry();
  registry.registerBundled([SYSTEMS_CHALK]);
  return registry;
}

export type ActiveThemeSelection = Readonly<{
  overrideId?: string;
  documentDefault?: string;
  legacyId?: string;
}>;

/** Applies host, document, then legacy active-theme precedence. */
export function selectActiveThemeId(
  selection: ActiveThemeSelection,
): string | undefined {
  return selection.overrideId ?? selection.documentDefault ?? selection.legacyId;
}

export { LEGACY_VISUAL_FALLBACKS } from "./legacy-defaults.js";
export { SYSTEMS_CHALK_SHAPE } from "./systems-chalk.js";
export {
  freezeThemeRegistry,
  getActiveTheme,
  getActiveThemeId,
  getRegisteredThemeIds,
  hasTheme,
  registerTheme,
  registerThemes,
  resetThemeRegistry,
  resolveTheme,
  setActiveThemeId,
} from "./runtime-registry.js";
