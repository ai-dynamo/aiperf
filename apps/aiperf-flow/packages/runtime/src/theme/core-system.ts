// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal core theme system for bootstrap with light/dark variants and inheritance.

/** Minimal color role identifiers for core theming. */
export type CoreThemeRole = "text" | "background" | "accent";

/** Theme variant selector. */
export type ThemeVariant = "light" | "dark";

/**
 * A color definition in hex format (e.g., "#FFFFFF").
 * Must be 6 or 8 digit hex color.
 */
export type HexColor = string & { readonly __hex: unique symbol };

/**
 * Creates a validated hex color.
 * Throws if the format is invalid.
 */
export function createHexColor(value: string): HexColor {
  const HEX_PATTERN = /^#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;
  if (!HEX_PATTERN.test(value)) {
    throw new Error(
      `Invalid hex color "${value}". Expected #RRGGBB or #RRGGBBAA format.`,
    );
  }
  return value as HexColor;
}

/** Color values for a specific variant of a theme. */
export type CoreThemeVariantColors = Readonly<{
  text: HexColor;
  background: HexColor;
  accent: HexColor;
}>;

/** Core theme variant configuration. */
export type CoreThemeVariantConfig = Readonly<{
  light?: Partial<CoreThemeVariantColors>;
  dark?: Partial<CoreThemeVariantColors>;
}>;

/**
 * Error thrown when a role cannot be resolved.
 */
export class CoreThemeRoleNotFoundError extends Error {
  constructor(role: CoreThemeRole, variant: ThemeVariant, themeId: string) {
    super(
      `Role "${role}" not found in ${variant} variant of theme "${themeId}"`,
    );
    this.name = "CoreThemeRoleNotFoundError";
  }
}

/**
 * Error thrown when theme inheritance is cyclic.
 */
export class CoreThemeInheritanceCycleError extends Error {
  constructor(cycle: readonly string[]) {
    super(`Theme inheritance cycle detected: ${cycle.join(" -> ")}`);
    this.name = "CoreThemeInheritanceCycleError";
  }
}

/**
 * Error thrown when a theme parent is not found.
 */
export class CoreThemeNotFoundError extends Error {
  constructor(themeId: string) {
    super(`Theme "${themeId}" not found`);
    this.name = "CoreThemeNotFoundError";
  }
}

/**
 * Immutable minimal core theme with light/dark variants and inheritance.
 *
 * A CoreTheme defines minimal color roles (text, background, accent) for
 * both light and dark variants. Themes can inherit from a base theme,
 * with role lookup cascading through the inheritance chain.
 */
export class CoreTheme {
  readonly #id: string;
  readonly #parent: CoreTheme | undefined;
  readonly #colors: Readonly<Partial<{
    light: Partial<CoreThemeVariantColors>;
    dark: Partial<CoreThemeVariantColors>;
  }>>;

  /**
   * Creates a new CoreTheme.
   *
   * @param id - Unique theme identifier
   * @param config - Light/dark variant color configurations
   * @param parent - Optional parent theme for inheritance
   *
   * Throws if variant colors are incomplete and no parent is available
   * to inherit missing roles from.
   */
  constructor(
    id: string,
    config: CoreThemeVariantConfig,
    parent?: CoreTheme,
  ) {
    this.#id = id;
    this.#parent = parent;

    const colors: Partial<{
      light: Partial<CoreThemeVariantColors>;
      dark: Partial<CoreThemeVariantColors>;
    }> = {};

    // Resolve light variant colors if provided, or if parent has light variant
    if (config.light !== undefined || (parent && parent.#colors.light)) {
      colors.light = Object.freeze(
        this.#resolveVariantColors("light", config.light),
      );
    }

    // Resolve dark variant colors if provided, or if parent has dark variant
    if (config.dark !== undefined || (parent && parent.#colors.dark)) {
      colors.dark = Object.freeze(
        this.#resolveVariantColors("dark", config.dark),
      );
    }

    this.#colors = Object.freeze(colors);

    Object.freeze(this);
  }

  /**
   * Resolves colors for a single variant, cascading through inheritance.
   * Returns partial colors - missing roles will throw on access via getRole().
   */
  #resolveVariantColors(
    variant: ThemeVariant,
    provided?: Partial<CoreThemeVariantColors>,
  ): Partial<CoreThemeVariantColors> {
    const resolved: Partial<CoreThemeVariantColors> = {};

    // Add provided roles
    if (provided) {
      for (const role of ["text", "background", "accent"] as const) {
        if (provided[role]) {
          resolved[role] = provided[role]!;
        }
      }
    }

    // Try to inherit missing roles from parent
    if (this.#parent) {
      for (const role of ["text", "background", "accent"] as const) {
        if (!resolved[role]) {
          try {
            resolved[role] = this.#parent.getRole(role, variant);
          } catch {
            // Parent doesn't have this role in this variant, skip
          }
        }
      }
    }

    return resolved;
  }

  /** Returns the theme identifier. */
  id(): string {
    return this.#id;
  }

  /**
   * Returns the parent theme, if any.
   */
  parent(): CoreTheme | undefined {
    return this.#parent;
  }

  /**
   * Looks up a role value for the given variant.
   *
   * @param role - The role to look up
   * @param variant - The variant (light or dark)
   * @returns The hex color value for the role
   *
   * Throws CoreThemeRoleNotFoundError if the role cannot be resolved.
   */
  getRole(role: CoreThemeRole, variant: ThemeVariant): HexColor {
    const colors = this.#colors[variant];
    if (!colors || !(role in colors)) {
      throw new CoreThemeRoleNotFoundError(role, variant, this.#id);
    }
    return colors[role];
  }

  /**
   * Returns all roles for a given variant.
   *
   * @param variant - The variant (light or dark)
   * @returns All defined colors for the variant (may be partial)
   * @throws CoreThemeRoleNotFoundError if the variant is not available
   */
  getVariant(variant: ThemeVariant): Readonly<Partial<CoreThemeVariantColors>> {
    const colors = this.#colors[variant];
    if (!colors) {
      throw new CoreThemeRoleNotFoundError("text", variant, this.#id);
    }
    return colors;
  }

  /**
   * Returns a snapshot of both light and dark variants.
   */
  getAllVariants(): Readonly<
    Partial<{
      light: Partial<CoreThemeVariantColors>;
      dark: Partial<CoreThemeVariantColors>;
    }>
  > {
    return this.#colors;
  }

  /**
   * Validates that no cycles exist in the inheritance chain.
   *
   * @throws CoreThemeInheritanceCycleError if a cycle is detected
   */
  validateNoCycles(): void {
    const visited = new Set<string>();
    const path: string[] = [];

    let current: CoreTheme | undefined = this;
    while (current) {
      if (visited.has(current.#id)) {
        const cycleStart = path.indexOf(current.#id);
        const cycle = [...path.slice(cycleStart), current.#id];
        throw new CoreThemeInheritanceCycleError(cycle);
      }
      visited.add(current.#id);
      path.push(current.#id);
      current = current.#parent;
    }
  }
}

/**
 * Immutable registry for managing multiple CoreThemes.
 */
export class CoreThemeRegistry {
  readonly #themes = new Map<string, CoreTheme>();
  #frozen = false;

  /**
   * Registers one or more themes.
   *
   * @param themes - Themes to register
   * @throws If registry is frozen or theme IDs are duplicated
   */
  register(...themes: readonly CoreTheme[]): void {
    if (this.#frozen) {
      throw new Error("CoreThemeRegistry is frozen");
    }

    for (const theme of themes) {
      if (this.#themes.has(theme.id())) {
        throw new Error(`Theme "${theme.id()}" is already registered`);
      }
      // Validate no cycles
      theme.validateNoCycles();
      this.#themes.set(theme.id(), theme);
    }
  }

  /**
   * Retrieves a theme by ID.
   *
   * @param id - Theme identifier
   * @returns The theme, or undefined if not found
   */
  get(id: string): CoreTheme | undefined {
    return this.#themes.get(id);
  }

  /**
   * Checks if a theme exists.
   */
  has(id: string): boolean {
    return this.#themes.has(id);
  }

  /**
   * Returns all registered theme IDs in sorted order.
   */
  ids(): readonly string[] {
    return Object.freeze(
      [...this.#themes.keys()].sort((a, b) => a.localeCompare(b)),
    );
  }

  /**
   * Freezes the registry, preventing further registrations.
   */
  freeze(): void {
    this.#frozen = true;
    Object.freeze(this);
  }

  /**
   * Returns whether the registry is frozen.
   */
  isFrozen(): boolean {
    return this.#frozen;
  }
}

/**
 * Creates the bootstrap core theme registry with default light/dark themes.
 */
export function createBootstrapCoreRegistry(): CoreThemeRegistry {
  const registry = new CoreThemeRegistry();

  // Create minimal base theme for light variant
  const baseLight = new CoreTheme("core-base-light", {
    light: {
      text: createHexColor("#000000"),
      background: createHexColor("#FFFFFF"),
      accent: createHexColor("#0066CC"),
    },
  });

  // Create minimal base theme for dark variant
  const baseDark = new CoreTheme("core-base-dark", {
    dark: {
      text: createHexColor("#FFFFFF"),
      background: createHexColor("#000000"),
      accent: createHexColor("#66CCFF"),
    },
  });

  // Create light theme variant with inheritance from light base
  const light = new CoreTheme(
    "core-light",
    {
      light: {
        text: createHexColor("#1A1A1A"),
        background: createHexColor("#F5F5F5"),
        accent: createHexColor("#0052A3"),
      },
    },
    baseLight,
  );

  // Create dark theme variant with inheritance from dark base
  const dark = new CoreTheme(
    "core-dark",
    {
      dark: {
        text: createHexColor("#E8E8E8"),
        background: createHexColor("#121212"),
        accent: createHexColor("#4DB8FF"),
      },
    },
    baseDark,
  );

  registry.register(baseLight, baseDark, light, dark);
  registry.freeze();

  return registry;
}
