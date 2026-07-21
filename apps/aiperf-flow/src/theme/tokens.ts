/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Semantic-role -> Tailwind class helpers, mirroring the `@theme` tokens in `index.css`.

export type SurfaceRole = "page" | "chrome" | "elevated" | "panel";
export type InkRole = "primary" | "secondary" | "tertiary" | "quaternary" | "link";
export type StrokeRole = "primary" | "secondary" | "tertiary";
export type AccentRole = "primary" | "tint" | "control";
export type CategoryRole =
  | "green"
  | "yellow"
  | "purple"
  | "blue"
  | "red"
  | "orange"
  | "cyan"
  | "gray";

export function surfaceClassName(role: SurfaceRole): string {
  return `bg-surface-${role}`;
}

export function inkClassName(role: InkRole): string {
  return `text-ink-${role}`;
}

export function strokeClassName(role: StrokeRole): string {
  return `border-stroke-${role}`;
}

export function accentClassName(role: AccentRole): string {
  return `text-accent-${role}`;
}

// Tailwind's compiler only picks up classes it can see as literal strings in source, so a
// dynamically interpolated `bg-category-${role}` would be purged for any role whose exact
// string doesn't happen to appear verbatim elsewhere. These tables keep every supported role
// visible as a whole literal string (mirrors `GRID_COLS_CLASSES` in `layout/Grid.tsx`).
const CATEGORY_TEXT_CLASSES: Record<CategoryRole, string> = {
  green: "text-category-green",
  yellow: "text-category-yellow",
  purple: "text-category-purple",
  blue: "text-category-blue",
  red: "text-category-red",
  orange: "text-category-orange",
  cyan: "text-category-cyan",
  gray: "text-category-gray",
};

const CATEGORY_BG_CLASSES: Record<CategoryRole, string> = {
  green: "bg-category-green",
  yellow: "bg-category-yellow",
  purple: "bg-category-purple",
  blue: "bg-category-blue",
  red: "bg-category-red",
  orange: "bg-category-orange",
  cyan: "bg-category-cyan",
  gray: "bg-category-gray",
};

const CATEGORY_BG_TINT_CLASSES: Record<CategoryRole, string> = {
  green: "bg-category-green/10",
  yellow: "bg-category-yellow/10",
  purple: "bg-category-purple/10",
  blue: "bg-category-blue/10",
  red: "bg-category-red/10",
  orange: "bg-category-orange/10",
  cyan: "bg-category-cyan/10",
  gray: "bg-category-gray/10",
};

// SVG shapes (<rect>/<path>/<circle>/...) are painted by the CSS `fill`/`stroke` properties —
// `background-color`/`border-color` (what `categoryBgClassName`/`strokeClassName` emit) have NO
// effect on them. This bug shipped three times in this codebase (Chart.tsx, the
// aiperf-metrics-accumulator sweep-line chart, the weka-timing-transforms-interactive lane
// boxes) before being traced to exactly this mismatch — always reach for these two helpers,
// never `categoryBgClassName`, when coloring a hand-drawn SVG element. See "SVG shapes need
// fill-/stroke- classes" in SKILL.md.
const CATEGORY_FILL_CLASSES: Record<CategoryRole, string> = {
  green: "fill-category-green",
  yellow: "fill-category-yellow",
  purple: "fill-category-purple",
  blue: "fill-category-blue",
  red: "fill-category-red",
  orange: "fill-category-orange",
  cyan: "fill-category-cyan",
  gray: "fill-category-gray",
};

const CATEGORY_STROKE_CLASSES: Record<CategoryRole, string> = {
  green: "stroke-category-green",
  yellow: "stroke-category-yellow",
  purple: "stroke-category-purple",
  blue: "stroke-category-blue",
  red: "stroke-category-red",
  orange: "stroke-category-orange",
  cyan: "stroke-category-cyan",
  gray: "stroke-category-gray",
};

export function categoryClassName(role: CategoryRole): string {
  return CATEGORY_TEXT_CLASSES[role];
}

export function categoryBgClassName(role: CategoryRole): string {
  return CATEGORY_BG_CLASSES[role];
}

/** 10%-opacity variant of `categoryBgClassName`, used for subtle background tints. */
export function categoryBgTintClassName(role: CategoryRole): string {
  return CATEGORY_BG_TINT_CLASSES[role];
}

/**
 * `fill-category-*` for coloring an SVG shape's fill. Use this — never `categoryBgClassName` —
 * on any `<rect>`/`<path>`/`<circle>`/`<polygon>`/`<ellipse>`/`<polyline>`. No `fill="currentColor"`
 * attribute needed; this class sets the `fill` CSS property directly.
 */
export function categoryFillClassName(role: CategoryRole): string {
  return CATEGORY_FILL_CLASSES[role];
}

/**
 * `stroke-category-*` for coloring an SVG shape's stroke. Use this — never `strokeClassName`
 * (which emits a `border-*` class, also inert on SVG) — on any SVG element's `stroke`. No
 * `stroke="currentColor"` attribute needed; this class sets the `stroke` CSS property directly.
 */
export function categoryStrokeClassName(role: CategoryRole): string {
  return CATEGORY_STROKE_CLASSES[role];
}
