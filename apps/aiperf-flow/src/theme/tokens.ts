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
