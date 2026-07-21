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

export function categoryClassName(role: CategoryRole): string {
  return `text-category-${role}`;
}
