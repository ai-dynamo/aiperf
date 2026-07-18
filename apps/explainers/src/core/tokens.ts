/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Visual tokens for explainers chrome and diagram defaults.
 * Keep hex values in sync with `:root` custom properties in `index.css`.
 *
 * Layering (dark → light): page < chrome/stage < elevated/box.
 * Boxes must sit above the stage fill or they disappear.
 */
export const tokens = {
  text: {
    primary: "#FFFFFF",
    secondary: "#A3A3A3",
    tertiary: "#8A8A8A",
    quaternary: "#6B6B6B",
    link: "#87C3FF",
    onAccent: "#0C0C0C",
  },
  bg: {
    page: "#0C0C0C",
    chrome: "#141416",
    elevated: "#222226",
  },
  fill: {
    primary: "rgba(255, 255, 255, 0.16)",
    secondary: "rgba(255, 255, 255, 0.10)",
    tertiary: "rgba(255, 255, 255, 0.06)",
    quaternary: "rgba(255, 255, 255, 0.04)",
  },
  stroke: {
    primary: "rgba(255, 255, 255, 0.28)",
    secondary: "rgba(255, 255, 255, 0.18)",
    tertiary: "rgba(255, 255, 255, 0.10)",
  },
  accent: {
    primary: "#3FA266",
    control: "#3FA266",
  },
  category: {
    green: "#3FA266",
    yellow: "#F1B467",
    purple: "#9386F2",
    blue: "#599CE7",
    red: "#FC6B83",
    orange: "#F0A060",
    cyan: "#5BC0DE",
    gray: "#8A8A8A",
  },
  radius: {
    control: 10,
    card: 16,
    stage: 16,
    pill: 999,
    box: 14,
  },
  diagram: {
    strokeWidth: 1.6,
    dashed: "6 5",
  },
} as const;

export type Tokens = typeof tokens;
