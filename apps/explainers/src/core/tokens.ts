/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Visual tokens for explainers chrome and diagram defaults.
 * Keep hex values in sync with `:root` custom properties in `index.css`.
 *
 * Green walkthrough skin: solid green = current, soft green = done.
 * Layering (dark → light): page < chrome/stage < elevated/box.
 */
export const tokens = {
  text: {
    primary: "#F4F4F5",
    secondary: "#A1A1AA",
    tertiary: "#71717A",
    quaternary: "#52525B",
    link: "#86EFAC",
    onAccent: "#052e16",
  },
  bg: {
    page: "#09090B",
    chrome: "#0F0F12",
    elevated: "#18181B",
  },
  fill: {
    primary: "rgba(244, 244, 245, 0.14)",
    secondary: "rgba(244, 244, 245, 0.09)",
    tertiary: "rgba(244, 244, 245, 0.05)",
    quaternary: "rgba(244, 244, 245, 0.03)",
  },
  stroke: {
    primary: "rgba(244, 244, 245, 0.22)",
    secondary: "rgba(244, 244, 245, 0.12)",
    tertiary: "rgba(244, 244, 245, 0.07)",
  },
  accent: {
    primary: "#3FA266",
    control: "#3FA266",
  },
  category: {
    green: "#3FA266",
    yellow: "#EAB308",
    purple: "#A78BFA",
    blue: "#60A5FA",
    red: "#FB7185",
    orange: "#FB923C",
    cyan: "#22D3EE",
    gray: "#8A8A8A",
  },
  radius: {
    control: 10,
    card: 20,
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
