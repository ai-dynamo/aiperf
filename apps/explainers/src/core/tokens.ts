/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Visual tokens for explainers chrome and diagram defaults.
 * Keep hex values in sync with `:root` custom properties in `index.css`.
 *
 * Systems-atlas skin: NVIDIA greens on a restrained graphite frame.
 * Layering (dark → light): page < chrome < elevated stage.
 */
export const tokens = {
  text: {
    primary: "#F1F2F4",
    secondary: "#A7AAB4",
    tertiary: "#747985",
    quaternary: "#4F535E",
    link: "#B8D95A",
    onAccent: "#090A0D",
  },
  bg: {
    page: "#08090B",
    chrome: "#0D0E12",
    elevated: "#14151A",
    panel: "#20242A",
  },
  fill: {
    primary: "rgba(241, 242, 244, 0.12)",
    secondary: "rgba(241, 242, 244, 0.075)",
    tertiary: "rgba(241, 242, 244, 0.045)",
    quaternary: "rgba(241, 242, 244, 0.025)",
  },
  stroke: {
    primary: "rgba(218, 221, 230, 0.22)",
    secondary: "rgba(218, 221, 230, 0.11)",
    tertiary: "rgba(218, 221, 230, 0.06)",
  },
  accent: {
    primary: "#76B900",
    control: "#4DB7C5",
  },
  category: {
    green: "#A5C63B",
    yellow: "#D6B84A",
    purple: "#8CCB5E",
    blue: "#669BC4",
    red: "#C95F73",
    orange: "#C9864D",
    cyan: "#4DB7C5",
    gray: "#7E838E",
  },
  radius: {
    control: 999,
    card: 6,
    stage: 5,
    pill: 999,
    box: 5,
  },
  diagram: {
    strokeWidth: 1.6,
    dashed: "6 5",
  },
} as const;

export type Tokens = typeof tokens;
