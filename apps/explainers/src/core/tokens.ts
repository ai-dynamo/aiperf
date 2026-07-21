/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Visual tokens for explainers chrome and diagram defaults.
 * Keep hex values in sync with `:root` custom properties in `index.css`.
 *
 * NVIDIA-deck skin: flat white slides, NVIDIA green accent, boxy corners.
 * Matches the reference deck's editorial print language, not a dark app chrome.
 */
export const tokens = {
  text: {
    primary: "#000000",
    secondary: "#555555",
    tertiary: "#A7A7A7",
    quaternary: "#C4C4C4",
    link: "#3D6B00",
    onAccent: "#000000",
  },
  bg: {
    page: "#FFFFFF",
    chrome: "#FAFAFA",
    elevated: "#FFFFFF",
    panel: "#F7F7F7",
  },
  fill: {
    primary: "rgba(15, 12, 8, 0.08)",
    secondary: "rgba(15, 12, 8, 0.05)",
    tertiary: "rgba(15, 12, 8, 0.03)",
    quaternary: "rgba(15, 12, 8, 0.015)",
  },
  stroke: {
    primary: "#111111",
    secondary: "#E4E4E4",
    tertiary: "#EFEFEF",
  },
  accent: {
    primary: "#76B900",
    tint: "#F2F7EA",
    control: "#3987A6",
  },
  category: {
    green: "#5E8A1F",
    yellow: "#B08A1E",
    purple: "#6E4FA6",
    blue: "#2A78D6",
    red: "#A63244",
    orange: "#B05E2A",
    cyan: "#3987A6",
    gray: "#7E838E",
  },
  radius: {
    control: 0,
    card: 0,
    stage: 0,
    pill: 0,
    box: 0,
  },
  diagram: {
    strokeWidth: 4.32,
    dashed: "16.2 13.5",
  },
} as const;

export type Tokens = typeof tokens;
