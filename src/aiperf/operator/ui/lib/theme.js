// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// AIPerf dark theme - NVIDIA design system
export const palette = {
  // Base layers (neutral grays, no blue tint)
  bg: '#0b0d0f',
  bgCard: '#121518',
  bgRaised: '#1a1e22',

  // Borders
  border: '#282e34',
  borderHover: '#4a535c',
  borderSubtle: '#1e2328',

  // Text (neutral gray scale)
  dim: '#626a73',
  muted: '#858e97',
  sub: '#adb5bd',
  text: '#d9dde1',
  white: '#f4f6f7',

  // Accent (NVIDIA green)
  accent: '#76b900',
  accentDim: 'rgba(118,185,0,0.15)',

  // Semantic
  blue: '#3b82f6',
  cyan: '#26c6da',
  green: '#76b900',
  amber: '#ffc107',
  red: '#ef5350',
  pink: '#c78372',
  orange: '#c89966',
  teal: '#26c6da',
  indigo: '#668fbe',
  mauve: '#929aa1',

  // Compatibility aliases (used by other pages not being rewritten)
  base: '#0b0d0f',
  mantle: '#121518',
  crust: '#080a0b',
  surface0: '#282e34',
  surface1: '#4a535c',
  surface2: '#4a535c',
  overlay0: '#858e97',
  overlay1: '#adb5bd',
  overlay2: '#adb5bd',
  subtext0: '#adb5bd',
  subtext1: '#d9dde1',
  yellow: '#ffc107',
  peach: '#fb923c',
  maroon: '#ef5350',
  sapphire: '#26c6da',
  sky: '#26c6da',
  lavender: '#76b900',
  flamingo: '#c78372',
  rosewater: '#c78372',
};

// Semantic mappings
export const colors = {
  bg: palette.bg,
  bgAlt: palette.bgCard,
  bgElevated: palette.bgRaised,
  bgRaised: palette.bgRaised,

  border: palette.border,
  borderSubtle: palette.borderSubtle,

  text: palette.text,
  textMuted: palette.sub,
  textDim: palette.muted,

  accent: palette.accent,
  accentAlt: palette.blue,

  success: palette.green,
  warning: palette.amber,
  error: palette.red,
  info: palette.blue,

  // Job phase colors
  phaseRunning: palette.blue,
  phaseCompleted: palette.green,
  phaseFailed: palette.red,
  phasePending: palette.amber,
  phaseUnknown: palette.muted,
};

// Status to color mapping
export function phaseColor(phase) {
  const p = (phase || '').toLowerCase();
  if (p === 'running') return colors.phaseRunning;
  if (p === 'completed' || p === 'succeeded' || p === 'archived') return colors.phaseCompleted;
  if (p === 'failed' || p === 'error') return colors.phaseFailed;
  if (p === 'pending' || p === 'initializing') return colors.phasePending;
  return colors.phaseUnknown;
}

// Stable model-color assignment via string hash
const MODEL_COLORS = [
  palette.blue, '#76b900', palette.amber, palette.pink,
  palette.cyan, palette.teal, palette.orange, palette.indigo,
  palette.red,
];

/**
 * Get a stable color for a model name (hashed).
 * @param {string} model
 * @returns {string}
 */
export function modelColor(model) {
  if (!model) return palette.muted;
  let hash = 0;
  for (let i = 0; i < model.length; i++) {
    hash = ((hash << 5) - hash + model.charCodeAt(i)) | 0;
  }
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length];
}
