// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// AIPerf Operator — JS-side palette aligned to the dashboard-v2 token system
// in style.css. Used by Chart.js callsites (see lib/chart-theme.js) and any
// inline-style consumer that needs a JS literal for a CSS var.

export const palette = {
  bg: '#0c0c0c',
  bgCard: '#161616',
  bgRaised: '#222222',
  bgTile: '#0f0f0f',
  bgMid: '#1a1a1a',

  border: '#313131',
  borderHover: '#4b4b4b',
  borderSubtle: '#1a1a1a',

  dim: '#4b4b4b',
  muted: '#757575',
  sub: '#a7a7a7',
  text: '#eeeeee',
  white: '#ffffff',

  accent: '#76b900',
  accentHot: '#8ce200',
  accentDeep: '#5a8e00',
  accentDim: 'rgba(118, 185, 0, 0.15)',

  blue:  '#3b82f6',
  cyan:  '#26c6da',
  green: '#76b900',
  amber: '#ffc107',
  red:   '#ef5350',
  pink:  '#ab47bc',
};

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
  error:   palette.red,
  info:    palette.blue,

  phaseRunning:   palette.blue,
  phaseCompleted: palette.green,
  phaseFailed:    palette.red,
  phasePending:   palette.muted,
  phaseUnknown:   palette.dim,
};

export function phaseColor(phase) {
  const p = (phase || '').toLowerCase();
  if (p === 'running')                            return colors.phaseRunning;
  if (p === 'completed' || p === 'succeeded')     return colors.phaseCompleted;
  if (p === 'failed' || p === 'error')            return colors.phaseFailed;
  if (p === 'pending' || p === 'initializing')    return colors.phasePending;
  return colors.phaseUnknown;
}

const MODEL_COLORS = [
  '#76b900', '#3b82f6', '#26c6da', '#9fe870',
  '#ffc107', '#ef5350', '#ab47bc', '#a0d8ff',
];

export function modelColor(model) {
  if (!model) return palette.muted;
  let hash = 0;
  for (let i = 0; i < model.length; i++) {
    hash = ((hash << 5) - hash + model.charCodeAt(i)) | 0;
  }
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length];
}
