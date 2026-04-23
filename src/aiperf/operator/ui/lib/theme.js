// AIPerf dark theme - CONSOLE palette (amber-dominant, cyan secondary).
// Legacy color keys (blue/peach/teal/mauve/...) collapse onto the two-tone
// CONSOLE palette so pages that haven't been re-keyed still render amber/cyan
// rather than the pre-CONSOLE blue/orange/teal fan-out.
export const palette = {
  // Base layers (neutral grays, no blue tint)
  bg: '#0c0c0c',
  bgCard: '#161616',
  bgRaised: '#222222',

  // Borders
  border: '#313131',
  borderHover: '#4b4b4b',
  borderSubtle: '#1a1a1a',

  // Text (neutral gray scale)
  dim: '#4b4b4b',
  muted: '#757575',
  sub: '#a7a7a7',
  text: '#eeeeee',
  white: '#ffffff',

  // Accent — CONSOLE amber replaces NVIDIA green as the primary accent.
  accent: '#ff9f1c',
  accentDim: 'rgba(255,159,28,0.15)',

  // Semantic — primary = amber, secondary = cyan. All other named colors
  // collapse onto the same two-tone palette.
  blue: '#ff9f1c',     // formerly #3b82f6 — now amber (CONSOLE primary)
  cyan: '#3ad8e3',     // CONSOLE secondary data signal
  green: '#7ccf5e',    // SLO pass only
  amber: '#ff9f1c',
  red: '#ff5c5c',
  pink: '#ff9f1c',
  orange: '#ff9f1c',
  teal: '#3ad8e3',
  indigo: '#3ad8e3',
  mauve: '#3ad8e3',

  // Compatibility aliases (used by other pages not being rewritten)
  base: '#0c0c0c',
  mantle: '#161616',
  crust: '#080808',
  surface0: '#313131',
  surface1: '#4b4b4b',
  surface2: '#4b4b4b',
  overlay0: '#757575',
  overlay1: '#a7a7a7',
  overlay2: '#a7a7a7',
  subtext0: '#a7a7a7',
  subtext1: '#eeeeee',
  yellow: '#ff9f1c',
  peach: '#ff9f1c',
  maroon: '#ff5c5c',
  sapphire: '#3ad8e3',
  sky: '#3ad8e3',
  lavender: '#3ad8e3',
  flamingo: '#ff9f1c',
  rosewater: '#ff9f1c',
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

  // Job phase colors — CONSOLE: live = amber (peak/attention),
  // completed = green (pass), failed = red, pending/initializing = paper-faint.
  phaseRunning: palette.amber,
  phaseCompleted: palette.green,
  phaseFailed: palette.red,
  phasePending: 'rgba(244, 240, 225, 0.36)',
  phaseUnknown: palette.muted,
};

// Status to color mapping
export function phaseColor(phase) {
  const p = (phase || '').toLowerCase();
  if (p === 'running') return colors.phaseRunning;
  if (p === 'completed' || p === 'succeeded') return colors.phaseCompleted;
  if (p === 'failed' || p === 'error') return colors.phaseFailed;
  if (p === 'pending' || p === 'initializing') return colors.phasePending;
  return colors.phaseUnknown;
}

// Stable model-color assignment — CONSOLE: amber-first, then cyan.
// Downstream tints cycle through the two-tone palette plus red/green for
// outlier signals; no blue, no purple.
const MODEL_COLORS = [
  '#ff9f1c', '#3ad8e3', '#7ccf5e', '#ff5c5c',
  '#ffb547', '#f4f0e1',
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
