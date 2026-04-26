// AIPerf dark theme - NVIDIA design system
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

  // Accent (NVIDIA green)
  accent: '#76b900',
  accentDim: 'rgba(118,185,0,0.15)',

  // Semantic
  blue: '#3b82f6',
  cyan: '#26c6da',
  green: '#76b900',
  amber: '#ffc107',
  red: '#ef5350',
  pink: '#ab47bc',
  orange: '#fb923c',
  teal: '#26c6da',
  indigo: '#6366f1',
  mauve: '#ab47bc',

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
  yellow: '#ffc107',
  peach: '#fb923c',
  maroon: '#ef5350',
  sapphire: '#26c6da',
  sky: '#26c6da',
  lavender: '#76b900',
  flamingo: '#ab47bc',
  rosewater: '#ab47bc',
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
  if (p === 'completed' || p === 'succeeded') return colors.phaseCompleted;
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
