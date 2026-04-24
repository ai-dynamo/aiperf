// AIPerf dark theme — MCC (Mission Control Console) palette.
// Phosphor amber primary, phosphor cyan secondary, phosphor green for SLO-pass,
// CRT red for failures. Legacy color keys (blue/peach/teal/mauve/...) collapse
// onto this four-color palette so pages that haven't been re-keyed still render
// amber/cyan rather than the pre-MCC blue/orange/teal fan-out.
export const palette = {
  // Base layers — near-black with a warm undertone (no blue tint)
  bg: '#07070a',
  bgCard: '#0c0c11',
  bgRaised: '#111117',

  // Borders
  border: 'rgba(244, 238, 222, 0.11)',
  borderHover: 'rgba(244, 238, 222, 0.18)',
  borderSubtle: 'rgba(244, 238, 222, 0.06)',

  // Text (paper off-white scale)
  dim: 'rgba(244, 238, 222, 0.18)',
  muted: 'rgba(244, 238, 222, 0.38)',
  sub: 'rgba(244, 238, 222, 0.68)',
  text: '#f4eede',
  white: '#ffffff',

  // Accent — phosphor amber is the primary meter-needle color.
  accent: '#76b900',
  accentDim: 'rgba(118, 185, 0, 0.14)',

  // Semantic — primary = amber, secondary = cyan. All other named colors
  // collapse onto the phosphor palette.
  blue: '#76b900',     // legacy key re-pointed to amber (MCC primary)
  cyan: '#7eeaff',
  green: '#9fe870',    // SLO pass only
  amber: '#76b900',
  red: '#ff5964',
  pink: '#76b900',
  orange: '#76b900',
  teal: '#7eeaff',
  indigo: '#7eeaff',
  mauve: '#7eeaff',

  // Compatibility aliases (used by other pages not being rewritten)
  base: '#07070a',
  mantle: '#0c0c11',
  crust: '#000000',
  surface0: 'rgba(244, 238, 222, 0.11)',
  surface1: 'rgba(244, 238, 222, 0.18)',
  surface2: 'rgba(244, 238, 222, 0.18)',
  overlay0: 'rgba(244, 238, 222, 0.38)',
  overlay1: 'rgba(244, 238, 222, 0.68)',
  overlay2: 'rgba(244, 238, 222, 0.68)',
  subtext0: 'rgba(244, 238, 222, 0.68)',
  subtext1: '#f4eede',
  yellow: '#76b900',
  peach: '#76b900',
  maroon: '#ff5964',
  sapphire: '#7eeaff',
  sky: '#7eeaff',
  lavender: '#7eeaff',
  flamingo: '#76b900',
  rosewater: '#76b900',
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

  // Job phase colors — MCC: live = amber (peak/attention), completed = green
  // (pass), failed = red, pending/initializing = paper-faint.
  phaseRunning: palette.amber,
  phaseCompleted: palette.green,
  phaseFailed: palette.red,
  phasePending: 'rgba(244, 238, 222, 0.38)',
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

// Stable model-color assignment — MCC: amber-first, then cyan. Downstream tints
// cycle through the phosphor palette plus red/green for outlier signals.
const MODEL_COLORS = [
  '#76b900', '#7eeaff', '#9fe870', '#ff5964',
  '#8ce200', '#c4a5ff',
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
