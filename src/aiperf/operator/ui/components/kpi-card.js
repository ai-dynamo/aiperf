import { html } from 'htm/preact';
import { Sparkline } from './sparkline.js';
import { useCountUp } from '../lib/hooks.js';

const slugifyLabel = (s) => String(s ?? '').toLowerCase().trim().replace(/\s+/g, '-');

// Stroke colors keyed to SLO status. Neutral dim when no SLO is declared
// (or the value isn't numeric) so the line stays visible but uncoloured.
const SPARK_STROKE_GOOD = 'var(--green, #40a02b)';
const SPARK_STROKE_BAD = 'var(--red, #d20f39)';
const SPARK_STROKE_NEUTRAL = 'var(--muted, #7f849c)';

/** Evaluate an optional SLO against the tile's headline value.
 *
 *  ``slo`` shape: ``{ threshold, compare?, value? }``.
 *  - ``compare``: ``'lt'`` (default, lower is better) or ``'gt'`` (higher is better).
 *  - ``value``: numeric override; if omitted we try to parse ``value`` prop.
 *
 *  Returns ``null`` when the SLO is undeclared or numeric value is unknown —
 *  callers don't render a chip in that case.
 */
function evalSlo(slo, rawValue) {
  if (!slo || slo.threshold == null) return null;
  const probe = typeof slo.value === 'number' ? slo.value : rawValue;
  if (probe == null || !isFinite(probe)) return null;
  const compare = slo.compare === 'gt' ? 'gt' : 'lt';
  const ok = compare === 'lt' ? probe <= slo.threshold : probe >= slo.threshold;
  const prefix = compare === 'lt' ? '≤ ' : '≥ ';
  return {
    kind: ok ? 'good' : 'bad',
    icon: ok ? '✓' : '✗',
    label: prefix + slo.threshold,
  };
}

/**
 * Metric card — simple card, brand-colored value for key metrics.
 *
 * Optional ``slo`` prop renders a small green/red chip next to the label with
 * the threshold (e.g. ``✓ ≤ 500``). When undeclared, no chip renders.
 *
 * Optional ``points`` prop (chronological ``{t, v}`` samples) renders an
 * inline SVG sparkline between the headline value and the subtitle. Stroke
 * color follows SLO status: green when pass, red when fail, neutral when
 * the SLO is undeclared. The Sparkline component renders a stable placeholder
 * when fewer than 2 samples are available, so callers can always pass the prop.
 *
 * Optional ``icon`` prop (a Phosphor class name like ``"ph-trend-up"``) renders
 * a small tertiary-colored glyph in the label row.
 *
 * When ``value`` is numeric, the headline is gently animated via
 * :func:`useCountUp` on change. String values (``"---"``, pre-formatted
 * durations, etc.) pass through unchanged.
 *
 * @param {{
 *   label: string,
 *   value: string|number,
 *   unit?: string,
 *   color?: string,
 *   sub?: string,
 *   rawValue?: number,
 *   icon?: string,
 *   slo?: { threshold: number, compare?: 'lt' | 'gt', value?: number },
 *   points?: Array<{t: number, v: number}>,
 * }} props
 */
export function KpiCard({ label, value, unit, color, sub, rawValue, icon, slo, points }) {
  const valueStyle = color ? `color: ${color}` : '';
  const sloResult = evalSlo(slo, rawValue);
  const sparkStroke =
    sloResult?.kind === 'bad' ? SPARK_STROKE_BAD
    : sloResult?.kind === 'good' ? SPARK_STROKE_GOOD
    : SPARK_STROKE_NEUTRAL;

  // Animate the headline when it's numeric. Integer targets render as
  // rounded integers; fractional targets keep one decimal place so a ramp
  // like 0 -> 42.7 doesn't jitter between integer frames.
  const isInt = typeof value === 'number' && Number.isInteger(value);
  const animated = useCountUp(value, {
    duration: 400,
    formatter: isInt ? (v) => Math.round(v) : (v) => Number(v).toFixed(1),
  });
  const displayValue = typeof value === 'number' ? animated : (value ?? '—');

  return html`
    <div class="metric-card" data-testid=${'kpi-' + slugifyLabel(label)}>
      <div class="metric-label-row">
        <span class="metric-label">
          ${icon && html`<i class=${'ph ' + icon} aria-hidden="true" style="color: var(--text-tertiary); margin-right: var(--space-1)"></i>`}
          ${label}
        </span>
        ${sloResult && html`
          <span
            class=${'kpi-chip kpi-chip--' + sloResult.kind}
            data-testid=${'kpi-slo-' + slugifyLabel(label)}
          >
            <span>${sloResult.icon}</span>
            <span class="kpi-chip-thresh">${sloResult.label}</span>
          </span>
        `}
      </div>
      <div class="metric-val-row">
        <span class="metric-val" style=${valueStyle}>
          ${displayValue}
        </span>
        ${unit && html`<span class="metric-unit">${unit}</span>`}
      </div>
      <${Sparkline} points=${points} stroke=${sparkStroke} />
      ${sub && html`<div class="metric-sub">${sub}</div>`}
    </div>
  `;
}
