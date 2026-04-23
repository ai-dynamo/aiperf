import { html } from 'htm/preact';
import { Sparkline } from './sparkline.js';
import { useCountUp } from '../lib/hooks.js';

const slugifyLabel = (s) => String(s ?? '').toLowerCase().trim().replace(/\s+/g, '-');

const SPARK_STROKE_GOOD = 'var(--green, #7ccf5e)';
const SPARK_STROKE_BAD = 'var(--red, #ff5c5c)';
const SPARK_STROKE_NEUTRAL = 'var(--paper-faint, rgba(244,240,225,0.36))';

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
 * Metric card — CONSOLE meter-slot aesthetic.
 *
 * Inside a ``.meter-bank`` grid the CSS repaints the tile as a no-chrome
 * slot (hair top edge, amber underline on hover, 44px JetBrains Mono Bold
 * value). Outside the meter-bank (e.g. Job Detail's tile row) the legacy
 * ``.metric-card`` card styling applies; both get the new palette.
 *
 * Optional ``tone`` prop ("amber" | "green" | "red") colors the headline
 * value. If omitted, the value is paper-white by default, unless ``color``
 * is passed (legacy override).
 *
 * Optional ``slo`` prop renders a pass/fail chip next to the label.
 * Optional ``points`` renders an inline SVG sparkline. Optional ``icon``
 * (a Phosphor class like ``"ph-trend-up"``) prefixes the label.
 *
 * Preserves ``data-testid="kpi-<slug>"`` for e2e tests.
 *
 * @param {{
 *   label: string,
 *   value: string|number,
 *   unit?: string,
 *   color?: string,
 *   tone?: 'amber' | 'green' | 'red',
 *   sub?: string,
 *   rawValue?: number,
 *   icon?: string,
 *   slo?: { threshold: number, compare?: 'lt' | 'gt', value?: number },
 *   points?: Array<{t: number, v: number}>,
 * }} props
 */
export function KpiCard({ label, value, unit, color, tone, sub, rawValue, icon, slo, points }) {
  const sloResult = evalSlo(slo, rawValue);
  const sparkStroke =
    sloResult?.kind === 'bad' ? SPARK_STROKE_BAD
    : sloResult?.kind === 'good' ? SPARK_STROKE_GOOD
    : SPARK_STROKE_NEUTRAL;

  // Animate numeric headlines on change. String values pass through unchanged.
  const isInt = typeof value === 'number' && Number.isInteger(value);
  const animated = useCountUp(value, {
    duration: 400,
    formatter: isInt ? (v) => Math.round(v) : (v) => Number(v).toFixed(1),
  });
  const displayValue = typeof value === 'number' ? animated : (value ?? '—');

  // ``tone`` takes precedence over legacy ``color`` (kept for back-compat).
  const toneClass = tone ? ` metric-val--${tone}` : '';
  const valueStyle = color && !tone ? `color: ${color}` : '';

  return html`
    <div
      class="metric-card"
      data-testid=${'kpi-' + slugifyLabel(label)}
    >
      <div class="metric-label-row">
        <span class="metric-label">
          ${icon && html`<i class=${'ph ' + icon} aria-hidden="true"></i>`}
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
        <span class=${'metric-val' + toneClass} style=${valueStyle}>
          ${displayValue}
        </span>
        ${unit && html`<span class="metric-unit">${unit}</span>`}
      </div>
      <${Sparkline} points=${points} stroke=${sparkStroke} />
      ${sub && html`<div class="metric-sub">${sub}</div>`}
    </div>
  `;
}
