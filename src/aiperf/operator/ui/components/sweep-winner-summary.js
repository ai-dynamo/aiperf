// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';
import { fmtMilliseconds, fmtNumber, fmtReqPerSecond } from '../lib/format.js';
import { convergenceNote, plannerVerdict, slaBoundaryNote } from './sweep-winner-summary-helpers.js';

function formatMetricValue(value, unit) {
  if (unit === 'ms') return fmtMilliseconds(value);
  if (unit === 'req/s') return fmtReqPerSecond(value);
  return fmtNumber(value, 0);
}

function formatCv(cv) {
  return cv == null ? '---' : `${fmtNumber(cv * 100, 1)}%`;
}

const LABEL_STYLE =
  `font-size:var(--font-size-xs);color:${palette.muted};` +
  'text-transform:uppercase;letter-spacing:0.08em';

function Chip({ text, color, title }) {
  return html`
    <span
      class="sweep-verdict-label"
      title=${title ?? undefined}
      style=${
        `--verdict-label-color:${color};`
      }
    >${text}</span>
  `;
}

function EmptyCard({ metricLabel }) {
  return html`
    <section
      class="card"
      data-testid="sweep-winner-summary"
      style="margin-bottom: var(--space-4)"
    >
      <div class="card-title" style="margin:0 0 var(--space-1) 0">Planner verdict</div>
      <div class="text-dim" style="font-size:var(--font-size-sm)">
        No completed variation has a finite ${metricLabel} value yet.
      </div>
    </section>
  `;
}

function VerdictHeader({ metricLabel, direction }) {
  return html`
    <div>
      <div>
        <div class="card-title" style="margin:0">Planner verdict</div>
        <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">
          ${metricLabel} · ${direction}
        </div>
      </div>
    </div>
  `;
}

// The "no feasible point" case must not read like a recommendation. When
// feasible_count is 0 the planner found nothing that satisfied the SLA and fell
// back to ranking the whole pool, so the trial on display is the least-bad
// breach -- not an operating point anyone demonstrated is servable.
function FeasibilityLine({ verdict }) {
  if (!verdict || verdict.feasible === null) return null;
  if (verdict.noFeasiblePoint) {
    return html`
      <div
        class="sweep-verdict-note sweep-verdict-note--error"
        data-testid="sweep-winner-infeasible"
        style="margin-top:var(--space-3);font-size:var(--font-size-sm)"
      >
        <strong>No configuration met the SLA.</strong>
        <span class="text-dim">
          ${' '}Every scored iteration breached at least one constraint, so this is the
          best of the failing points, not a recommended operating point.
        </span>
      </div>
    `;
  }
  const label = verdict.feasible ? 'meets SLA' : 'breaches SLA';
  const color = verdict.feasible ? palette.green : palette.red;
  return html`
    <div data-testid="sweep-winner-feasibility" style="margin-top:var(--space-3);display:flex;gap:var(--space-2);align-items:center;flex-wrap:wrap">
      <${Chip} text=${label} color=${color} />
      <span class="text-dim" style="font-size:var(--font-size-xs)">
        ${verdict.feasibleCount} feasible iteration${verdict.feasibleCount === 1 ? '' : 's'} in this search
      </span>
    </div>
  `;
}

function ConvergenceLine({ note }) {
  if (!note) return null;
  return html`
    <div
      class="sweep-verdict-evidence"
      data-testid="sweep-winner-convergence"
      style="margin-top:var(--space-3)"
    >
      <div style=${LABEL_STYLE}>Why it stopped</div>
      <div class="text-dim" style="font-size:var(--font-size-sm);margin-top:2px">${note.text}</div>
    </div>
  `;
}

// The SLA boundary is the capacity number an operator takes away from a
// constrained search: the highest setting that held and the lowest that did not.
// It exists only in the artifact, so before this it was invisible in the UI.
function BoundaryLine({ boundary }) {
  if (!boundary) return null;
  const headline = boundary.bracketed
    ? `between ${boundary.passText} and ${boundary.failText}`
    : (boundary.passText
      ? `at or above ${boundary.passText} (nothing in range failed)`
      : `at or below ${boundary.failText} (nothing in range passed)`);
  return html`
    <div
      class="sweep-verdict-evidence"
      data-testid="sweep-winner-sla-boundary"
      style="margin-top:var(--space-3)"
    >
      <div style=${LABEL_STYLE}>SLA boundary</div>
      <div style="font-size:var(--font-size-sm);margin-top:2px">
        Highest sustainable ${boundary.dimension} lies ${headline}.
      </div>
      ${boundary.breachText && html`
        <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">
          First breach: ${boundary.breachText}${boundary.observed != null ? `, observed ${fmtNumber(boundary.observed, 1)}` : ''}
        </div>
      `}
      ${boundary.isCliff && html`
        <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">
          The response is discontinuous here, so the planner reports a bracket rather
          than a single boundary value. Do not interpolate inside it.
        </div>
      `}
    </div>
  `;
}

// A multi-objective run scores the winning trial on every objective, but only
// the first one fits the headline slot. The rest are listed here, each against
// its OWN label and unit and each positionally aligned with the API's
// null-preserving `objective_values` -- an objective the scorer could not
// produce reads "not measured" rather than silently inheriting its neighbour's
// number, which is what compacting the vector used to cause.
function SecondaryObjectives({ verdict }) {
  const rest = (verdict?.objectives ?? []).slice(1);
  if (rest.length === 0) return null;
  return html`
    <div data-testid="sweep-winner-objectives" style="margin-top:var(--space-3)">
      <div style=${LABEL_STYLE}>Other objectives</div>
      <div style="display:flex;gap:var(--space-4);flex-wrap:wrap;margin-top:2px">
        ${rest.map(objective => html`
          <div style="font-size:var(--font-size-sm)">
            <span class="text-dim">${objective.label}</span>${' '}
            ${objective.value == null
              ? html`<span style=${`color:${palette.subtext0};font-weight:700`}>not measured</span>`
              : html`<strong>${formatMetricValue(objective.value, objective.unit)}${objective.unit ? ` ${objective.unit}` : ''}</strong>`}
          </div>
        `)}
      </div>
    </div>
  `;
}

function ParetoNote({ verdict }) {
  if (!verdict?.isFront) return null;
  return html`
    <div class="text-dim" data-testid="sweep-winner-pareto" style="font-size:var(--font-size-xs);margin-top:var(--space-2)">
      One of ${verdict.frontSize} non-dominated trials on the Pareto front${verdict.truncated ? ' (list truncated; download search_history.json for the rest)' : ''}.
    </div>
  `;
}

/**
 * Winner card for a sweep.
 *
 * @param {object} props
 * @param {object|null} props.winner  Browser-derived pick from `pickObjectiveWinner` /
 *   `pickSweepWinner`. Used as the headline only when the planner published none.
 * @param {object|null} props.metric  Headline-metric meta (`{key, stat, label, unit}`).
 * @param {object|null} props.search  `detail.search_summary` -- the planner's own
 *   verdict. Authoritative when present.
 */
export function SweepWinnerSummary({ winner, metric, search }) {
  const verdict = plannerVerdict(search);
  const convergence = convergenceNote(search);
  const boundary = slaBoundaryNote(search);
  const metricLabel = metric?.label ?? winner?.metricKey ?? 'selected metric';

  // A search that produced no winning trial can still explain why it stopped,
  // and that is often the most useful thing on the page (an adaptive sweep that
  // was cancelled, or that found nothing feasible). Keep the card alive for it
  // rather than falling back to the bare "nothing yet" placeholder.
  if (!winner && !verdict) {
    if (!convergence) return html`<${EmptyCard} metricLabel=${metricLabel} />`;
    return html`
      <section class="card" data-testid="sweep-winner-summary" style="margin-bottom: var(--space-4)">
        <div class="card-title" style="margin:0">Planner verdict</div>
        <div class="text-dim" style="font-size:var(--font-size-sm);margin-top:var(--space-1)">
          No completed variation has a finite ${metricLabel} value yet.
        </div>
        <${ConvergenceLine} note=${convergence} />
      </section>
    `;
  }

  // Prefer the planner's objective for labelling when it published one; the
  // `metric` prop is whichever headline metric the page matched, and for an
  // adaptive sweep that is only coincidentally the optimized one.
  //
  // Both the label and the unit are resolved from the OBJECTIVE'S OWN metric,
  // never borrowed from the chart. Borrowing made the card's own headline
  // depend on an unrelated click: the same gemma-bo4 sweep rendered
  // "OUTPUT TOK/S | 1,648 tok/s" by default and
  // "OUTPUT TOKEN THROUGHPUT AVG | 1,648" under ?metric=inter_token_latency.avg
  // -- identical number, unit gone (and formatMetricValue silently dropped to
  // 0 decimals with it), because the reader had selected a different series.
  // The objective is a property of the sweep, so its caption cannot be a
  // function of the selector.
  const plannerObjective = verdict?.objectives?.[0] ?? null;
  const displayLabel = plannerObjective ? plannerObjective.label : metricLabel;
  const unit = (plannerObjective ? plannerObjective.unit : metric?.unit) ?? '';
  const higherIsBetter = plannerObjective ? plannerObjective.higherIsBetter : winner?.higherIsBetter;
  const direction = higherIsBetter ? 'higher is better' : 'lower is better';

  // Lead with what the variation actually tried ("concurrency=17") rather than
  // the planner's cell id ("search_iter_0008"), which is meaningless to a
  // reader. The id drops to the subtitle so it stays copyable for artifact
  // paths. Falls back to the id when no values were recorded.
  const fallbackLabel = winner ? (winner.label || `v${winner.variation_index}`) : null;
  const headline = verdict?.headline ?? winner?.valuesLabel ?? fallbackLabel ?? '---';
  // A null objective value means the scorer produced nothing for THIS objective
  // on the winning trial. Falling through to `winner.mean` would print a
  // different metric's number under the objective's label, which is the same
  // misattribution the API's null-preserving `objective_values` alignment
  // exists to prevent -- so say "not measured" instead of borrowing a number.
  const objectiveUnmeasured = Boolean(plannerObjective) && plannerObjective.value == null;
  const displayValue = plannerObjective ? plannerObjective.value : (winner?.mean ?? null);
  const variationIndex = verdict?.iterationIdx ?? winner?.variation_index ?? null;

  // Every identity field must come from ONE source. Mixing them shipped a card
  // whose headline and index were the planner's winning trial while the cell id
  // beside them belonged to a different variation entirely: on an archive whose
  // specSummary predates `objectives`/`sla_filters`, pickObjectiveWinner returns
  // null and `winner` degrades to the raw metric maximum -- the SLA-infeasible
  // peak. The card then read "concurrency=17 / search_iter_0005 / variation 8",
  // splicing the true winner's values onto a losing run's artifact path, so
  // following that id fetches the wrong run's results.
  //
  // The planner labels iteration N as `search_iter_%04d` (optuna_planner.py:226),
  // so when the verdict is authoritative the id is derived from ITS index rather
  // than borrowed from the card's own pick.
  const verdictCellId = verdict && Number.isFinite(verdict.iterationIdx)
    ? `search_iter_${String(verdict.iterationIdx).padStart(4, '0')}`
    : null;
  const subtitleId = verdict ? verdictCellId : (winner?.valuesLabel ? fallbackLabel : null);

  return html`
    <section
      class="card"
      data-testid="sweep-winner-summary"
      style=${
        `margin-bottom:var(--space-4);` +
        `border-color:${palette.peach}55;` +
        `background:linear-gradient(135deg, ${palette.peach}12, ${palette.bgCard} 44%);`
      }
    >
      <${VerdictHeader} metricLabel=${displayLabel} direction=${direction} />
      <div style="display:flex;align-items:end;justify-content:space-between;gap:var(--space-4);flex-wrap:wrap;margin-top:var(--space-3)">
        <div>
          <div style=${LABEL_STYLE}>Variation</div>
          <div
            style="font-size:var(--font-size-lg);font-weight:800;margin-top:2px"
            data-testid="sweep-winner-headline"
          >${headline}</div>
          <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">
            ${subtitleId ? `${subtitleId} · ` : ''}variation ${variationIndex ?? '---'}
          </div>
        </div>
        <div style="text-align:right">
          <div style=${LABEL_STYLE}>${displayLabel}</div>
          ${objectiveUnmeasured
            ? html`
              <div
                data-testid="sweep-winner-unmeasured"
                style=${`font-size:var(--font-size-sm);font-weight:700;line-height:1.1;color:${palette.subtext0}`}
              >not measured</div>
              <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:4px">
                The planner recorded no value for this objective on the winning trial.
              </div>
            `
            : html`
              <div style="font-size: var(--font-size-2xl);font-weight:850;line-height:1.1;font-variant-numeric:tabular-nums">
                ${formatMetricValue(displayValue, unit)}${unit ? html`<span style="font-size:var(--font-size-sm);font-weight:700;margin-left:6px;color:${palette.subtext0}">${unit}</span>` : null}
              </div>
            `}
          ${!verdict && winner && html`
            <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:4px">
              CV ${formatCv(winner.cv)} · n ${winner.n ?? '---'}
            </div>
          `}
        </div>
      </div>
      <${SecondaryObjectives} verdict=${verdict} />
      <${ParetoNote} verdict=${verdict} />
      <${FeasibilityLine} verdict=${verdict} />
      <${ConvergenceLine} note=${convergence} />
      <${BoundaryLine} boundary=${boundary} />
    </section>
  `;
}
