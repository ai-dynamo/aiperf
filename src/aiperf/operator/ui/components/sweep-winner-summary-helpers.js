// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Presentation model for the sweep winner card's planner-verdict section.
 *
 * Everything here reads `detail.search_summary`, the operator's projection of
 * the planner's own `search_history.json`. That distinction is the point of the
 * module: the winner, the SLA verdict, and the stopping reason are results the
 * planner ALREADY COMPUTED, and a client that recomputes them from per-variation
 * metrics is running a second implementation of the same rule. Two
 * implementations disagree exactly where it matters -- a constrained metric that
 * was never measured, a threshold missed by a hair, a trial the planner scored
 * but the aggregate rolled up differently. So the card presents the planner's
 * verdict as authoritative and keeps the JS-derived pick only as a fallback for
 * grid sweeps (no planner) and for adaptive sweeps whose trajectory has not been
 * harvested yet.
 *
 * These are pure functions with no `htm` import so they can be exercised
 * directly from Node, matching `live-variations-helpers.js`.
 */

import { formatVariationValues } from '../pages/sweep-detail-helpers.js';

const SLA_OP_SYMBOL = {
  lt: '<',
  le: '<=',
  gt: '>',
  ge: '>=',
};

/**
 * Unit for an objective's metric tag, resolved from the objective itself.
 *
 * The card used to borrow the unit from whichever headline metric the page had
 * matched, and blanked it whenever that metric was not the optimized one. The
 * same sweep then rendered `1,648tok/s` on its default URL and a bare `1,648`
 * after the user clicked an unrelated chart series -- the number lost its unit
 * because of a selection that has nothing to do with the objective.
 *
 * Deliberately keyed on the metric tag alone: an objective is `{metric, stat,
 * direction}` with no unit field, and the tag fully determines the unit for
 * every stat of it. An unmapped tag yields '' -- omitting a unit is honest,
 * inventing one is not.
 */
const UNIT_BY_METRIC_TAG = {
  request_latency: 'ms',
  time_to_first_token: 'ms',
  time_to_second_token: 'ms',
  inter_token_latency: 'ms',
  inter_chunk_latency: 'ms',
  request_throughput: 'req/s',
  output_token_throughput: 'tok/s',
  output_token_throughput_per_user: 'tok/s',
  total_token_throughput: 'tok/s',
  e2e_output_token_throughput: 'tok/s',
  prefill_throughput_per_user: 'tok/s',
};

/** Unit string for a metric tag, or '' when the tag has no known unit. */
export function metricUnit(metricTag) {
  return UNIT_BY_METRIC_TAG[String(metricTag ?? '')] ?? '';
}

/**
 * Human phrasing for each `convergence_reason` the exporter can write.
 *
 * Keys mirror the catalog in `docs/api/search-history.md`. An unmapped reason is
 * rendered verbatim rather than dropped: a stopping reason we cannot phrase is
 * still a stopping reason, and hiding it would put us back where we started.
 */
const CONVERGENCE_REASON_TEXT = {
  max_iterations: 'it used its full iteration budget',
  improvement_patience: 'no iteration improved on the best result for the configured patience window',
  plateau_cv: 'the objective flattened out across the recent window',
  posterior_regret_bound: 'the bound on remaining regret fell below the configured threshold',
  emmr: 'the expected-minimum-model-regret terminator fired',
  monotonic_precision_reached: 'the SLA boundary was bracketed to the target precision',
  monotonic_no_pass_in_range: 'no value in the configured range met the SLA',
  monotonic_no_failure_in_range: 'every value in the configured range met the SLA',
  smooth_isotonic_precision_reached: 'the SLA boundary was bracketed to the target precision',
  smooth_isotonic_cliff_precision_reached: 'the response is discontinuous at the boundary, so the planner reported a bracket instead of a single point',
  smooth_isotonic_no_pass_in_range: 'no value in the configured range met the SLA',
  smooth_isotonic_no_failure_in_range: 'every value in the configured range met the SLA',
  smooth_isotonic_pchip_fallback_bisection: 'the smooth fit failed its prerequisites, so the planner finished by bisection',
};

/** Shorten a dotted config path to its leaf: the prefix is identical everywhere. */
export function leafName(path) {
  const raw = String(path ?? '');
  if (!raw) return '';
  return raw.split('.').pop() || raw;
}

function formatSweptValue(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(3)));
}

/**
 * Describe the swept values of a trial, e.g. `concurrency=17`.
 *
 * Thin alias for `sweep-detail-helpers.formatVariationValues`, which owns every
 * guard this label needs. It used to be a second, unguarded implementation and
 * drifted exactly where the guards matter: `SearchBestTrial.variation_values`
 * is `dict[str, Any]`, so a truncation marker rendered as
 * `__aiperf_truncated__=true, limitBytes=256`, a nested object as
 * `tuning=[object Object]`, a null as `concurrency=null`, and a list as
 * `isl=1,2,3`. This string is the winner card's headline, so each of those was
 * the most prominent text on the page. Delegating keeps one implementation of
 * one concept.
 *
 * The upstream helper also accepts the JSON-string form the CR carries; the
 * planner hands us an already-parsed map, which it passes through unchanged.
 */
export function formatTrialValues(variationValues) {
  return formatVariationValues(variationValues);
}

/**
 * The planner's own winner, or null when there is no planner verdict to show.
 *
 * `feasible` is deliberately `null` rather than `true` when no SLA filter was
 * configured. The exporter defaults each iteration's feasibility to true when
 * nothing constrains it, so rendering that flag unguarded would advertise "meets
 * SLA" for a run that never had one. Callers must treat null as "no claim".
 */
export function plannerVerdict(search) {
  const trials = Array.isArray(search?.best_trials) ? search.best_trials : [];
  if (trials.length === 0) return null;
  const best = trials[0];
  if (!best || typeof best !== 'object') return null;

  const constrained = Number(search?.sla_filter_count ?? 0) > 0;
  const objectives = Array.isArray(search?.objectives) ? search.objectives : [];
  const values = Array.isArray(best.objective_values) ? best.objective_values : [];

  return {
    iterationIdx: Number.isFinite(Number(best.iteration_idx)) ? Number(best.iteration_idx) : null,
    headline: formatTrialValues(best.variation_values),
    variationValues: best.variation_values ?? null,
    // Indexed POSITIONALLY against `objective_values`, which the API preserves
    // at full length with explicit nulls for exactly this reason: an objective
    // the scorer could not produce keeps its slot, so objective N always reads
    // objective N's number. `value: null` therefore means "not measured for
    // this trial" and callers must render it as such -- substituting any other
    // number under this label is the misattribution the alignment protects.
    objectives: objectives.map((objective, index) => ({
      metric: objective?.metric ?? '',
      stat: objective?.stat ?? '',
      label: `${String(objective?.metric ?? '').replace(/_/g, ' ')} ${objective?.stat ?? ''}`.trim(),
      unit: metricUnit(objective?.metric),
      higherIsBetter: (objective?.direction ?? 'maximize') === 'maximize',
      value: typeof values[index] === 'number' && Number.isFinite(values[index]) ? values[index] : null,
    })),
    constrained,
    feasible: constrained ? Boolean(best.feasible) : null,
    feasibleCount: Number(best.feasible_count ?? 0),
    // feasible_count === 0 on a constrained run means NO iteration was both
    // scored and feasible, so the planner ranked the full pool instead. The
    // "winner" is then the least-bad infeasible point. Presenting that as the
    // recommended operating point would advertise a configuration nobody
    // demonstrated is servable, so the card has to say so out loud.
    noFeasiblePoint: constrained && Number(best.feasible_count ?? 0) === 0,
    isFront: trials.length > 1,
    frontSize: trials.length,
    truncated: Boolean(search?.best_trials_truncated),
  };
}

/**
 * Why the search stopped, in a sentence.
 *
 * A run count is uninterpretable without its stopping rule: "14 of 22" reads as
 * "8 runs are missing" when the truth is "the search decided 14 was enough".
 * The variations KPI tile has to hedge with "stopped early" precisely because
 * this reason was not on the API; with it, the card can be specific.
 *
 * `stop_kind` is the operator's classification of a string enum that grows with
 * every new planner, so this switch stays closed even as reasons are added.
 */
export function convergenceNote(search) {
  if (!search) return null;
  const kind = search.stop_kind ?? 'incomplete';
  const reason = search.convergence_reason ?? null;
  const count = Number(search.iteration_count ?? 0);
  const ran = count > 0 ? `${count} iteration${count === 1 ? '' : 's'}` : 'no iterations';

  if (kind === 'incomplete') {
    // Never say "converged" here. A null reason covers cancellation and crash
    // as well as a still-running search, and mislabelling a cancelled sweep as
    // converged is the specific error this field exists to prevent.
    return {
      kind,
      text: `Ran ${ran}. The planner recorded no stopping reason, so this search was cancelled, interrupted, or is still going.`,
    };
  }
  if (kind === 'budget_exhausted') {
    return {
      kind,
      text: `Ran ${ran} and stopped because it used its full iteration budget, not because it converged. A better configuration may lie outside the region it explored.`,
    };
  }
  if (reason === 'unknown') {
    return {
      kind,
      text: `Converged after ${ran}. The planner ended the search but recorded no specific reason.`,
    };
  }
  const phrase = CONVERGENCE_REASON_TEXT[reason];
  return {
    kind,
    text: phrase
      ? `Converged after ${ran} because ${phrase}.`
      : `Converged after ${ran} (${reason}).`,
  };
}

function boundaryEdgeValue(edge) {
  return formatSweptValue(typeof edge?.value === 'number' ? edge.value : NaN);
}

/**
 * The empirical SLA boundary: the highest value that passed and the lowest that
 * did not, plus the constraint that broke first.
 *
 * This is the single most actionable output of a constrained search -- it is the
 * capacity number an operator takes away -- and it is invisible without the
 * artifact. Returns null when the search had no SLA filters, because then every
 * feasibility verdict is vacuously true and `feasible_max` degenerates to
 * "highest value tried", which the variations table already shows and which is
 * not an SLA statement.
 */
export function slaBoundaryNote(search) {
  if (!search) return null;
  if (Number(search.sla_filter_count ?? 0) === 0) return null;
  const boundary = search.boundary_summary;
  if (!boundary || typeof boundary !== 'object') return null;

  const dim = leafName(boundary.swept_dim_path);
  const passValue = boundaryEdgeValue(boundary.feasible_max);
  const failValue = boundaryEdgeValue(boundary.infeasible_min);
  if (passValue == null && failValue == null) return null;

  const breach = boundary.infeasible_min?.first_breach ?? null;
  const op = SLA_OP_SYMBOL[breach?.op] ?? breach?.op ?? '';
  const breachText = breach?.metric_tag
    ? `${String(breach.metric_tag).replace(/_/g, ' ')} ${breach.stat ?? ''} ${op} ${breach.threshold ?? ''}`.replace(/\s+/g, ' ').trim()
    : null;

  return {
    dimension: dim,
    passValue,
    failValue,
    passText: passValue != null ? `${dim} ${passValue}` : null,
    failText: failValue != null ? `${dim} ${failValue}` : null,
    breachText,
    observed: typeof breach?.observed === 'number' && Number.isFinite(breach.observed)
      ? breach.observed
      : null,
    // A bracket is only honest when both edges were actually observed. With one
    // edge the run only proved a lower or upper bound, and claiming a bracket
    // would invent a limit the search never located.
    bracketed: passValue != null && failValue != null,
    // The smooth-isotonic planner sets this when the response is discontinuous
    // at the boundary: it is reporting an honest bracket rather than a point,
    // and a reader who treats the midpoint as the limit will over-provision.
    isCliff: boundary.boundary_type === 'cliff',
    bindingConstraint: boundary.binding_constraint ?? null,
  };
}
