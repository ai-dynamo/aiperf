# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the ``sla_breach_knee`` post-process handler.

Re-exported from ``aiperf.search_recipes.post_process`` -- consumers should
import :class:`SLABreachKnee` from there to match the other built-in handlers.
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = ["SLABreachKnee"]


def _sla_filter_module() -> Any:
    """Lazy accessor for ``sweep_sla_filter`` helpers.

    Import deferred: ``aiperf.orchestrator.aggregation.sweep_sla_filter``
    transitively pulls ``aiperf.config``, but ``aiperf.config.__init__`` ->
    ``_models_benchmark`` imports ``aiperf.search_recipes._base`` at module
    top, so importing this module during config-package init would
    partially-initialize and cycle. A method-body import side-steps the
    timing issue without restructuring the package graph. Centralizing the
    deferred import here keeps both call sites in lock-step if the symbol
    set changes.
    """
    from aiperf.orchestrator.aggregation import sweep_sla_filter

    return sweep_sla_filter


class SLABreachKnee:
    """Locate the SLA-feasibility boundary along a single swept parameter.

    Walks ``per_combination_metrics`` in swept-value order, evaluates every
    ``sla_filter`` per row, and reports ``max_passing_<leaf>`` (largest
    feasible value) and ``first_failing_<leaf>`` (smallest infeasible value).
    The leaf is the swept_param's last dotted segment (e.g. ``concurrency``
    from ``phases.profiling.concurrency``), mirroring
    :class:`DegradationKneeDetect`.

    ``first_failing_breach`` reports the FIRST filter (input order) that fails
    at ``first_failing_<leaf>``; ``observed`` is ``null`` when the metric was
    missing from the row (treated as infeasible -- same policy as
    :func:`aiperf.orchestrator.aggregation.sweep_sla_filter.passes_filter`).
    ``monotonicity_check`` is ``false`` when any feasible point appears at a
    higher swept value than an infeasible point -- flagged as signal (usually
    SUT instability), never raised as an error.

    Required ``params`` keys:

    - ``sla_filters`` (list[SLAFilter | dict]): filters echoed from the recipe.
    - ``swept_param`` (str): dotted-path swept on the axis, e.g.
      ``"phases.profiling.concurrency"``.

    Returns:
        A dict with the following fields:

        - ``swept_param`` (str): echoes the input dotted-path.
        - ``max_passing_<leaf>`` (Any | None): largest swept value satisfying
          every filter; ``None`` when no row is feasible. The value type is
          preserved from the input (int stays int, float stays float).
        - ``first_failing_<leaf>`` (Any | None): smallest swept value failing
          any filter; ``None`` when every row is feasible. Type preserved.
        - ``first_failing_breach`` (dict | None): first-filter-order breach
          record at ``first_failing_<leaf>``; ``None`` when no row fails.
          Carries ``metric_tag``, ``stat``, ``op``, ``threshold``, and
          ``observed`` (``None`` when the metric was missing).
        - ``all_points`` (list[dict]): per-row records keyed by ``<leaf>``,
          each with ``feasible`` (bool) and ``breaches`` (list of breach
          records, empty when feasible).
        - ``monotonicity_check`` (bool): ``False`` when a feasible point
          follows an infeasible point in ascending swept order; ``True`` for
          empty input (vacuous).
        - ``filters`` (list[dict]): input filters serialized via
          ``sla_filter_to_dict`` for stable JSON output.

    Example:
        >>> agg = {"per_combination_metrics": [
        ...     {"parameters": {"phases.profiling.concurrency": 256},
        ...      "metrics": {"time_to_first_token_p95": {"mean": 150.0}}},
        ...     {"parameters": {"phases.profiling.concurrency": 384},
        ...      "metrics": {"time_to_first_token_p95": {"mean": 213.4}}},
        ... ]}
        >>> from aiperf.config.sweep.adaptive import SLAFilter
        >>> out = SLABreachKnee().process(agg, {
        ...     "sla_filters": [SLAFilter(metric_tag="time_to_first_token",
        ...         stat="p95", op="lt", threshold=200.0)],
        ...     "swept_param": "phases.profiling.concurrency",
        ... })
        >>> out["max_passing_concurrency"], out["first_failing_concurrency"]
        (256, 384)
    """

    name: ClassVar[str] = "sla_breach_knee"
    description: ClassVar[str] = (
        "Locate the SLA feasibility boundary along a swept parameter and "
        "emit max_passing / first_failing values plus per-point breach detail."
    )

    def process(
        self,
        sweep_aggregate: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Return max-passing and first-failing SLA boundary artifacts."""
        mod = _sla_filter_module()

        swept_param = str(params["swept_param"])
        sla_filters = list(params["sla_filters"])
        leaf = swept_param.split(".")[-1]

        # Per-combination ``parameters`` dicts can use either the leaf name
        # (``concurrency``) or the full dotted path; recipes pass the full
        # dotted path via ``swept_param``. Accept either form so this handler
        # doesn't break depending on where the row originated.
        param_key: str
        rows = sweep_aggregate.get("per_combination_metrics") or []
        if rows and swept_param in (rows[0].get("parameters") or {}):
            param_key = swept_param
        else:
            param_key = leaf
        rows = [r for r in rows if param_key in (r.get("parameters") or {})]
        # Sort using float coercion as the key only; the original value type
        # (int vs float) is preserved when emitted into all_points / summary
        # fields so downstream JSON keeps `256` rather than `256.0`.
        rows.sort(key=lambda r: float(r["parameters"][param_key]))

        # Why: when retried trials at the same swept value disagree (one
        # passes the SLA, the other fails), naive walk-and-emit leaks the
        # value into BOTH `max_passing_<leaf>` and `first_failing_<leaf>`,
        # producing a zero-width bracket and a spuriously-True
        # monotonicity_check. Collapse same-x rows up front using
        # all-pass-required: the row group is feasible only if EVERY trial at
        # that x passed every filter, otherwise infeasible. This is the
        # defensible rule because (a) the user is asking "is this concurrency
        # safe to ship", and one failing trial means "no", and (b) it makes
        # `max_passing < first_failing` strictly disjoint by construction.
        all_points: list[dict[str, Any]] = _collapse_same_x_rows(
            rows, param_key, leaf=leaf, sla_filters=sla_filters, mod=mod
        )

        feasible_points = [p for p in all_points if p["feasible"]]
        infeasible_points = [p for p in all_points if not p["feasible"]]
        max_passing = (
            max(feasible_points, key=lambda p: float(p[leaf]))[leaf]
            if feasible_points
            else None
        )
        first_failing_point = (
            min(infeasible_points, key=lambda p: float(p[leaf]))
            if infeasible_points
            else None
        )
        first_failing = first_failing_point[leaf] if first_failing_point else None
        first_failing_breach = (
            first_failing_point["breaches"][0] if first_failing_point else None
        )

        # Monotonicity: walking ascending swept order, no feasible may follow
        # an infeasible. A False flag here is informational, never an error.
        seen_infeasible = False
        monotonic = True
        for point in all_points:
            if not point["feasible"]:
                seen_infeasible = True
            elif seen_infeasible:
                monotonic = False
                break

        return {
            "swept_param": swept_param,
            f"max_passing_{leaf}": max_passing,
            f"first_failing_{leaf}": first_failing,
            "first_failing_breach": first_failing_breach,
            "all_points": all_points,
            "monotonicity_check": monotonic,
            "filters": [mod.sla_filter_to_dict(f) for f in sla_filters],
        }


def _collapse_same_x_rows(
    rows: list[dict[str, Any]],
    swept_param: str,
    *,
    leaf: str,
    sla_filters: list[Any],
    mod: Any,
) -> list[dict[str, Any]]:
    """Group rows by swept-x and collapse to one point per x with all-pass-required.

    Each input row is evaluated independently; rows sharing the same swept
    value are then merged into a single emitted point. The merged point is
    feasible iff EVERY underlying row was feasible (no breaches anywhere).
    Breach records are concatenated in input order so the consumer can still
    audit which trial(s) failed which filter.

    Why: see :class:`SLABreachKnee.process` rationale. This function exists
    only to make the collapse step easy to test in isolation and to keep
    process() readable.
    """
    # Preserve first-seen order of distinct x values (rows are already sorted
    # ascending by float(x), so the dict iteration order matches that).
    grouped: dict[float, dict[str, Any]] = {}
    for row in rows:
        raw_value = row["parameters"][swept_param]
        x_key = float(raw_value)
        metrics = row.get("metrics") or {}
        row_breaches = _evaluate_breaches(metrics, sla_filters, mod)
        if x_key not in grouped:
            grouped[x_key] = {
                leaf: raw_value,
                "feasible": not row_breaches,
                "breaches": list(row_breaches),
            }
        else:
            existing = grouped[x_key]
            existing["feasible"] = existing["feasible"] and not row_breaches
            existing["breaches"].extend(row_breaches)
    return list(grouped.values())


def _evaluate_breaches(
    metrics: dict[str, Any],
    sla_filters: list[Any],
    mod: Any,
) -> list[dict[str, Any]]:
    """Return per-filter breach records for one row, preserving filter order."""
    breaches: list[dict[str, Any]] = []
    for f in sla_filters:
        metric_tag = f["metric_tag"] if isinstance(f, dict) else f.metric_tag
        stat = f["stat"] if isinstance(f, dict) else f.stat
        op = f["op"] if isinstance(f, dict) else f.op
        threshold = float(f["threshold"] if isinstance(f, dict) else f.threshold)
        observed = mod.read_metric_value(metrics, metric_tag, stat)
        passed = observed is not None and bool(mod.OP_TO_FN[op](observed, threshold))
        if not passed:
            breaches.append(
                {
                    "metric_tag": metric_tag,
                    "stat": stat,
                    "op": op,
                    "threshold": threshold,
                    "observed": observed,
                }
            )
    return breaches
