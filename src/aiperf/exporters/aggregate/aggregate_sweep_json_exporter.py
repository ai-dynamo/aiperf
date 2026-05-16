# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for sweep aggregate results.

Reads sweep sections from the enclosing
:class:`~aiperf.orchestrator.aggregation.base.AggregateResult` -- schema
preserved byte-compatible with PR #699.
"""

from __future__ import annotations

import orjson

from aiperf.common.finite import scrub_non_finite
from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepJsonExporter(AggregateBaseExporter):
    """Exports sweep aggregate results to JSON format.

    Reads the per-combination sweep sections directly from the enclosing
    :class:`AggregateResult` (carried on ``config.result``):

    - ``result.metadata`` -- carries ``sweep_parameters``, ``num_combinations``,
      ``sweep_mode``, ``best_configurations`` (lifted out into its own top-level
      JSON key), ``pareto_optimal`` (likewise), and other strategy metadata.
    - ``result.metrics`` -- the ``per_combination_metrics`` list produced by
      :meth:`SweepAnalyzer.compute`.

    Output JSON top-level keys (PR #699 schema):

    - ``aggregation_type`` -- always ``"sweep"`` for this exporter.
    - ``num_profile_runs`` -- total number of runs (variations x trials).
    - ``num_successful_runs`` -- runs that completed without error.
    - ``failed_runs`` -- list of ``{"label", "error"}`` dicts.
    - ``metadata`` -- sweep parameter definitions + ``num_combinations`` (with
      ``best_configurations`` / ``pareto_optimal`` / ``trends`` /
      ``per_value_aggregates`` lifted out so they don't double-emit).
    - ``per_combination_metrics`` -- list of ``{parameters, metrics}`` per cell.
    - ``best_configurations`` -- per-objective single-best entry.
    - ``pareto_optimal`` -- list of non-dominated parameter dicts.

    Constructor surface matches sibling aggregate exporters
    (``AggregateConfidenceJsonExporter`` etc.): just ``(config, **kwargs)``.

    Example:
        >>> result = AggregateResult(
        ...     aggregation_type="sweep",
        ...     num_runs=15,
        ...     num_successful_runs=15,
        ...     metadata={"sweep_parameters": [...], "num_combinations": 3,
        ...               "best_configurations": {...}, "pareto_optimal": [...]},
        ...     metrics=[{"parameters": {"concurrency": 10}, "metrics": {...}}],
        ... )  # doctest: +SKIP
        >>> exp = AggregateSweepJsonExporter(AggregateExporterConfig(result, dir))  # doctest: +SKIP
        >>> await exp.export()  # doctest: +SKIP
    """

    _NON_TOP_LEVEL_METADATA_KEYS = frozenset(
        {
            "best_configurations",
            "pareto_optimal",
            "trends",
            "per_value_aggregates",
        }
    )

    def get_file_name(self) -> str:
        """Return ``"profile_export_aiperf_sweep.json"``."""
        return "profile_export_aiperf_sweep.json"

    def _generate_content(self) -> str:
        """Serialize wrapper fields + four sweep sections as indented JSON.

        Numpy scalars are unwrapped by ``OPT_SERIALIZE_NUMPY``; this is
        load-bearing because :class:`SweepAnalyzer` propagates numpy
        ``float64`` values from the underlying ``ConfidenceMetric``.
        """
        # Filter out sections that get their own top-level key so they
        # don't double-emit under "metadata" too.
        metadata = {
            key: value
            for key, value in self._result.metadata.items()
            if key not in self._NON_TOP_LEVEL_METADATA_KEYS
        }

        output = {
            "aggregation_type": self._result.aggregation_type,
            "num_profile_runs": self._result.num_runs,
            "num_successful_runs": self._result.num_successful_runs,
            "failed_runs": self._result.failed_runs or [],
            "metadata": metadata,
            "per_combination_metrics": self._result.metrics or [],
            "best_configurations": self._result.metadata.get("best_configurations", {}),
            "pareto_optimal": self._result.metadata.get("pareto_optimal", []),
        }
        return orjson.dumps(
            scrub_non_finite(output),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        ).decode("utf-8")
