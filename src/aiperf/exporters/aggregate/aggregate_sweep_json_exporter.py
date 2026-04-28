# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for sweep aggregate results."""

import orjson

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepJsonExporter(AggregateBaseExporter):
    """Exports sweep aggregate results to JSON format.

    The sweep aggregate contains:
    - metadata: Parameter name, values, counts
    - per_combination_metrics: Metrics for each parameter combination
    - best_configurations: Best values for key metrics
    - pareto_optimal: List of Pareto optimal parameter combinations

    Design:
    - Uses the dict returned by SweepAnalyzer.compute()
    - Serializes directly to JSON (no Pydantic models needed)
    - Ensures consistency with confidence aggregate format
    """

    def get_file_name(self) -> str:
        """Return JSON file name.

        Returns:
            str: "profile_export_aiperf_sweep.json"
        """
        return "profile_export_aiperf_sweep.json"

    def _generate_content(self) -> str:
        """Generate JSON content from sweep aggregate result.

        The result contains:
        - result.metadata: Contains sweep metadata + best_configurations, pareto_optimal
        - result.metrics: Contains per_combination_metrics (the actual metrics dict)

        Output structure:
        {
            "aggregation_type": "sweep",
            "num_profile_runs": 15,
            "num_successful_runs": 15,
            "failed_runs": [],
            "metadata": {...},
            "per_combination_metrics": {...},
            "best_configurations": {...},
            "pareto_optimal": [...]
        }

        Returns:
            str: JSON content string
        """
        # Build the output structure
        output = {}

        # Add AggregateResult fields at top level
        output["aggregation_type"] = self._result.aggregation_type
        output["num_profile_runs"] = self._result.num_runs
        output["num_successful_runs"] = self._result.num_successful_runs
        output["failed_runs"] = (
            self._result.failed_runs if self._result.failed_runs else []
        )

        # Extract metadata (excluding the sweep-specific sections)
        metadata = {}
        for key, value in self._result.metadata.items():
            if key not in [
                "best_configurations",
                "pareto_optimal",
                "trends",
                "per_value_aggregates",
            ]:
                metadata[key] = value

        output["metadata"] = metadata

        # Add per_combination_metrics (stored in result.metrics)
        output["per_combination_metrics"] = self._result.metrics

        # Add sweep-specific sections from metadata
        output["best_configurations"] = self._result.metadata.get(
            "best_configurations", {}
        )
        output["pareto_optimal"] = self._result.metadata.get("pareto_optimal", [])

        # Serialize to JSON with indentation
        # OPT_SERIALIZE_NUMPY handles numpy types (float64, int64, etc.)
        return orjson.dumps(
            output, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
        ).decode("utf-8")
