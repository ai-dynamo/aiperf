# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for detailed aggregate results."""

import orjson

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateDetailedJsonExporter(AggregateBaseExporter):
    """Exports detailed aggregate results (per-request combined percentiles) to JSON."""

    def get_file_name(self) -> str:
        return "profile_export_aiperf_collated.json"

    def _generate_content(self) -> str:
        from aiperf import __version__ as aiperf_version

        output = {
            "schema_version": "1.0.0",
            "aiperf_version": aiperf_version,
            "description": (
                "Collated per-request metrics across all runs. "
                "Pools individual request-level values from every run into a single population "
                "and computes combined percentiles (p50, p90, p95, p99). "
                "Contrast with profile_export_aiperf_aggregate.json, which computes statistics "
                "over run-level summary values."
            ),
            "metadata": {
                "aggregation_type": self._result.aggregation_type,
                "num_profile_runs": self._result.num_runs,
                "num_successful_runs": self._result.num_successful_runs,
                "failed_runs": self._result.failed_runs,
                **self._result.metadata,
            },
            "metrics": self._result.metrics,
        }

        return orjson.dumps(output, option=orjson.OPT_INDENT_2).decode("utf-8")
