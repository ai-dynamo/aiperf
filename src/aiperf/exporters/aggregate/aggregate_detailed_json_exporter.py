# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for detailed aggregate results."""

import orjson

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateDetailedJsonExporter(AggregateBaseExporter):
    """Exports detailed aggregate results (per-request combined percentiles) to JSON."""

    def get_file_name(self) -> str:
        return "profile_export_aiperf_detailed.json"

    def _generate_content(self) -> str:
        from importlib.metadata import version as get_version

        try:
            aiperf_version = get_version("aiperf")
        except Exception:
            aiperf_version = "unknown"

        output = {
            "schema_version": "1.0.0",
            "aiperf_version": aiperf_version,
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
