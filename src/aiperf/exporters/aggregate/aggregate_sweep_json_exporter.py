# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for sweep aggregate results."""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepJsonExporter(AggregateBaseExporter):
    """Exports sweep-level aggregate results to JSON format.

    Writes the sweep dict (metadata, per_combination_metrics,
    best_configurations, pareto_optimal) directly as JSON.
    """

    def __init__(self, config, sweep_dict: dict[str, Any], **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._sweep_dict = sweep_dict

    def get_file_name(self) -> str:
        return "profile_export_aiperf_sweep.json"

    def _generate_content(self) -> str:
        return orjson.dumps(
            self._sweep_dict,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        ).decode()
