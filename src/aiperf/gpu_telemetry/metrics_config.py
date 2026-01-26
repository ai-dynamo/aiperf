# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU telemetry metrics configuration utilities."""

import re
from pathlib import Path

from aiperf.common.enums.metric_enums import (
    EnergyMetricUnit,
    FrequencyMetricUnit,
    GenericMetricUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    MetricUnitT,
    PowerMetricUnit,
    TemperatureMetricUnit,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.gpu_telemetry import constants


def _title_case_metric_name(name: str) -> str:
    """
    Convert metric name to proper title case with special handling for acronyms.

    Handles special cases like GPU, XID, SM, etc. to remain fully capitalized.

    Args:
        name: The metric name to convert

    Returns:
        Title-cased metric name with proper acronym capitalization

    Example:
        >>> _title_case_metric_name("gpu power usage")
        "GPU Power Usage"
        >>> _title_case_metric_name("xid errors")
        "XID Errors"
        >>> _title_case_metric_name("sm clock frequency")
        "SM Clock Frequency"
    """
    # List of acronyms that should be fully capitalized
    acronyms = {
        "gpu", "xid", "sm", "nvlink", "pci", "pcie", "cpu", "ram", "vram", "ecc",
    }  # fmt: skip

    return " ".join(
        word.upper() if word.lower() in acronyms else word.capitalize()
        for word in name.split()
    )


class MetricsConfigLoader(AIPerfLoggerMixin):
    """GPU telemetry metrics configuration loader.

    Parses and loads custom GPU metrics from CSV files and provides utilities
    for metric unit inference and configuration building.
    """

    def __init__(self) -> None:
        """Initialize the metrics config loader."""
        super().__init__()

    def parse_custom_metrics_csv(self, csv_path: Path) -> list[tuple[str, str, str]]:
        """
        Parse DCGM-style custom metrics CSV file.

        Format:
            # Comment lines start with #
            DCGM_FIELD_NAME, metric_type, help message

        Args:
            csv_path: Path to the CSV file

        Returns:
            List of (dcgm_field, metric_type, help_message) tuples

        Example:
            >>> parse_custom_metrics_csv(Path("metrics.csv"))
            [('DCGM_FI_DEV_POWER_USAGE', 'gauge', 'Power draw (in W)'),
             ('DCGM_FI_DEV_GPU_UTIL', 'gauge', 'GPU utilization (in %)')]
        """
        custom_metrics = []

        with open(csv_path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                # Parse CSV line (split on first 2 commas only, allow commas in help message)
                parts = [p.strip() for p in line.split(",", 2)]
                if len(parts) != 3:
                    self.warning(
                        f"Skipping invalid line {line_num} in {csv_path}: "
                        f"expected 3 comma-separated values, got {len(parts)}"
                    )
                    continue

                dcgm_field, metric_type, help_msg = parts

                if not dcgm_field.startswith("DCGM_FI_"):
                    self.warning(
                        f"Skipping line {line_num} in {csv_path}: "
                        f"DCGM field '{dcgm_field}' should start with 'DCGM_FI_'"
                    )
                    continue

                if metric_type not in ("gauge", "counter"):
                    self.warning(
                        f"Skipping line {line_num} in {csv_path}: "
                        f"metric type '{metric_type}' should be 'gauge' or 'counter'"
                    )
                    continue

                custom_metrics.append((dcgm_field, metric_type, help_msg))

        return custom_metrics

    def _infer_unit_from_help(self, help_msg: str) -> MetricUnitT:
        """
        Infer metric unit from help message like 'Power draw (in W)'.

        Args:
            help_msg: Help message containing unit in "(in UNIT)" format

        Returns:
            Appropriate MetricUnitT enum value

        Example:
            >>> _infer_unit_from_help("Power draw (in W)")
            PowerMetricUnit.WATT
            >>> _infer_unit_from_help("GPU utilization (in %)")
            GenericMetricUnit.PERCENT
        """
        # Extract unit from "(in UNIT)" pattern
        # Use [^\s)]+ instead of [^)]+ to avoid ReDoS from overlapping quantifiers
        # (whitespace matches both \s+ and [^)]+, causing O(n²) backtracking)
        match = re.search(r"\(in\s+([^\s)]+)\)", help_msg, re.IGNORECASE)
        if not match:
            return GenericMetricUnit.COUNT

        unit_str = match.group(1).strip().lower()

        unit_mapping = {
            "w": PowerMetricUnit.WATT,
            "%": GenericMetricUnit.PERCENT,
            "percent": GenericMetricUnit.PERCENT,
            "gb": MetricSizeUnit.GIGABYTES,
            "mb": MetricSizeUnit.MEGABYTES,
            "kb": MetricSizeUnit.KILOBYTES,
            "mhz": FrequencyMetricUnit.MEGAHERTZ,
            "ghz": FrequencyMetricUnit.GIGAHERTZ,
            "c": TemperatureMetricUnit.CELSIUS,
            "°c": TemperatureMetricUnit.CELSIUS,
            "celsius": TemperatureMetricUnit.CELSIUS,
            "count": GenericMetricUnit.COUNT,
            "us": MetricTimeUnit.MICROSECONDS,
            "ms": MetricTimeUnit.MILLISECONDS,
            "s": MetricTimeUnit.SECONDS,
            "mj": EnergyMetricUnit.MEGAJOULE,
            "j": EnergyMetricUnit.JOULE,
        }

        return unit_mapping.get(unit_str, GenericMetricUnit.COUNT)

    def build_custom_metrics_from_csv(
        self, custom_csv_path: Path
    ) -> tuple[list[tuple[str, str, MetricUnitT]], dict[str, str]]:
        """
        Build custom GPU telemetry metrics from CSV file.

        Parses the CSV and returns NEW custom metrics and their DCGM field mappings.
        Does not mutate any constants - caller should apply returned mappings.

        Args:
            custom_csv_path: Path to custom metrics CSV file

        Returns:
            Tuple of (custom_metrics, new_dcgm_mappings) where:
            - custom_metrics: List of (display_name, field_name, unit) tuples for NEW metrics
            - new_dcgm_mappings: Dict of new DCGM field name -> internal field name mappings

        Example:
            >>> loader = MetricsConfigLoader()
            >>> new_metrics, new_mappings = loader.build_custom_metrics_from_csv(
            ...     Path("my_metrics.csv")
            ... )
            >>> len(new_metrics)  # Only new metrics, not defaults
            3
            >>> new_mappings  # New DCGM field mappings to apply
            {'DCGM_FI_DEV_SM_CLOCK': 'sm_clock', ...}
        """
        existing_field_names = {
            cfg[1] for cfg in constants.GPU_TELEMETRY_METRICS_CONFIG
        }
        existing_dcgm_fields = set(constants.DCGM_TO_FIELD_MAPPING.keys())

        try:
            custom_metrics = self.parse_custom_metrics_csv(custom_csv_path)
        except Exception as e:
            self.error(f"Failed to parse {custom_csv_path}: {e}")
            return ([], {})

        if not custom_metrics:
            return ([], {})

        new_metrics = []
        new_dcgm_mappings = {}

        for dcgm_field, _metric_type, help_msg in custom_metrics:
            if dcgm_field in existing_dcgm_fields:
                self.debug(
                    f"Skipping DCGM field already in default config: {dcgm_field}"
                )
                continue

            internal_name = dcgm_field.replace("DCGM_FI_DEV_", "").lower()
            display_name = help_msg.split("(")[0].strip()

            if not display_name:
                # Fallback: convert internal_name to title case
                display_name = internal_name.replace("_", " ")

            display_name = _title_case_metric_name(display_name)

            unit = self._infer_unit_from_help(help_msg)

            new_dcgm_mappings[dcgm_field] = internal_name

            new_metrics.append((display_name, internal_name, unit))
            existing_field_names.add(internal_name)
            self.debug(f"Added custom metric: {display_name} ({internal_name})")
        return (new_metrics, new_dcgm_mappings)
