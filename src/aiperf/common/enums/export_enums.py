# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.enums import ExportFormat as ExportFormat
from aiperf.common.enums.enums import ExportLevel as ExportLevel
from aiperf.common.enums.enums import ServerMetricsFormat as ServerMetricsFormat


class RecordExportFormat(CaseInsensitiveStrEnum):
    """Format options for per-record metrics export.

    Controls which output files are generated for per-record benchmark data.
    Default selection is JSONL only.
    """

    CSV = "csv"
    """Export per-record metrics in CSV tabular format with flat column layout.
    Best for: Spreadsheet analysis, tabular comparison, pandas DataFrames."""

    JSONL = "jsonl"
    """Export per-record metrics in line-delimited JSON with nested metadata.
    Best for: Programmatic access, structured analysis, debugging."""


class SummaryFormat(CaseInsensitiveStrEnum):
    """Defines the file format for summary exports."""

    JSON = "json"
    """JSON format for summaries."""

    YAML = "yaml"
    """YAML format for summaries."""
