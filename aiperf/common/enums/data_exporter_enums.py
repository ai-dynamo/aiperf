# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ConsoleExporterType(CaseInsensitiveStrEnum):
    ERRORS = "errors"
    EXPERIMENTAL_METRICS = "experimental_metrics"
    INTERNAL_METRICS = "internal_metrics"
    METRICS = "metrics"


class DataExporterType(CaseInsensitiveStrEnum):
    JSON = "json"
    CSV = "csv"


class ExportLevel(CaseInsensitiveStrEnum):
    """Export level for benchmark data.

    - SUMMARY: Export only aggregated/summarized metrics (default, most compact)
    - RECORDS: Export per-record metrics after aggregation with display unit conversion
    - RAW: Export raw parsed records with full request/response data (most detailed)
    """

    SUMMARY = "summary"
    RECORDS = "records"
    RAW = "raw"
