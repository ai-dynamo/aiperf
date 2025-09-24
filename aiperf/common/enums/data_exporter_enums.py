# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ConsoleExporterType(CaseInsensitiveStrEnum):
    ERRORS = "errors"
    EXPERIMENTAL_METRICS = "experimental_metrics"
    INTERNAL_METRICS = "internal_metrics"
    METRICS = "metrics"
    TELEMETRY = "telemetry"


class DataExporterType(CaseInsensitiveStrEnum):
    JSON = "json"
    CSV = "csv"
