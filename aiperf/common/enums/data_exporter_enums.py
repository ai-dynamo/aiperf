# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ConsoleExporterType(CaseInsensitiveStrEnum):
    METRICS = "metrics"
    ERRORS = "errors"


class DataExporterType(CaseInsensitiveStrEnum):
    JSON = "json"
    CSV = "csv"
