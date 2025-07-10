# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class DataExporterType(CaseInsensitiveStrEnum):
    CONSOLE = "console"
    CONSOLE_ERROR = "console_error"
    JSON = "json"
