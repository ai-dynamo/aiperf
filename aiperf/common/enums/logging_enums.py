# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AIPerfLogLevel(CaseInsensitiveStrEnum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
