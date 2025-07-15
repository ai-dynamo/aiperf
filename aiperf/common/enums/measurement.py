# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class MeasurementMode(CaseInsensitiveStrEnum):
    REQUEST_COUNT = "request_count"
    INTERVAL = "interval"
