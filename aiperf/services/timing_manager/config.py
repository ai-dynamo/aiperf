# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from pydantic import BaseModel


class TimingMode(str, Enum):
    """Enum for the different timing modes."""

    FIXED_SCHEDULE = "fixed_schedule"
    CONCURRENCY = "concurrency"
    RATE = "rate"


class TimingManagerConfig(BaseModel):
    timing_mode: TimingMode = TimingMode.FIXED_SCHEDULE
