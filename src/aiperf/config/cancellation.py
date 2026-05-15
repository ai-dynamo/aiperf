# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request cancellation configuration shared by phase configuration types."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig


class CancellationConfig(BaseConfig):
    """
    Configuration for request cancellation testing.

    Enables testing server behavior when clients cancel requests mid-flight.
    """

    model_config = ConfigDict(extra="forbid")

    rate: Annotated[
        float,
        Field(
            ge=0.0,
            le=100.0,
            description="Percentage of requests to cancel (0-100). "
            "0.5 means half a percent; 10.0 means 10%% of requests will be cancelled.",
        ),
    ]

    delay: Annotated[
        float,
        Field(
            ge=0.0,
            default=0.0,
            description="Seconds to wait after sending before cancelling. "
            "0.0 means cancel immediately after send.",
        ),
    ]
