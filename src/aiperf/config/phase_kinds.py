# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rate-controlled and fixed-schedule phase config models."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.config.phases import BasePhaseConfig, RampSpec
from aiperf.plugin.enums import PhaseType


class RatePhaseConfig(BasePhaseConfig):
    """Base for rate-controlled phases. Not instantiated directly."""

    rate: Annotated[
        float,
        Field(
            gt=0,
            description="Target request rate in requests per second (must be > 0).",
        ),
    ]

    rate_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp rate from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]


class PoissonPhase(RatePhaseConfig):
    """Poisson-distributed request arrivals at the target rate."""

    type: Annotated[
        Literal[PhaseType.POISSON],
        Field(description="Poisson-distributed rate-controlled arrivals."),
    ]


class GammaPhase(RatePhaseConfig):
    """Gamma-distributed request arrivals with configurable smoothness."""

    type: Annotated[
        Literal[PhaseType.GAMMA],
        Field(description="Gamma-distributed rate-controlled arrivals."),
    ]

    smoothness: Annotated[
        float | None,
        Field(
            gt=0,
            default=None,
            description="Gamma distribution shape parameter (must be > 0). "
            "1.0 = Poisson, <1 = bursty, >1 = regular.",
        ),
    ]


class ConstantPhase(RatePhaseConfig):
    """Constant-rate request arrivals (fixed inter-arrival time)."""

    type: Annotated[
        Literal[PhaseType.CONSTANT],
        Field(description="Constant rate-controlled arrivals."),
    ]


class UserCentricPhase(RatePhaseConfig):
    """N concurrent users sharing a global request rate.

    Requires multi-turn conversations. Each user gets a proportional
    share of the global ``rate``.
    """

    type: Annotated[
        Literal[PhaseType.USER_CENTRIC],
        Field(description="N users sharing a global request rate."),
    ]

    users: Annotated[
        int,
        Field(
            ge=1,
            description="Number of simulated concurrent users (must be >= 1). "
            "Requests distributed across users to achieve global rate.",
        ),
    ]

    @model_validator(mode="after")
    def validate_user_centric_constraints(self) -> UserCentricPhase:
        """Validate user-centric mode constraints."""
        if self.sessions is not None and self.sessions < self.users:
            raise ValueError(
                f"Phase '{self.name}': --num-sessions ({self.sessions}) must be "
                f">= --num-users ({self.users}). Each user needs at least one session."
            )

        if self.requests is not None and self.requests < self.users:
            raise ValueError(
                f"Phase '{self.name}': --request-count ({self.requests}) must be "
                f">= --num-users ({self.users}). Each user needs at least one request."
            )

        return self


# =============================================================================
# FIXED SCHEDULE PHASE
# =============================================================================


class FixedSchedulePhase(BasePhaseConfig):
    """Replay requests at predetermined timestamps from a trace dataset.

    Stop condition not required -- the trace dataset determines when the
    phase ends.
    """

    _stop_condition_required: ClassVar[bool] = False

    type: Annotated[
        Literal[PhaseType.FIXED_SCHEDULE],
        Field(description="Replay requests at trace timestamps."),
    ]

    auto_offset: Annotated[
        bool,
        Field(
            default=True,
            description="Normalize trace timestamps to start at 0. "
            "Subtracts minimum timestamp from all entries.",
        ),
    ]

    start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            default=None,
            description="Filter out trace requests before this timestamp in ms (must be >= 0).",
        ),
    ]

    end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            default=None,
            description="Filter out trace requests after this timestamp in ms (must be >= 0).",
        ),
    ]

    @model_validator(mode="after")
    def _validate_fixed_schedule_constraints(self) -> Self:
        if self.auto_offset and self.start_offset is not None:
            raise ValueError("auto_offset cannot be True when start_offset is set")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.start_offset > self.end_offset
        ):
            raise ValueError("start_offset must be <= end_offset")
        return self
