# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration - Reusable Type Definitions

Distribution types are defined in aiperf.config.distributions and re-exported
here. This module also defines SequenceDistributionEntry and related
validation utilities.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import ConfigDict, Field, model_validator

from aiperf.config.base import BaseConfig
from aiperf.config.distributions import (
    Distribution,
    EmpiricalDistribution,
    EmpiricalPoint,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
    PeakEntry,
    SamplingDistribution,
)

__all__ = [
    "Distribution",
    "EmpiricalDistribution",
    "EmpiricalPoint",
    "FixedDistribution",
    "LogNormalDistribution",
    "MultimodalDistribution",
    "NormalDistribution",
    "PeakEntry",
    "SamplingDistribution",
    "SequenceDistributionEntry",
    "validate_probability_distribution",
]


class SequenceDistributionEntry(BaseConfig):
    """Defines a single entry in an ISL/OSL probability distribution.

    AIPerf supports multi-modal token length distributions, allowing
    different ISL/OSL combinations with relative frequencies for
    realistic workload modeling.

    YAML Representation:
        sequence_distribution:
          - {isl: 128, osl: 64, probability: 40}
          - {isl: {mean: 512, stddev: 50}, osl: 256, probability: 35}
          - {isl: {mean: 2048, median: 1800}, osl: 512, probability: 25}
    """

    isl: Annotated[
        SamplingDistribution,
        Field(
            description="Input sequence length (tokens). "
            "Can be a fixed integer, a {mean, stddev} distribution, "
            "or any distribution like {mean: 512, median: 400} for lognormal."
        ),
    ]

    isl_stddev: Annotated[
        float | None,
        Field(
            ge=0.0,
            default=None,
            description="Shorthand for ISL standard deviation. "
            "If provided when isl is an integer, creates a normal distribution. "
            "Cannot be used when isl is already a distribution dict.",
        ),
    ]

    osl: Annotated[
        SamplingDistribution,
        Field(
            description="Output sequence length (tokens). "
            "Can be a fixed integer, a {mean, stddev} distribution, "
            "or any typed distribution."
        ),
    ]

    osl_stddev: Annotated[
        float | None,
        Field(
            ge=0.0,
            default=None,
            description="Shorthand for OSL standard deviation. "
            "If provided when osl is an integer, creates a normal distribution. "
            "Cannot be used when osl is already a distribution dict.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def merge_stddev_shorthand(cls, data: Any) -> Any:
        """Merge isl_stddev/osl_stddev shorthand into isl/osl fields."""
        if not isinstance(data, dict):
            return data

        if "isl_stddev" in data and data["isl_stddev"] is not None:
            isl = data.get("isl")
            if isinstance(isl, (int, float)):
                data["isl"] = {
                    "mean": float(isl),
                    "stddev": data["isl_stddev"],
                }
            data["isl_stddev"] = None

        if "osl_stddev" in data and data["osl_stddev"] is not None:
            osl = data.get("osl")
            if isinstance(osl, (int, float)):
                data["osl"] = {
                    "mean": float(osl),
                    "stddev": data["osl_stddev"],
                }
            data["osl_stddev"] = None

        return data

    probability: Annotated[
        float,
        Field(
            ge=0.0,
            le=100.0,
            description="Relative probability weight for this distribution bucket (0-100). "
            "Weights are normalized across all entries. "
            "Example: probability=40 means 40%% of requests use this ISL/OSL.",
        ),
    ]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"isl": 128, "osl": 64, "probability": 40},
                {
                    "isl": 512,
                    "isl_stddev": 50,
                    "osl": 256,
                    "osl_stddev": 25,
                    "probability": 35,
                },
            ]
        },
    )


def validate_probability_distribution(
    entries: list[SequenceDistributionEntry],
) -> list[SequenceDistributionEntry]:
    """Validate that a probability distribution sums to approximately 100."""
    total = sum(entry.probability for entry in entries)
    if not (99.0 <= total <= 101.0):
        raise ValueError(
            f"Sequence distribution probabilities must sum to ~100, got {total}. "
            f"Individual probabilities: {[e.probability for e in entries]}"
        )
    return entries
