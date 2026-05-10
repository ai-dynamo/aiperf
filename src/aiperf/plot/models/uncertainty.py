# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data contract models for latency-throughput uncertainty plots."""

from pydantic import Field, ValidationInfo, field_validator

from aiperf.common.models.base_models import AIPerfBaseModel


class BenchmarkPoint(AIPerfBaseModel):
    """Single benchmark operating-point measurement with uncertainty."""

    x_mean: float = Field(description="Mean value on the x-axis (e.g., latency)")
    y_mean: float = Field(description="Mean value on the y-axis (e.g., throughput)")
    x_ci_low: float = Field(description="Lower bound of x-axis confidence interval")
    x_ci_high: float = Field(description="Upper bound of x-axis confidence interval")
    y_ci_low: float = Field(description="Lower bound of y-axis confidence interval")
    y_ci_high: float = Field(description="Upper bound of y-axis confidence interval")
    cov_xy: float | None = Field(
        default=None,
        description="Sample covariance between x and y metrics; enables rotated ellipses when non-None and non-zero",
    )
    label: str | None = Field(
        default=None,
        description="Optional text label for this point (e.g., concurrency level)",
    )
    n_runs: int | None = Field(
        default=None,
        ge=1,
        description="Number of profiling runs that produced this point; used to flag low-confidence ellipses (n < 3)",
    )

    @field_validator("x_ci_low")
    @classmethod
    def x_ci_low_le_mean(cls, v: float, info: ValidationInfo) -> float:
        if "x_mean" in info.data and v > info.data["x_mean"]:
            raise ValueError(
                f"x_ci_low ({v}) must be <= x_mean ({info.data['x_mean']})"
            )
        return v

    @field_validator("x_ci_high")
    @classmethod
    def x_ci_high_ge_mean(cls, v: float, info: ValidationInfo) -> float:
        if "x_mean" in info.data and v < info.data["x_mean"]:
            raise ValueError(
                f"x_ci_high ({v}) must be >= x_mean ({info.data['x_mean']})"
            )
        return v

    @field_validator("y_ci_low")
    @classmethod
    def y_ci_low_le_mean(cls, v: float, info: ValidationInfo) -> float:
        if "y_mean" in info.data and v > info.data["y_mean"]:
            raise ValueError(
                f"y_ci_low ({v}) must be <= y_mean ({info.data['y_mean']})"
            )
        return v

    @field_validator("y_ci_high")
    @classmethod
    def y_ci_high_ge_mean(cls, v: float, info: ValidationInfo) -> float:
        if "y_mean" in info.data and v < info.data["y_mean"]:
            raise ValueError(
                f"y_ci_high ({v}) must be >= y_mean ({info.data['y_mean']})"
            )
        return v


class UncertaintySeries(AIPerfBaseModel):
    """A named series of benchmark points for multi-series uncertainty plots."""

    name: str = Field(description="Series display name (e.g., 'request_count=20')")
    points: list[BenchmarkPoint] = Field(description="Operating points in this series")


class LatencyThroughputUncertaintyData(AIPerfBaseModel):
    """Container for all benchmark points and plot metadata."""

    points: list[BenchmarkPoint] = Field(
        default_factory=list,
        description="List of benchmark operating points (single-series mode)",
    )
    series: list[UncertaintySeries] = Field(
        default_factory=list,
        description="Named series for multi-series mode (overrides points when non-empty)",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for ellipses (0.90, 0.95, or 0.99)",
    )
    title: str | None = Field(default=None, description="Plot title")
    x_label: str | None = Field(default=None, description="X-axis label")
    y_label: str | None = Field(default=None, description="Y-axis label")
    group_by: str | None = Field(default=None, description="Column to group data by")

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: float) -> float:
        if v not in {0.90, 0.95, 0.99}:
            raise ValueError(f"confidence_level must be 0.90, 0.95, or 0.99 (got {v})")
        return v

    def get_series(self) -> list[UncertaintySeries]:
        """Return series list, wrapping single-series points if needed."""
        if self.series:
            return self.series
        if self.points:
            return [UncertaintySeries(name="Mean", points=self.points)]
        return []
