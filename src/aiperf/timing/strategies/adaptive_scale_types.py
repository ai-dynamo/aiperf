# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared adaptive scale controller types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AdaptiveControllerPhase = Literal["discover", "sustain", "complete"]

MIN_ASSESSMENT_PERIOD_SEC = 1.0


@dataclass(slots=True)
class WindowRequestSample:
    request_latency_ns: int
    """End-to-end request latency in nanoseconds."""
    ttft_ns: int | None = None
    """Time to first token in nanoseconds, when observed."""
    inter_token_latency_ns: float | None = None
    """Inter-token latency in nanoseconds, when enough tokens were observed."""
    output_sequence_length: int | None = None
    """Output sequence length in tokens, when usage data was observed."""


@dataclass(slots=True)
class WindowStats:
    samples: list[int]
    """Successful request latency samples in nanoseconds."""
    errors: int
    """Completed requests that failed or lacked required latency data."""
    elapsed_sec: float
    """Assessment window duration in seconds."""
    ttft_samples: list[int] | None = None
    """Successful TTFT samples in nanoseconds."""
    itl_samples: list[float] | None = None
    """Successful ITL samples in nanoseconds."""
    successful_requests: list[WindowRequestSample] | None = None
    """Per-request quality inputs for successful completed requests."""
    cancelled: int = 0
    """Requests cancelled during the assessment window."""
    start_ns: int | None = None
    """Window start monotonic timestamp in nanoseconds."""
    end_ns: int | None = None
    """Window end monotonic timestamp in nanoseconds."""

    @property
    def total(self) -> int:
        return len(self.samples) + self.errors + self.cancelled

    @property
    def ttfts(self) -> list[int]:
        return self.ttft_samples or []

    @property
    def itls(self) -> list[float]:
        return self.itl_samples or []

    @property
    def requests(self) -> list[WindowRequestSample]:
        return self.successful_requests or []

    @property
    def output_sequence_lengths(self) -> list[int]:
        return [
            sample.output_sequence_length
            for sample in self.requests
            if sample.output_sequence_length is not None
        ]

    @property
    def throughput(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return len(self.samples) / self.elapsed_sec

    @property
    def output_token_throughput(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return sum(self.output_sequence_lengths) / self.elapsed_sec
