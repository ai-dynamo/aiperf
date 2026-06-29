# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared adaptive scale controller types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AdaptiveControllerPhase = Literal["discover", "sustain", "complete"]

MIN_ASSESSMENT_PERIOD_SEC = 1.0


@dataclass(slots=True)
class WindowStats:
    samples: list[int]
    errors: int
    elapsed_sec: float
    start_ns: int | None = None
    end_ns: int | None = None

    @property
    def total(self) -> int:
        return len(self.samples) + self.errors

    @property
    def throughput(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return len(self.samples) / self.elapsed_sec
