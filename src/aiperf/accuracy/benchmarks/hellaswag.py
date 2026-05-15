# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class HellaSwagBenchmark(AIPerfLoggerMixin):
    """Registered placeholder for a future HellaSwag loader.

    `load_problems()` intentionally raises NotImplementedError in this release;
    use the MMLU benchmark when a working accuracy loader is required.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        raise NotImplementedError(
            "hellaswag benchmark is not yet implemented; only 'mmlu' is available in this release."
        )
