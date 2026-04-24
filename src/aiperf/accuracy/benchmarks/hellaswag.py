# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.config import BenchmarkRun


class HellaSwagBenchmark(AIPerfLoggerMixin):
    """HellaSwag commonsense reasoning benchmark loader."""

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        raise NotImplementedError
