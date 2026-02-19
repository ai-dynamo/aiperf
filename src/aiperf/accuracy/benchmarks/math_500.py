# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin


class Math500Benchmark(AIPerfLoggerMixin):
    """MATH-500 benchmark loader for mathematical reasoning evaluation."""

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        raise NotImplementedError
