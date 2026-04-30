# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


async def load_benchmark_problems(user_config: UserConfig) -> list[BenchmarkProblem]:
    """Load benchmark problems from the configured benchmark, resolving n_shots defaults.

    Called once by AccuracyDatasetLoader. Ground-truth answers and task names
    are stamped onto each Conversation and shipped to processors via
    DatasetConfiguredNotification, so processors never call this directly.
    """
    acc_cfg = user_config.accuracy
    benchmark_cls = plugins.get_class(PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark)

    n_shots = acc_cfg.n_shots
    if n_shots is None:
        meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark)
        n_shots = meta.get("default_n_shots", 0)

    benchmark = benchmark_cls(user_config=user_config)
    return await benchmark.load_problems(
        tasks=acc_cfg.tasks,
        n_shots=n_shots,
        enable_cot=acc_cfg.enable_cot,
    )
