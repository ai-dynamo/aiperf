# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


async def load_benchmark_problems(run: BenchmarkRun) -> list[BenchmarkProblem]:
    """Load benchmark problems from the configured benchmark, resolving n_shots defaults.

    Called once by AccuracyDatasetLoader. Ground-truth answers and task names
    are stamped onto each Conversation and shipped to processors via
    DatasetConfiguredNotification, so processors never call this directly.
    """
    acc_cfg = run.cfg.accuracy
    if acc_cfg is None or not acc_cfg.enabled:
        raise RuntimeError(
            "load_benchmark_problems called without accuracy configuration enabled"
        )
    benchmark_cls = plugins.get_class(PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark)

    meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark)

    n_shots = acc_cfg.n_shots
    if n_shots is None:
        n_shots = meta.get("default_n_shots", 0)

    enable_cot = acc_cfg.enable_cot
    if enable_cot is None:
        enable_cot = bool(meta.get("default_enable_cot", False))

    benchmark = benchmark_cls(run=run)
    return await benchmark.load_problems(
        tasks=acc_cfg.tasks,
        n_shots=n_shots,
        enable_cot=enable_cot,
    )
