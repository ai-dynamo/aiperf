# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for SpeedBenchLoader after config-v3 port."""

import pytest

from aiperf.config import BenchmarkRun
from aiperf.dataset.loader.speed_bench import SpeedBenchLoader


@pytest.mark.asyncio
async def test_speed_bench_loader_constructor_accepts_run(
    default_user_run: BenchmarkRun,
) -> None:
    loader = SpeedBenchLoader(
        run=default_user_run,
        hf_dataset_name="nvidia/SPEED-Bench",
        hf_split="train",
    )
    assert loader.run is default_user_run
    assert loader.category is None
    assert loader.hf_dataset_name == "nvidia/SPEED-Bench"


@pytest.mark.asyncio
async def test_speed_bench_loader_accepts_category(
    default_user_run: BenchmarkRun,
) -> None:
    loader = SpeedBenchLoader(
        run=default_user_run,
        category="coding",
        hf_dataset_name="nvidia/SPEED-Bench",
    )
    assert loader.category == "coding"
