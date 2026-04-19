# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for MMVUDatasetLoader after config-v3 port."""

import pytest

from aiperf.config import BenchmarkRun
from aiperf.dataset.loader.mmvu import MMVUDatasetLoader


@pytest.mark.asyncio
async def test_mmvu_loader_constructor_accepts_run(
    default_user_run: BenchmarkRun,
) -> None:
    loader = MMVUDatasetLoader(
        run=default_user_run,
        hf_dataset_name="yale-nlp/MMVU",
        hf_split="validation",
    )
    assert loader.run is default_user_run
    assert loader.video_column == "video"


@pytest.mark.asyncio
async def test_mmvu_loader_custom_video_column(
    default_user_run: BenchmarkRun,
) -> None:
    loader = MMVUDatasetLoader(
        run=default_user_run,
        video_column="clip",
        hf_dataset_name="yale-nlp/MMVU",
    )
    assert loader.video_column == "clip"
