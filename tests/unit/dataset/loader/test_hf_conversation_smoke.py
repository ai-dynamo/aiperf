# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for HFConversationDatasetLoader after config-v3 port."""

import pytest

from aiperf.config import BenchmarkRun
from aiperf.dataset.loader.hf_conversation import HFConversationDatasetLoader


@pytest.mark.asyncio
async def test_hf_conversation_loader_constructor_accepts_run(
    default_user_run: BenchmarkRun,
) -> None:
    loader = HFConversationDatasetLoader(
        run=default_user_run,
        conversation_column="conversation",
        hf_dataset_name="lmarena-ai/VisionArena-Chat",
        hf_split="train",
    )
    assert loader.run is default_user_run
    assert loader.conversation_column == "conversation"
    assert loader.message_content_key == "content"
    assert loader.image_column is None


@pytest.mark.asyncio
async def test_hf_conversation_loader_accepts_optional_args(
    default_user_run: BenchmarkRun,
) -> None:
    loader = HFConversationDatasetLoader(
        run=default_user_run,
        conversation_column="conversations",
        message_content_key="value",
        image_column="image",
        hf_dataset_name="lmms-lab/LLaVA-OneVision-Data",
        hf_subset="sharegpt4o",
    )
    assert loader.message_content_key == "value"
    assert loader.image_column == "image"
    assert loader.hf_subset == "sharegpt4o"
