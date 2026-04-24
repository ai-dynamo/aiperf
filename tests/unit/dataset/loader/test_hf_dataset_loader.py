# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.loader.hf_instruction_response import (
    HFInstructionResponseDatasetLoader,
)
from aiperf.plugin.enums import DatasetSamplingStrategy

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000/v1/chat/completions"],
    },
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


def _make_run(**overrides: Any) -> BenchmarkRun:
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    config = BenchmarkConfig(**kwargs)
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def run() -> BenchmarkRun:
    return _make_run()


@pytest.fixture
async def loader(run: BenchmarkRun) -> HFInstructionResponseDatasetLoader:
    return HFInstructionResponseDatasetLoader(
        run=run,
        hf_dataset_name="AI-MO/NuminaMath-TIR",
        hf_split="train",
        prompt_column="problem",
    )


@pytest.mark.asyncio
class TestBaseHFDatasetLoader:
    async def test_preferred_sampling_strategy_is_sequential(self, loader):
        assert (
            loader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_attributes_stored(self, loader):
        assert loader.hf_dataset_name == "AI-MO/NuminaMath-TIR"
        assert loader.hf_split == "train"
        assert loader.hf_subset is None

    async def test_subset_stored_when_provided(self, run):
        loader = HFInstructionResponseDatasetLoader(
            run=run,
            hf_dataset_name="test/dataset",
            hf_split="validation",
            hf_subset="subset-a",
            prompt_column="text",
        )
        assert loader.hf_subset == "subset-a"

    async def test_load_dataset_wraps_error_in_dataset_loader_error(self, loader):
        with (
            patch.object(
                loader, "_load_hf_dataset", side_effect=RuntimeError("network error")
            ),
            pytest.raises(DatasetLoaderError, match="Failed to load"),
        ):
            await loader.load_dataset()

    async def test_load_dataset_returns_dataset_dict(self, loader):
        fake_dataset = [{"problem": "2+2=?"}]
        with patch.object(loader, "_load_hf_dataset", return_value=fake_dataset):
            result = await loader.load_dataset()
        assert result == {"dataset": fake_dataset}

    async def test_load_hf_dataset_calls_load_dataset_with_correct_args(self, run):
        loader = HFInstructionResponseDatasetLoader(
            run=run,
            hf_dataset_name="test/data",
            hf_split="test",
            hf_subset="my-subset",
            prompt_column="q",
        )
        mock_load_dataset = MagicMock(return_value=[])
        with patch(
            "aiperf.dataset.loader.base_hf_dataset.hf_load_dataset", mock_load_dataset
        ):
            loader._load_hf_dataset()

        mock_load_dataset.assert_called_once_with(
            "test/data",
            name="my-subset",
            split="test",
            trust_remote_code=False,
            streaming=False,
        )


@pytest.mark.asyncio
class TestHFInstructionResponseDatasetLoader:
    async def test_converts_rows_to_conversations(self, loader):
        data = {
            "dataset": [
                {"problem": "What is 2+2?"},
                {"problem": "Solve for x: x^2 = 9"},
            ]
        }
        conversations = await loader.convert_to_conversations(data)

        assert len(conversations) == 2
        assert all(isinstance(c, Conversation) for c in conversations)
        assert conversations[0].turns[0].texts[0].contents[0] == "What is 2+2?"
        assert conversations[1].turns[0].texts[0].contents[0] == "Solve for x: x^2 = 9"

    async def test_each_row_becomes_single_turn(self, loader):
        data = {"dataset": [{"problem": "Prove Fermat's Last Theorem."}]}
        conversations = await loader.convert_to_conversations(data)

        assert len(conversations[0].turns) == 1

    async def test_skips_empty_prompt_rows(self, loader):
        data = {
            "dataset": [
                {"problem": ""},
                {"problem": "   "},
                {"problem": None},
                {"problem": "Valid problem"},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid problem"

    async def test_skips_missing_prompt_column(self, loader):
        data = {"dataset": [{"other_field": "value"}, {"problem": "Valid"}]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_session_ids_are_unique(self, loader):
        data = {"dataset": [{"problem": f"Q{i}"} for i in range(5)]}
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5

    async def test_empty_dataset_returns_empty_list(self, loader):
        data = {"dataset": []}
        conversations = await loader.convert_to_conversations(data)
        assert conversations == []

    async def test_uses_configured_prompt_column(self, run):
        loader = HFInstructionResponseDatasetLoader(
            run=run,
            hf_dataset_name="test/data",
            hf_split="train",
            prompt_column="question",
        )
        data = {"dataset": [{"question": "What is the capital of France?"}]}
        conversations = await loader.convert_to_conversations(data)

        assert conversations[0].turns[0].texts[0].contents[0] == (
            "What is the capital of France?"
        )
