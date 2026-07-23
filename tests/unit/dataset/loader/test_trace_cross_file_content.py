# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-file content distinction for trace loaders sharing PromptGenerator."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from aiperf.config.dataset.content import (
    PrefixPromptConfig,
    PromptConfig,
)
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.bailian_trace import BailianTraceDatasetLoader
from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

MOCK_CORPUS_CONTENT = " ".join([f"word{i}" for i in range(1024)]) + "\n"

_TEST_BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _mock_parallel_decode_tokenizer(mock_tokenizer_cls):
    """Route ``parallel_decode``'s tokenizer load to the mock."""
    with patch(
        "aiperf.common.tokenizer.Tokenizer.from_pretrained",
        lambda name, **kw: mock_tokenizer_cls.from_pretrained(name),
    ):
        yield


@pytest.fixture
def real_prompt_generator(mock_tokenizer_cls):
    """Build a real PromptGenerator backed by the mock tokenizer."""
    tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
    prompts = PromptConfig(block_size=_TEST_BLOCK_SIZE)
    prefix_prompts = PrefixPromptConfig(pool_size=None, length=None)
    with patch("builtins.open", mock_open(read_data=MOCK_CORPUS_CONTENT)):
        return PromptGenerator(
            prompts=prompts, prefix_prompts=prefix_prompts, tokenizer=tokenizer
        )


@pytest.fixture
def default_user_config():
    """Real v2 BenchmarkRun (named ``default_user_config`` for minimal churn)."""
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(model_names=["test-model"])


def _write_jsonl(tmp_path: Path, name: str, lines: list[str]) -> str:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _write_burst_csv(
    tmp_path: Path, name: str, rows: list[tuple[float, int, int]]
) -> str:
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Request tokens", "Response tokens"])
        for ts, req, resp in rows:
            writer.writerow([ts, req, resp])
    return str(p)


class TestMooncakeCrossFileContent:
    """Cross-file collision regression for MooncakeTraceDatasetLoader."""

    def _make_loader(
        self, filename: str, pg, user_config
    ) -> MooncakeTraceDatasetLoader:
        return MooncakeTraceDatasetLoader(
            filename=filename,
            run=user_config,
            prompt_generator=pg,
            default_block_size=_TEST_BLOCK_SIZE,
        )

    def _convert_first_prompt(self, loader: MooncakeTraceDatasetLoader) -> str:
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        return conversations[0].turns[0].texts[0].contents[0]

    def test_mooncake_distinct_content_across_files(
        self, tmp_path, real_prompt_generator, default_user_config
    ):
        line_a = '{"timestamp": 1, "input_length": 8, "output_length": 4, "hash_ids": [101, 202]}'
        file_a = _write_jsonl(tmp_path, "trace_a.jsonl", [line_a])
        file_b = _write_jsonl(
            tmp_path,
            "trace_b.jsonl",
            [
                line_a,
                '{"timestamp": 99, "input_length": 8, "output_length": 4, "hash_ids": [333]}',
            ],
        )

        loader_a = self._make_loader(file_a, real_prompt_generator, default_user_config)
        prompt_a = self._convert_first_prompt(loader_a)

        loader_b = self._make_loader(file_b, real_prompt_generator, default_user_config)
        prompt_b = self._convert_first_prompt(loader_b)

        assert prompt_a != prompt_b, (
            "Same hash_ids in different files must produce different content: "
            f"{prompt_a!r} == {prompt_b!r}"
        )

    def test_mooncake_deterministic_within_file(
        self, tmp_path, real_prompt_generator, default_user_config
    ):
        lines = [
            '{"session_id": "s1", "input_length": 8, "output_length": 4, "hash_ids": [42, 99]}',
            '{"session_id": "s1", "delay": 1, "input_length": 8, "output_length": 4, "hash_ids": [42, 99]}',
        ]
        f = _write_jsonl(tmp_path, "trace_repeat.jsonl", lines)

        loader = self._make_loader(f, real_prompt_generator, default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        turn0_prompt = conversations[0].turns[0].texts[0].contents[0]
        turn1_prompt = conversations[0].turns[1].texts[0].contents[0]
        assert turn0_prompt == turn1_prompt


class TestBailianCrossFileContent:
    """Cross-file collision regression for BailianTraceDatasetLoader."""

    def _make_loader(self, filename: str, pg, user_config) -> BailianTraceDatasetLoader:
        return BailianTraceDatasetLoader(
            filename=filename,
            run=user_config,
            prompt_generator=pg,
            default_block_size=_TEST_BLOCK_SIZE,
        )

    def _first_prompt(self, loader: BailianTraceDatasetLoader) -> str:
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        return conversations[0].turns[0].texts[0].contents[0]

    def test_bailian_distinct_content_across_files(
        self, tmp_path, real_prompt_generator, default_user_config
    ):
        line_a = (
            '{"chat_id": 1, "parent_chat_id": -1, "timestamp": 1.0, '
            '"input_length": 8, "output_length": 4, "type": "text", '
            '"turn": 1, "hash_ids": [555, 666]}'
        )
        file_a = _write_jsonl(tmp_path, "bailian_a.jsonl", [line_a])
        file_b = _write_jsonl(
            tmp_path,
            "bailian_b.jsonl",
            [
                line_a,
                '{"chat_id": 2, "parent_chat_id": -1, "timestamp": 2.0, '
                '"input_length": 8, "output_length": 4, "type": "text", '
                '"turn": 1, "hash_ids": [777]}',
            ],
        )

        loader_a = self._make_loader(file_a, real_prompt_generator, default_user_config)
        prompt_a = self._first_prompt(loader_a)

        loader_b = self._make_loader(file_b, real_prompt_generator, default_user_config)
        prompt_b = self._first_prompt(loader_b)

        assert prompt_a != prompt_b, (
            "Bailian: same hash_ids across files must yield distinct content."
        )

    def test_bailian_deterministic_within_file(
        self, tmp_path, real_prompt_generator, default_user_config
    ):
        lines = [
            '{"chat_id": 1, "parent_chat_id": -1, "timestamp": 1.0, '
            '"input_length": 8, "output_length": 4, "type": "text", '
            '"turn": 1, "hash_ids": [42, 99]}',
            '{"chat_id": 2, "parent_chat_id": 1, "timestamp": 2.0, '
            '"input_length": 8, "output_length": 4, "type": "text", '
            '"turn": 2, "hash_ids": [42, 99]}',
        ]
        f = _write_jsonl(tmp_path, "bailian_repeat.jsonl", lines)
        loader = self._make_loader(f, real_prompt_generator, default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        prompts = [t.texts[0].contents[0] for t in conversations[0].turns]
        assert prompts[0] == prompts[1]


class TestBurstGPTCrossFileContent:
    """Cross-file collision regression for BurstGPTTraceDatasetLoader."""

    def _make_loader(
        self, filename: str, pg, user_config
    ) -> BurstGPTTraceDatasetLoader:
        return BurstGPTTraceDatasetLoader(
            filename=filename,
            run=user_config,
            prompt_generator=pg,
            default_block_size=_TEST_BLOCK_SIZE,
        )

    def test_burst_gpt_load_clears_cache_between_files(
        self, tmp_path, real_prompt_generator, default_user_config
    ):
        real_prompt_generator._cache[1] = [9999, 9998]

        f_a = _write_burst_csv(tmp_path, "burst_a.csv", [(1.0, 8, 4), (2.0, 8, 4)])
        loader_a = self._make_loader(f_a, real_prompt_generator, default_user_config)
        loader_a.load_dataset()

        assert 1 not in real_prompt_generator._cache

    def test_burst_gpt_trace_id_changes_between_files(
        self, tmp_path, real_prompt_generator, default_user_config
    ):
        f_a = _write_burst_csv(tmp_path, "burst_a.csv", [(1.0, 8, 4)])
        f_b = _write_burst_csv(tmp_path, "burst_b.csv", [(99.0, 8, 4)])

        loader_a = self._make_loader(f_a, real_prompt_generator, default_user_config)
        loader_a.load_dataset()
        trace_id_a = loader_a._trace_id

        loader_b = self._make_loader(f_b, real_prompt_generator, default_user_config)
        loader_b.load_dataset()
        trace_id_b = loader_b._trace_id

        assert trace_id_a != trace_id_b
        assert real_prompt_generator._hash_id_corpus_rng._trace_id == trace_id_b
