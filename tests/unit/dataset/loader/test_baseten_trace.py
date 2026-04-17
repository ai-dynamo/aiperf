# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

from aiperf.common.config import EndpointConfig, InputConfig, UserConfig
from aiperf.dataset.loader.baseten_trace import BasetenTraceDatasetLoader
from aiperf.dataset.loader.models import BasetenTrace


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def _mock_prompt_generator() -> Mock:
    generator = Mock()
    generator._decoded_cache = {}
    generator.tokenizer.resolved_name = "test-tokenizer"
    return generator


class TestBasetenTraceDatasetLoader:
    def test_can_load_parquet_schema(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 10,
                    "prompt": "hello",
                    "input_tokens": 3,
                    "output_tokens": 4,
                }
            ],
        )

        assert BasetenTraceDatasetLoader.can_load(filename=path) is True

    def test_can_load_parquet_schema_without_optional_columns(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 10,
                    "prompt": "hello",
                    "input_tokens": 3,
                    "output_tokens": 4,
                }
            ],
        )

        assert BasetenTraceDatasetLoader.can_load(filename=path) is True

    def test_load_dataset_normalizes_timestamps_and_groups_sessions(
        self, tmp_path: Path
    ):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 250,
                    "prompt": "second prompt",
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_hashes": [7, 8],
                    "provided_session_id": "1",
                    "poor_man_session_id": 42,
                    "request_canceled": 0,
                    "block_size": 64,
                },
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "first prompt",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_hashes": [1, 2],
                    "provided_session_id": "2",
                    "poor_man_session_id": 42,
                    "request_canceled": 1,
                    "block_size": 64,
                },
                {
                    "timestamp_start_unix_ms": 300,
                    "prompt": "other session prompt",
                    "input_tokens": 15,
                    "output_tokens": 6,
                    "total_hashes": [9],
                    "provided_session_id": "3",
                    "poor_man_session_id": 99,
                    "request_canceled": 0,
                    "block_size": 64,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            user_config=UserConfig(endpoint=EndpointConfig(model_names=["test-model"])),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert list(dataset.keys()) == ["42", "99"]
        assert [trace.timestamp for trace in dataset["42"]] == [0, 150]
        assert dataset["42"][0].text_input == "first prompt"
        assert dataset["42"][0].request_canceled == 1
        assert dataset["42"][0].request_body == {
            "min_tokens": 5,
            "hash_ids": [1, 2],
            "block_size": 64,
        }

    def test_convert_to_conversations_uses_literal_prompts(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "literal prompt",
                    "input_tokens": 10,
                    "output_tokens": 12,
                    "total_hashes": [11, 12],
                    "provided_session_id": "a",
                    "poor_man_session_id": 7,
                    "block_size": 64,
                }
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            user_config=UserConfig(endpoint=EndpointConfig(model_names=["test-model"])),
            prompt_generator=_mock_prompt_generator(),
        )

        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert turn.texts[0].contents == ["literal prompt"]
        assert turn.max_tokens == 12
        assert turn.request_body == {
            "min_tokens": 12,
            "hash_ids": [11, 12],
            "block_size": 64,
        }

    def test_sessions_are_ordered_by_first_timestamp_not_session_id(
        self, tmp_path: Path
    ):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 300,
                    "prompt": "later session",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 1,
                },
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "earlier session",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 999,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            user_config=UserConfig(endpoint=EndpointConfig(model_names=["test-model"])),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert list(dataset.keys()) == ["999", "1"]

    def test_trace_session_sample_ratio_samples_whole_sessions(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "s1-t1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 100,
                },
                {
                    "timestamp_start_unix_ms": 200,
                    "prompt": "s1-t2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 100,
                },
                {
                    "timestamp_start_unix_ms": 300,
                    "prompt": "s2-t1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 200,
                },
                {
                    "timestamp_start_unix_ms": 400,
                    "prompt": "s2-t2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 200,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            user_config=UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(trace_session_sample_ratio=0.01, random_seed=7),
            ),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert len(dataset) == 1
        kept_session = next(iter(dataset.values()))
        assert len(kept_session) == 2

    def test_trace_session_sampling_skips_when_no_effective_session_key(
        self, tmp_path: Path
    ):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "row-1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "unique-1",
                },
                {
                    "timestamp_start_unix_ms": 200,
                    "prompt": "row-2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "unique-2",
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            user_config=UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(trace_session_sample_ratio=0.01, random_seed=7),
            ),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert len(dataset) == 2

    def test_trace_session_sampling_falls_back_to_poor_man_session_id(
        self, tmp_path: Path
    ):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "s1-t1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "unique-1",
                    "poor_man_session_id": 100,
                },
                {
                    "timestamp_start_unix_ms": 200,
                    "prompt": "s1-t2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "unique-2",
                    "poor_man_session_id": 100,
                },
                {
                    "timestamp_start_unix_ms": 300,
                    "prompt": "s2-t1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "unique-3",
                    "poor_man_session_id": 200,
                },
                {
                    "timestamp_start_unix_ms": 400,
                    "prompt": "s2-t2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "unique-4",
                    "poor_man_session_id": 200,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            user_config=UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(trace_session_sample_ratio=0.01, random_seed=7),
            ),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert len(dataset) == 1
        kept_session = next(iter(dataset.values()))
        assert len(kept_session) == 2


class TestBasetenTraceModel:
    def test_model_maps_alias_fields(self):
        trace = BasetenTrace(
            timestamp_start_unix_ms=123,
            prompt="hello",
            input_tokens=10,
            output_tokens=20,
            total_hashes=[1, 2],
            __version__="0.0.11",
        )

        assert trace.dataset_version == "0.0.11"
