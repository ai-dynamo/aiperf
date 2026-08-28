# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.environment import Environment
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader import baseten_trace as baseten_trace_module
from aiperf.dataset.loader.baseten_trace import (
    BasetenTrace,
    BasetenTraceDatasetLoader,
    count_baseten_parquet_records_and_sessions,
)
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def _write_arrow(path: Path, rows: list[dict], batch_size: int = 2) -> Path:
    table = pa.Table.from_pylist(rows)
    with (
        pa.OSFile(str(path), "wb") as sink,
        ipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table, max_chunksize=batch_size)
    return path


def _mock_prompt_generator() -> Mock:
    generator = Mock()
    generator._decoded_cache = {}
    generator.tokenizer.resolved_name = "test-tokenizer"
    return generator


def _make_run(input_file: str | Path | None = None, **kwargs):
    if input_file is not None:
        kwargs.setdefault("input_file", str(input_file))
        kwargs.setdefault("custom_dataset_type", CustomDatasetType.BASETEN_TRACE)
    return make_run_from_cli(CLIConfig(model_names=["test-model"], **kwargs))


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
                    "total_hashes": [1, 2],
                    "provided_session_id": "s",
                    "poor_man_session_id": 1,
                    "block_size": 64,
                    "request_canceled": 0,
                }
            ],
        )

        assert BasetenTraceDatasetLoader.can_load(filename=path) is True

    def test_can_load_parquet_schema_without_optional_columns(self, tmp_path: Path):
        # Only the four REQUIRED_COLUMNS; every optional column is absent.
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET,
            "BASETEN_SESSION_COLUMN",
            "poor_man_session_id",
        )
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
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert list(dataset.keys()) == ["42", "99"]
        assert [trace.timestamp for trace in dataset["42"]] == [0, 150]
        assert dataset["42"][0].text_input == "first prompt"
        # No "prompt" override: the completions endpoint emits single-prompt
        # payloads as bare strings itself.
        assert dataset["42"][0].request_body == {
            "min_tokens": 5,
            "hash_ids": [1, 2],
            "block_size": 64,
        }

    @pytest.mark.parametrize("suffix", [".arrow", ".ipc"])
    def test_arrow_ipc_load_matches_parquet(self, tmp_path: Path, suffix: str) -> None:
        rows = [
            {
                "timestamp_start_unix_ms": timestamp,
                "prompt": f"prompt-{index}",
                "input_tokens": index + 1,
                "output_tokens": index,
                "total_hashes": [index, index + 1],
                "poor_man_session_id": index // 2,
                "block_size": 64,
            }
            for index, timestamp in enumerate(range(100, 500, 10), start=1)
        ]
        parquet_path = _write_parquet(tmp_path / "trace.parquet", rows)
        arrow_path = _write_arrow(tmp_path / f"trace{suffix}", rows, batch_size=3)

        def load(path: Path) -> dict[str, list[dict]]:
            loader = BasetenTraceDatasetLoader(
                filename=str(path),
                run=_make_run(
                    path,
                    replay_speedup=2.0,
                    trace_session_sample_ratio=0.1,
                    random_seed=7,
                ),
                prompt_generator=_mock_prompt_generator(),
            )
            return {
                session_id: [trace.model_dump() for trace in traces]
                for session_id, traces in loader.load_dataset().items()
            }

        assert BasetenTraceDatasetLoader.can_load(filename=arrow_path) is True
        assert load(arrow_path) == load(parquet_path)

    def test_count_arrow_ipc_records_and_sessions(self, tmp_path: Path) -> None:
        path = _write_arrow(
            tmp_path / "trace.arrow",
            [
                {
                    "timestamp_start_unix_ms": index,
                    "prompt": str(index),
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "poor_man_session_id": session_id,
                }
                for index, session_id in enumerate((1, 1, 2, None))
            ],
        )

        assert count_baseten_parquet_records_and_sessions(str(path)) == (4, 3)

    def test_arrow_ipc_opens_once_per_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_arrow(
            tmp_path / "trace.arrow",
            [
                {
                    "timestamp_start_unix_ms": index,
                    "prompt": str(index),
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
                for index in range(12)
            ],
            batch_size=2,
        )
        run = _make_run(path)
        original_open = baseten_trace_module._open_arrow_ipc
        open_calls = 0

        def counting_open(file_path: str | Path) -> Any:
            nonlocal open_calls
            open_calls += 1
            return original_open(file_path)

        monkeypatch.setattr(baseten_trace_module, "_open_arrow_ipc", counting_open)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=run,
            prompt_generator=_mock_prompt_generator(),
        )

        loader.load_dataset()

        assert open_calls == 3

    def test_parquet_opens_once_per_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": index,
                    "prompt": str(index),
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
                for index in range(12)
            ],
        )
        run = _make_run(path)
        parquet_file_class = pq.ParquetFile
        open_calls = 0

        def counting_open(*args: Any, **kwargs: Any) -> Any:
            nonlocal open_calls
            open_calls += 1
            return parquet_file_class(*args, **kwargs)

        monkeypatch.setattr(baseten_trace_module.pq, "ParquetFile", counting_open)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=run,
            prompt_generator=_mock_prompt_generator(),
        )

        loader.load_dataset()

        assert open_calls == 1

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
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )

        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert turn.texts[0].contents == ["literal prompt"]
        assert turn.max_tokens == 12
        assert turn.extra_body == {
            "min_tokens": 12,
            "hash_ids": [11, 12],
            "block_size": 64,
        }

    def test_convert_to_conversations_grouped_session_is_self_contained(
        self, tmp_path: Path
    ):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "this is turn 1",
                    "input_tokens": 10,
                    "output_tokens": 12,
                    "poor_man_session_id": 7,
                },
                {
                    "timestamp_start_unix_ms": 200,
                    "prompt": "this is turn 1, response to turn 1, and now turn 2 with all context included.",
                    "input_tokens": 22,
                    "output_tokens": 13,
                    "poor_man_session_id": 7,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )

        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert (
            conversations[0].context_mode
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        assert len(conversations[0].turns) == 2

    def _write_multi_turn(self, tmp_path: Path) -> Path:
        return _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 1_000,
                    "prompt": "A-1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "A",
                },
                {
                    "timestamp_start_unix_ms": 3_000,
                    "prompt": "A-2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "A",
                },
                {
                    "timestamp_start_unix_ms": 2_000,
                    "prompt": "B-1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "provided_session_id": "B",
                },
            ],
        )

    def test_convert_to_conversations_open_loop_strict_explodes_rows(
        self, tmp_path: Path
    ):
        path = self._write_multi_turn(tmp_path)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, open_loop_replay=True, open_loop_strict=True),
            prompt_generator=_mock_prompt_generator(),
        )

        conversations = loader.convert_to_conversations(loader.load_dataset())

        assert [len(conv.turns) for conv in conversations] == [1, 1, 1]
        assert all(conv.context_mode is None for conv in conversations)
        by_prompt = {
            conv.turns[0].texts[0].contents[0]: conv.turns[0] for conv in conversations
        }
        # Absolute (normalized) recorded timestamps are kept; no delays.
        assert {p: t.timestamp for p, t in by_prompt.items()} == {
            "A-1": 0,
            "B-1": 1_000,
            "A-2": 2_000,
        }
        assert all(turn.delay is None for turn in by_prompt.values())

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
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        ordered_prompts = [traces[0].text_input for traces in dataset.values()]
        assert ordered_prompts == ["earlier session", "later session"]

    def _write_null_session_mix(self, tmp_path: Path) -> Path:
        return _write_parquet(
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
                {
                    "timestamp_start_unix_ms": 500,
                    "prompt": "null-1",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": None,
                },
                {
                    "timestamp_start_unix_ms": 600,
                    "prompt": "null-2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": None,
                },
            ],
        )

    def test_trace_session_sampling_keeps_null_session_rows(self, tmp_path: Path):
        path = self._write_null_session_mix(tmp_path)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, trace_session_sample_ratio=0.9999, random_seed=7),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        prompts = sorted(
            trace.text_input for traces in dataset.values() for trace in traces
        )
        assert prompts == ["null-1", "null-2", "s1-t1", "s1-t2", "s2-t1", "s2-t2"]
        # Null-session rows become synthesized single-turn sessions.
        assert sorted(len(traces) for traces in dataset.values()) == [1, 1, 2, 2]

    def test_trace_session_sampling_mid_ratio_keeps_exact_null_subset(
        self, tmp_path: Path
    ):
        rows = [
            {
                "timestamp_start_unix_ms": 100 + i,
                "prompt": f"s1-t{i}",
                "input_tokens": 5,
                "output_tokens": 1,
                "poor_man_session_id": 100,
            }
            for i in range(2)
        ] + [
            {
                "timestamp_start_unix_ms": 200 + i,
                "prompt": f"null-{i:02d}",
                "input_tokens": 5,
                "output_tokens": 1,
                "poor_man_session_id": None,
            }
            for i in range(20)
        ]
        path = _write_parquet(tmp_path / "trace.parquet", rows)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, trace_session_sample_ratio=0.5, random_seed=7),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        null_prompts = sorted(
            trace.text_input
            for traces in dataset.values()
            for trace in traces
            if trace.text_input.startswith("null-")
        )
        # A proper subset of the 20 null rows: a regression to keep-all or
        # drop-all null rows fails here.
        assert 0 < len(null_prompts) < 20

    def test_trace_session_sampling_null_rows_deterministic_across_loads(
        self, tmp_path: Path
    ):
        path = self._write_null_session_mix(tmp_path)

        def load() -> list[list[str]]:
            loader = BasetenTraceDatasetLoader(
                filename=str(path),
                run=_make_run(path, trace_session_sample_ratio=0.5, random_seed=7),
                prompt_generator=_mock_prompt_generator(),
            )
            return [
                [trace.text_input for trace in traces]
                for traces in loader.load_dataset().values()
            ]

        first, second = load(), load()

        assert first == second
        assert first, "mid-ratio sampling should keep at least one session"

    def test_trace_session_sampling_uses_provided_session_id_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET,
            "BASETEN_SESSION_COLUMN",
            "poor_man_session_id",
        )
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
            run=_make_run(path, trace_session_sample_ratio=0.01, random_seed=7),
            prompt_generator=_mock_prompt_generator(),
        )

        parquet_file_class = pq.ParquetFile

        class NoReadParquetFile:
            def __init__(self, *args, **kwargs) -> None:
                self._parquet_file = parquet_file_class(*args, **kwargs)

            def __getattr__(self, name: str):
                if name == "read":
                    raise AssertionError("sampled metadata must use bounded batches")
                return getattr(self._parquet_file, name)

        monkeypatch.setattr(baseten_trace_module.pq, "ParquetFile", NoReadParquetFile)

        dataset = loader.load_dataset()

        assert list(dataset) == ["unique-1"]
        assert sorted(
            trace.text_input for traces in dataset.values() for trace in traces
        ) == ["row-1"]

    def test_trace_session_sampling_uses_configured_poor_man_session_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET,
            "BASETEN_SESSION_COLUMN",
            "poor_man_session_id",
        )
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
            run=_make_run(path, trace_session_sample_ratio=0.01, random_seed=7),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        assert len(dataset) == 1
        kept_session = next(iter(dataset.values()))
        assert len(kept_session) == 2

    def test_sampling_and_grouping_use_same_session_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET,
            "BASETEN_SESSION_COLUMN",
            "poor_man_session_id",
        )

        # Sampling and grouping must reuse the configured key; switching after
        # filtering would shred sessions and regroup null-key rows.
        # Fixture shape is pinned to the sampling RNG (root seed 42 from the
        # autouse fixture): 7 poor_man sessions draw the first 7 uniforms (pairB
        # dropped, pairA kept, singletons eat the rest), then the 3 null-poor_man
        # "s0" rows draw the next 3 (t0 kept, t1 dropped, t2 kept).
        def row(ts: int, prompt: str, poor: int | None, provided: str | None = None):
            return {
                "timestamp_start_unix_ms": ts,
                "prompt": prompt,
                "input_tokens": 5,
                "output_tokens": 1,
                "provided_session_id": provided,
                "poor_man_session_id": poor,
            }

        rows = [
            row(0, "pairB-t0", 20),
            row(10, "pairB-t1", 20),
            row(100, "pairA-t0", 10),
            row(110, "pairA-t1", 10),
            *(row(200 + 100 * i, f"single-{i}", 31 + i) for i in range(5)),
            *(row(1_000 + 100 * i, f"s0-t{i}", None, "s0") for i in range(3)),
        ]
        path = _write_parquet(tmp_path / "trace.parquet", rows)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, trace_session_sample_ratio=0.4),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        session_prompts = [
            [trace.text_input for trace in traces] for traces in dataset.values()
        ]
        kept = {prompt for prompts in session_prompts for prompt in prompts}
        # Guard against fixture/RNG drift: sampling must actually filter.
        assert "pairB-t0" not in kept and "pairA-t0" in kept and "s0-t0" in kept
        # The sampled poor_man pair stays whole in ONE session.
        assert ["pairA-t0", "pairA-t1"] in session_prompts
        # Row-sampled null-poor_man rows stay synthesized single-turn sessions;
        # they must never be regrouped into a multi-turn session with holes.
        for prompts in session_prompts:
            if any(prompt.startswith("s0-") for prompt in prompts):
                assert len(prompts) == 1

    @pytest.mark.parametrize(
        ("session_column", "expected_sessions"),
        [
            param("provided_session_id", 4, id="provided"),
            param("poor_man_session_id", 2, id="poor-man"),
        ],
    )  # fmt: skip
    def test_resolver_session_count_uses_same_key_as_loader(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        session_column: str,
        expected_sessions: int,
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET,
            "BASETEN_SESSION_COLUMN",
            session_column,
        )
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

        assert count_baseten_parquet_records_and_sessions(str(path)) == (
            4,
            expected_sessions,
        )

    def test_fixed_schedule_offsets_filter_relative_window_on_unix_timestamps(
        self, tmp_path: Path
    ):
        # fixed_schedule_start_offset/end_offset are inclusive bounds [start, end],
        # applied to normalized trace-relative timestamps (first row becomes t=0).
        base_ts_ms = 1_730_000_000_000
        one_hour_ms = 60 * 60 * 1000
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": base_ts_ms,
                    "prompt": "hour0",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 1,
                },
                {
                    "timestamp_start_unix_ms": base_ts_ms + (2 * one_hour_ms),
                    "prompt": "hour2",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 2,
                },
                {
                    "timestamp_start_unix_ms": base_ts_ms + (3 * one_hour_ms),
                    "prompt": "hour3",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 3,
                },
                {
                    "timestamp_start_unix_ms": base_ts_ms + (4 * one_hour_ms),
                    "prompt": "hour4",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": 4,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(
                path,
                fixed_schedule=True,
                fixed_schedule_start_offset=2 * one_hour_ms,
                fixed_schedule_end_offset=3 * one_hour_ms,
            ),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()
        traces = [trace for session in dataset.values() for trace in session]

        assert sorted(trace.text_input for trace in traces) == ["hour2", "hour3"]
        assert sorted(int(trace.timestamp or 0) for trace in traces) == [
            2 * one_hour_ms,
            3 * one_hour_ms,
        ]

    def test_offset_window_selects_recorded_time_under_replay_speedup(
        self, tmp_path: Path
    ):
        one_minute_ms = 60_000
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 1_000_000 + minute * one_minute_ms,
                    "prompt": f"minute{minute}",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "poor_man_session_id": minute,
                }
                for minute in range(10)
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(
                path,
                fixed_schedule=True,
                fixed_schedule_end_offset=one_minute_ms,
                replay_speedup=10.0,
            ),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()
        traces = [trace for session in dataset.values() for trace in session]

        # The end offset selects the first RECORDED minute (inclusive bounds);
        # the kept timestamps are then compressed by replay_speedup.
        assert sorted(trace.text_input for trace in traces) == ["minute0", "minute1"]
        assert sorted(trace.timestamp for trace in traces) == [0, 6_000]

    def _write_hinted_single_row(self, tmp_path: Path) -> Path:
        return _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "hinted",
                    "input_tokens": 5,
                    "output_tokens": 4,
                    "total_hashes": [1, 2],
                    "block_size": 64,
                }
            ],
        )

    @pytest.mark.parametrize(
        "omit_kv_hints, expected_body",
        [
            param(
                False,
                {"min_tokens": 4, "hash_ids": [1, 2], "block_size": 64},
                id="hints_present",
            ),
            param(True, {"min_tokens": 4}, id="hints_omitted"),
        ],
    )  # fmt: skip
    def test_set_request_body_omit_kv_hints_controls_cache_hints(
        self, tmp_path: Path, omit_kv_hints: bool, expected_body: dict
    ):
        path = self._write_hinted_single_row(tmp_path)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, omit_kv_hints=omit_kv_hints),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        trace = next(iter(dataset.values()))[0]
        assert trace.request_body == expected_body

    @pytest.mark.parametrize(
        "force_min_tokens, expected_body",
        [
            param(
                True,
                {"min_tokens": 4, "hash_ids": [1, 2], "block_size": 64},
                id="min_tokens_pinned",
            ),
            param(
                False,
                {"hash_ids": [1, 2], "block_size": 64},
                id="min_tokens_not_set",
            ),
        ],
    )  # fmt: skip
    def test_set_request_body_force_min_tokens_controls_min_tokens(
        self, tmp_path: Path, force_min_tokens: bool, expected_body: dict
    ):
        path = self._write_hinted_single_row(tmp_path)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, force_min_tokens=force_min_tokens),
            prompt_generator=_mock_prompt_generator(),
        )

        conversations = loader.convert_to_conversations(loader.load_dataset())

        turn = conversations[0].turns[0]
        assert turn.extra_body == expected_body
        # max_tokens handling is untouched by the min_tokens gate.
        assert turn.max_tokens == 4

    def test_load_dataset_zero_output_tokens_floored_to_one(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "canceled row",
                    "input_tokens": 5,
                    "output_tokens": 0,
                    "request_canceled": 1,
                },
                {
                    "timestamp_start_unix_ms": 200,
                    "prompt": "normal row",
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "request_canceled": 0,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )

        conversations = loader.convert_to_conversations(loader.load_dataset())

        turns = {
            turn.texts[0].contents[0]: turn
            for conv in conversations
            for turn in conv.turns
        }
        assert turns["canceled row"].max_tokens == 1
        assert turns["normal row"].max_tokens == 7

    @pytest.mark.parametrize(
        ("run_kwargs", "expected_columns"),
        [
            param(
                {},
                {
                    "block_size",
                    "cached_tokens_reference",
                    "duration_e2e_ms",
                    "duration_ttft_ms",
                    "input_tokens",
                    "output_tokens",
                    "poor_man_session_id",
                    "prompt",
                    "timestamp_start_unix_ms",
                    "total_hashes",
                },
                id="default",
            ),
            param(
                {"omit_kv_hints": True},
                {
                    "cached_tokens_reference",
                    "duration_e2e_ms",
                    "duration_ttft_ms",
                    "input_tokens",
                    "output_tokens",
                    "poor_man_session_id",
                    "prompt",
                    "timestamp_start_unix_ms",
                },
                id="omit-kv-hints",
            ),
            param(
                {"open_loop_replay": False},
                {
                    "block_size",
                    "cached_tokens_reference",
                    "duration_e2e_ms",
                    "duration_ttft_ms",
                    "input_tokens",
                    "output_tokens",
                    "poor_man_session_id",
                    "prompt",
                    "timestamp_start_unix_ms",
                    "total_hashes",
                },
                id="closed-loop",
            ),
        ],
    )  # fmt: skip
    def test_load_dataset_validates_sample_and_skips_unused_columns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        run_kwargs: dict[str, bool],
        expected_columns: set[str],
    ) -> None:
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": index,
                    "prompt": f"prompt-{index}",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "output_text": "unused completion",
                    "model_name": "unused model",
                    "request_canceled": 1,
                    "__version__": "unused version",
                    "poor_man_session_id": 1,
                    "total_hashes": [1, 2],
                    "block_size": 64,
                    "duration_e2e_ms": 10,
                    "duration_ttft_ms": 4,
                    "cached_tokens_reference": 2,
                }
                for index in range(12)
            ],
        )
        original_validate = BasetenTrace.model_validate
        original_iter_batches = pq.ParquetFile.iter_batches
        validation_calls = 0
        requested_columns: list[set[str]] = []

        def count_validation_calls(row: dict) -> BasetenTrace:
            nonlocal validation_calls
            validation_calls += 1
            return original_validate(row)

        def capture_columns(parquet_file, *args, **kwargs):
            requested_columns.append(set(kwargs["columns"]))
            return original_iter_batches(parquet_file, *args, **kwargs)

        monkeypatch.setattr(BasetenTrace, "model_validate", count_validation_calls)
        monkeypatch.setattr(pq.ParquetFile, "iter_batches", capture_columns)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, **run_kwargs),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()
        traces = [trace for session in dataset.values() for trace in session]

        assert validation_calls == 10
        assert len(traces) == 12
        assert requested_columns == [expected_columns]
        assert all(not hasattr(trace, "output_text") for trace in traces)
        assert all(not hasattr(trace, "model_name") for trace in traces)
        assert all(not hasattr(trace, "request_canceled") for trace in traces)
        assert all(not hasattr(trace, "dataset_version") for trace in traces)
        # Recorded outcomes must survive load for later fidelity comparison,
        # regardless of replay mode.
        assert all(trace.duration_e2e_ms == 10 for trace in traces)
        assert all(trace.duration_ttft_ms == 4 for trace in traces)
        assert all(trace.cached_tokens_reference == 2 for trace in traces)

    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            param("timestamp_start_unix_ms", -1, id="negative-timestamp"),
            param("input_tokens", None, id="null-input-tokens"),
            param("output_tokens", -1, id="negative-output-tokens"),
            param("prompt", None, id="null-prompt"),
        ],
    )  # fmt: skip
    def test_load_dataset_validates_malformed_required_field_after_sample(
        self,
        tmp_path: Path,
        field: str,
        invalid_value: object,
    ) -> None:
        rows = [
            {
                "timestamp_start_unix_ms": index,
                "prompt": f"prompt-{index}",
                "input_tokens": 5,
                "output_tokens": 1,
            }
            for index in range(11)
        ]
        rows[-1][field] = invalid_value
        path = _write_parquet(tmp_path / "trace.parquet", rows)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path),
            prompt_generator=_mock_prompt_generator(),
        )

        with pytest.raises(ValidationError, match=field):
            loader.load_dataset()

    def test_load_dataset_normalizes_null_hashes_after_validation_sample(
        self, tmp_path: Path
    ) -> None:
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": index,
                    "prompt": f"prompt-{index}",
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "total_hashes": [1] if index < 10 else None,
                }
                for index in range(11)
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path),
            prompt_generator=_mock_prompt_generator(),
        )

        traces = [
            trace for session in loader.load_dataset().values() for trace in session
        ]

        assert traces[-1].total_hashes == []

    def test_request_body_uses_capped_output_length(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "cap me",
                    "input_tokens": 5,
                    "output_tokens": 10,
                    "poor_man_session_id": 1,
                },
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, synthesis_max_osl=3),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()
        trace = next(iter(next(iter(dataset.values()))))

        assert trace.output_length == 3
        assert trace.request_body == {"min_tokens": 3}

    def test_synthesis_speedup_ratio_rejected(self, tmp_path: Path):
        path = self._write_hinted_single_row(tmp_path)
        # No custom_dataset_type: the auto-detected path must be rejected by
        # the loader itself, not only by explicit CLI-flag validation.
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(path),
                synthesis_speedup_ratio=10.0,
            )
        )

        with pytest.raises(ValueError, match="--replay-speedup"):
            BasetenTraceDatasetLoader(
                filename=str(path),
                run=run,
                prompt_generator=_mock_prompt_generator(),
            )

    def test_non_speedup_synthesis_still_loads(self, tmp_path: Path):
        path = self._write_hinted_single_row(tmp_path)
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, synthesis_output_len_multiplier=2.0),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()

        trace = next(iter(dataset.values()))[0]
        assert trace.output_length == 8

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            param("synthesis_prefix_len_multiplier", 2.0, id="prefix-len"),
            param("synthesis_prefix_root_multiplier", 2, id="prefix-root"),
            param("synthesis_prompt_len_multiplier", 2.0, id="prompt-len"),
        ],
    )  # fmt: skip
    def test_hash_reshaping_synthesis_rejected(
        self, tmp_path: Path, field: str, value: float | int
    ):
        path = self._write_hinted_single_row(tmp_path)
        # No custom_dataset_type: the auto-detected path must be rejected by
        # the loader itself, not only by explicit CLI-flag validation.
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(path),
                **{field: value},
            )
        )

        with pytest.raises(ValueError, match="hash_ids"):
            BasetenTraceDatasetLoader(
                filename=str(path),
                run=run,
                prompt_generator=_mock_prompt_generator(),
            )

    def test_output_len_synthesis_keeps_hash_ids_verbatim(self, tmp_path: Path):
        # The wire still sends the recorded prompt under output-length
        # synthesis, so the forwarded KV hints must stay the recorded ones.
        path = _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "hinted",
                    "input_tokens": 5,
                    "output_tokens": 4,
                    "total_hashes": [1, 2, 3],
                    "block_size": 64,
                }
            ],
        )
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(path, synthesis_output_len_multiplier=2.0),
            prompt_generator=_mock_prompt_generator(),
        )

        dataset = loader.load_dataset()
        trace = next(iter(dataset.values()))[0]

        assert trace.request_body["hash_ids"] == [1, 2, 3]
        assert trace.input_length == 5
        assert trace.output_length == 8


class TestBasetenTraceModel:
    def test_model_accepts_null_hashes_and_numeric_provided_session_id(self):
        trace = BasetenTrace(
            timestamp_start_unix_ms=123,
            prompt="hello",
            input_tokens=10,
            output_tokens=20,
            total_hashes=None,
            provided_session_id=42,
        )

        assert trace.total_hashes == []
        assert trace.provided_session_id == 42


class TestSynthesisHooks:
    def test_synthesis_exclude_fields(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "trace.parquet", [])
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )

        excluded = loader._synthesis_exclude_fields()

        assert "prompt" in excluded
        assert "request_body" in excluded
        assert "provided_session_id" in excluded
        assert "poor_man_session_id" in excluded
        assert "total_hashes" in excluded

    def test_reconstruct_traces_preserves_original_metadata(
        self, tmp_path: Path
    ) -> None:
        path = _write_parquet(tmp_path / "trace.parquet", [])
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )
        originals = [
            BasetenTrace(
                timestamp_start_unix_ms=100,
                prompt="original",
                input_tokens=10,
                output_tokens=20,
                poor_man_session_id=7,
                total_hashes=[1, 2],
                block_size=64,
            )
        ]
        synth_dicts = [
            {"timestamp": 5, "input_length": 50, "output_length": 60},
        ]

        result = loader._reconstruct_traces(originals, synth_dicts)

        assert len(result) == 1
        assert result[0].timestamp == 5
        assert result[0].input_length == 50
        assert result[0].output_length == 60
        # Request bodies are built once in load_dataset, after the max-OSL cap.
        assert result[0].request_body is None
        assert result[0].prompt == "original"
        assert result[0].poor_man_session_id == 7
        assert result[0].total_hashes == [1, 2]

    def test_reconstruct_traces_uses_last_original_for_extra_synth_rows(
        self, tmp_path: Path
    ) -> None:
        path = _write_parquet(tmp_path / "trace.parquet", [])
        loader = BasetenTraceDatasetLoader(
            filename=str(path),
            run=_make_run(),
            prompt_generator=_mock_prompt_generator(),
        )
        originals = [
            BasetenTrace(
                timestamp_start_unix_ms=100,
                prompt="only-original",
                input_tokens=10,
                output_tokens=20,
                poor_man_session_id=7,
            )
        ]
        synth_dicts = [
            {"timestamp": 5, "input_length": 50, "output_length": 60},
            {"timestamp": 6, "input_length": 51, "output_length": 61},
        ]

        result = loader._reconstruct_traces(originals, synth_dicts)

        assert len(result) == 2
        assert result[1].timestamp == 6
        assert result[1].input_length == 51
        assert result[1].output_length == 61
        assert result[1].prompt == "only-original"
        assert result[1].poor_man_session_id == 7


class TestExtraInputsCollisionGuardAutoDetect:
    """The CLI converter guard (_reject_baseten_trace_extra_input_collisions)
    only sees explicit --custom-dataset-type baseten_trace; auto-detected
    Parquet traces bypass it, so the loader mirrors the same rejection at
    construction time."""

    def _write_single_row(self, tmp_path: Path) -> Path:
        return _write_parquet(
            tmp_path / "trace.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "hello",
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_hashes": [1, 2],
                    "block_size": 64,
                }
            ],
        )

    def _autodetect_run(self, path: Path, **kwargs):
        # No custom_dataset_type: mirrors `aiperf ... --input-file x.parquet`
        # with type auto-detection, which the converter guard cannot see.
        return make_run_from_cli(
            CLIConfig(model_names=["test-model"], input_file=str(path), **kwargs)
        )

    def _make_loader(self, path: Path, **cli_kwargs) -> BasetenTraceDatasetLoader:
        return BasetenTraceDatasetLoader(
            filename=str(path),
            run=self._autodetect_run(path, **cli_kwargs),
            prompt_generator=_mock_prompt_generator(),
        )

    def test_min_tokens_collision_rejected_on_auto_detected_trace(
        self, tmp_path: Path
    ) -> None:
        path = self._write_single_row(tmp_path)
        with pytest.raises(
            ValueError,
            match="--extra-inputs min_tokens is overwritten per-turn by the "
            "baseten_trace loader; pass --no-force-min-tokens to send your value",
        ):
            self._make_loader(path, extra_inputs=["min_tokens:5"])

    def test_min_tokens_collision_cleared_by_no_force_min_tokens(
        self, tmp_path: Path
    ) -> None:
        path = self._write_single_row(tmp_path)
        loader = self._make_loader(
            path, extra_inputs=["min_tokens:5"], force_min_tokens=False
        )

        dataset = loader.load_dataset()

        trace = next(iter(dataset.values()))[0]
        assert "min_tokens" not in trace.request_body

    def test_hash_ids_collision_rejected_on_auto_detected_trace(
        self, tmp_path: Path
    ) -> None:
        path = self._write_single_row(tmp_path)
        with pytest.raises(
            ValueError,
            match="--extra-inputs hash_ids is overwritten per-turn by the "
            "baseten_trace loader; pass --omit-kv-hints to send your value",
        ):
            self._make_loader(path, extra_inputs=["hash_ids:999"])

    def test_hash_ids_collision_cleared_by_omit_kv_hints(self, tmp_path: Path) -> None:
        path = self._write_single_row(tmp_path)
        loader = self._make_loader(
            path, extra_inputs=["hash_ids:999"], omit_kv_hints=True
        )

        dataset = loader.load_dataset()

        trace = next(iter(dataset.values()))[0]
        assert "hash_ids" not in trace.request_body

    def test_block_size_collision_rejected_on_auto_detected_trace(
        self, tmp_path: Path
    ) -> None:
        path = self._write_single_row(tmp_path)
        with pytest.raises(
            ValueError,
            match="--extra-inputs block_size is overwritten per-turn by the "
            "baseten_trace loader; pass --omit-kv-hints to send your value",
        ):
            self._make_loader(path, extra_inputs=["block_size:16"])
