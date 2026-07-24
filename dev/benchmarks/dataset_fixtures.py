# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic local fixtures for the dataset-load comparison harness."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import orjson

from dev.benchmarks.dataset_format_catalog import (
    DEFAULT_BENCHMARK_ROWS,
    FORMAT_PROFILES,
    FormatCase,
    SourceEnvelope,
    shared_synthetic_inline,
)

DEFAULT_MODEL = "test-model"
BLOCK_SIZE = 64


def _write_jsonl(path: Path, records: Sequence[object]) -> None:
    path.write_bytes(b"".join(orjson.dumps(record) + b"\n" for record in records))


def _speed_bench_row(
    *,
    question_id: str,
    category: str = "coding",
    messages: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    if messages is None:
        messages = [{"role": "user", "content": "Implement binary search."}]
    return {
        "question_id": question_id.ljust(32, "0"),
        "category": category,
        "sub_category": None,
        "source": "https://example.test/speed-bench",
        "src_id": question_id.ljust(32, "0"),
        "difficulty": None,
        "multiturn": len(messages) > 1,
        "messages": messages,
    }


def _baseten_rows() -> list[dict[str, object]]:
    return [
        {
            "timestamp_start_unix_ms": 1_000,
            "prompt": "A-1 reconstructed prompt text",
            "input_tokens": 128,
            "output_tokens": 50,
            "total_hashes": [10, 11],
            "provided_session_id": "A",
            "poor_man_session_id": 1,
            "block_size": BLOCK_SIZE,
            "request_canceled": 0,
            "duration_e2e_ms": 800,
            "duration_ttft_ms": 120,
            "cached_tokens_reference": 0,
        },
        {
            "timestamp_start_unix_ms": 3_000,
            "prompt": "A-2 reconstructed prompt text",
            "input_tokens": 192,
            "output_tokens": 40,
            "total_hashes": [10, 11, 12],
            "provided_session_id": "A",
            "poor_man_session_id": 1,
            "block_size": BLOCK_SIZE,
            "request_canceled": 0,
            "duration_e2e_ms": 700,
            "duration_ttft_ms": 110,
            "cached_tokens_reference": 128,
        },
        {
            "timestamp_start_unix_ms": 10_000,
            "prompt": "B-1 reconstructed prompt text",
            "input_tokens": 96,
            "output_tokens": 30,
            "total_hashes": [20],
            "provided_session_id": "B",
            "poor_man_session_id": 2,
            "block_size": BLOCK_SIZE,
            "request_canceled": 0,
            "duration_e2e_ms": 600,
            "duration_ttft_ms": 100,
            "cached_tokens_reference": 0,
        },
    ]


def write_baseten_parquet(
    path: Path, rows: Sequence[dict[str, object]] | None = None
) -> None:
    """Write a minimal Baseten replay Parquet fixture."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "baseten_trace fixtures require pyarrow; install pyarrow to benchmark baseten_trace"
        ) from error
    table = pa.Table.from_pylist(list(rows or _baseten_rows()))
    pq.write_table(table, path)


def generate_local_fixtures(
    directory: Path,
    *,
    rows: int | None = None,
    tokens_per_row: int | None = None,
) -> dict[str, Path]:
    """Generate deterministic on-disk fixtures and return format → path mapping."""
    directory.mkdir(parents=True, exist_ok=True)
    if rows is None and tokens_per_row is None:
        single_turn = [
            {"text": "alpha beta gamma"},
            {"session_id": "s-a", "text": "turn one"},
            {"session_id": "s-a", "text": "turn two"},
        ]
        multi_turn = [
            {"session_id": "m1", "turns": [{"text": "q1"}, {"text": "q2"}]},
            {"session_id": "m2", "turns": [{"text": "only"}]},
        ]
        raw_payload = [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": DEFAULT_MODEL,
                "max_tokens": 16,
            },
            {
                "messages": [{"role": "user", "content": "bye"}],
                "model": DEFAULT_MODEL,
                "max_tokens": 16,
            },
        ]
        inputs_json = {
            "data": [
                {"session_id": "session-001", "payloads": raw_payload},
                {"session_id": "session-002", "payloads": [raw_payload[0]]},
            ]
        }
        random_pool = [{"text": "alpha beta gamma"}]
        mooncake_trace = [
            {"timestamp": 0, "text_input": "alpha beta gamma", "output_length": 16},
            {"timestamp": 1, "text_input": "turn one", "output_length": 16},
            {"timestamp": 2, "text_input": "turn two", "output_length": 16},
        ]
        bailian_trace = [
            {
                "chat_id": 1,
                "parent_chat_id": -1,
                "timestamp": 0,
                "input_length": 3,
                "output_length": 16,
                "type": "text",
                "turn": 1,
            },
            {
                "chat_id": 2,
                "parent_chat_id": 1,
                "timestamp": 1,
                "input_length": 2,
                "output_length": 16,
                "type": "text",
                "turn": 2,
            },
        ]
        burst_gpt_trace = [(0, 3, 16), (1, 2, 16)]
        sagemaker_texts = ["alpha beta gamma"]
        speed_bench_rows = [
            _speed_bench_row(question_id="speed-coding-1"),
            _speed_bench_row(
                question_id="speed-coding-2",
                messages=[
                    {"role": "user", "content": "First turn."},
                    {"role": "assistant", "content": "Acknowledged."},
                    {"role": "user", "content": "Second turn."},
                ],
            ),
        ]
    else:
        if rows is None or rows <= 0:
            raise ValueError("rows must be positive")
        if tokens_per_row is None or tokens_per_row <= 0:
            raise ValueError("tokens_per_row must be positive")
        text = " ".join(["token"] * tokens_per_row)
        single_turn = [
            {"session_id": f"single-{index:06d}", "text": text} for index in range(rows)
        ]
        multi_turn = [
            {"session_id": f"multi-{index:06d}", "turns": [{"text": text}]}
            for index in range(rows)
        ]
        raw_payload = [
            {
                "messages": [{"role": "user", "content": text}],
                "model": DEFAULT_MODEL,
                "max_tokens": 16,
            }
            for _ in range(rows)
        ]
        inputs_json = {"data": [{"session_id": "session-001", "payloads": raw_payload}]}
        random_pool = [{"text": text}]
        mooncake_trace = [
            {"timestamp": index, "text_input": text, "output_length": 16}
            for index in range(rows)
        ]
        bailian_trace = [
            {
                "chat_id": index + 1,
                "parent_chat_id": -1 if index == 0 else index,
                "timestamp": index,
                "input_length": tokens_per_row,
                "output_length": 16,
                "type": "text",
                "turn": index + 1,
            }
            for index in range(rows)
        ]
        burst_gpt_trace = [(index, tokens_per_row, 16) for index in range(rows)]
        sagemaker_texts = [text] * rows
        speed_bench_rows = [
            _speed_bench_row(question_id=f"speed-coding-{index + 1}")
            for index in range(rows)
        ]

    sagemaker_data_capture = []
    for index, capture_text in enumerate(sagemaker_texts):
        captured_input = orjson.dumps(
            {
                "messages": [{"role": "user", "content": capture_text}],
                "max_tokens": 16,
            }
        ).decode()
        captured_output = orjson.dumps({"usage": {"completion_tokens": 2}}).decode()
        sagemaker_data_capture.append(
            {
                "captureData": {
                    "endpointInput": {"data": captured_input, "encoding": "JSON"},
                    "endpointOutput": {"data": captured_output, "encoding": "JSON"},
                },
                "eventMetadata": {
                    "eventId": f"event-{index}",
                    "inferenceTime": "2026-07-20T00:00:00Z",
                },
            }
        )

    paths = {
        "single_turn": directory / "single_turn.jsonl",
        "multi_turn": directory / "multi_turn.jsonl",
        "raw_payload": directory / "raw_payload.jsonl",
        "inputs_json": directory / "inputs.json",
        "random_pool": directory / "random_pool.jsonl",
        "mooncake_trace": directory / "mooncake_trace.jsonl",
        "bailian_trace": directory / "bailian_trace.jsonl",
        "burst_gpt_trace": directory / "burst_gpt_trace.csv",
        "sagemaker_data_capture": directory / "sagemaker_data_capture.jsonl",
        "speed_bench": directory / "speed_bench.jsonl",
        "baseten_trace": directory / "baseten_trace.parquet",
    }
    _write_jsonl(paths["single_turn"], single_turn)
    _write_jsonl(paths["multi_turn"], multi_turn)
    _write_jsonl(paths["raw_payload"], raw_payload)
    paths["inputs_json"].write_bytes(orjson.dumps(inputs_json) + b"\n")
    _write_jsonl(paths["random_pool"], random_pool)
    _write_jsonl(paths["mooncake_trace"], mooncake_trace)
    _write_jsonl(paths["bailian_trace"], bailian_trace)
    paths["burst_gpt_trace"].write_text(
        "Timestamp,Request tokens,Response tokens\n"
        + "".join(
            f"{timestamp},{input_length},{output_length}\n"
            for timestamp, input_length, output_length in burst_gpt_trace
        ),
        encoding="utf-8",
    )
    _write_jsonl(paths["sagemaker_data_capture"], sagemaker_data_capture)
    _write_jsonl(paths["speed_bench"], speed_bench_rows)
    write_baseten_parquet(paths["baseten_trace"])
    return paths


def generate_format_cases(
    directory: Path,
    *,
    rows: int | None = None,
    tokens_per_row: int | None = None,
    include_public: bool = True,
) -> list[FormatCase]:
    """Build the full verified catalog of benchmark cases."""
    local_paths = generate_local_fixtures(
        directory, rows=rows, tokens_per_row=tokens_per_row
    )
    entry_count = rows if rows is not None else DEFAULT_BENCHMARK_ROWS
    cases: list[FormatCase] = []

    for name, profile in FORMAT_PROFILES.items():
        if profile.source_kind == "local_file":
            path = local_paths[name]
            source = SourceEnvelope(kind="local_file", path=str(path.resolve()))
            options = dict(profile.verified_options)
        elif profile.source_kind == "inline_synthetic":
            inline = shared_synthetic_inline(
                rankings=name == "synthetic_rankings",
                entries=entry_count,
                tokens_per_row=tokens_per_row,
            )
            source = SourceEnvelope(kind="inline_synthetic", inline=inline)
            options = dict(profile.verified_options)
        elif profile.source_kind == "public_cached":
            if not include_public:
                continue
            source = SourceEnvelope(
                kind="public_cached",
                public={"pin_key": profile.public_pin},
            )
            options = dict(profile.verified_options)
            if "max_conversations" in options:
                options["max_conversations"] = entry_count
        else:
            raise ValueError(f"unsupported source kind for {name}")
        cases.append(
            FormatCase(
                format=name,
                fixture_id=f"generated-{name}",
                options=options,
                source=source,
            )
        )
    return cases
