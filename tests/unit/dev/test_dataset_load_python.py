# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.common.models import Conversation, ModelEndpointInfo, Text, Turn
from dev.benchmarks import dataset_load_python
from dev.benchmarks.dataset_format_catalog import SourceEnvelope
from dev.benchmarks.dataset_load_compare import generate_fixtures


def _local_source(path: Path) -> SourceEnvelope:
    return SourceEnvelope(kind="local_file", path=str(path))


def _source_json(path: Path) -> str:
    import json

    return json.dumps(_local_source(path).to_dict(), separators=(",", ":"))


RAW_PAYLOADS = [
    {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "max_tokens": 16,
    },
    {
        "messages": [{"role": "user", "content": "bye"}],
        "model": "test-model",
        "max_tokens": 16,
    },
]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def format_cases(tmp_path: Path) -> dict[str, tuple[Path, tuple[int, int, int]]]:
    single_turn_path = tmp_path / "single_turn.jsonl"
    _write_jsonl(
        single_turn_path,
        [
            {"text": "alpha beta gamma"},
            {"session_id": "s-a", "text": "turn one"},
            {"session_id": "s-a", "text": "turn two"},
        ],
    )

    multi_turn_path = tmp_path / "multi_turn.jsonl"
    _write_jsonl(
        multi_turn_path,
        [
            {"session_id": "m1", "turns": [{"text": "q1"}, {"text": "q2"}]},
            {"session_id": "m2", "turns": [{"text": "only"}]},
        ],
    )

    raw_payload_path = tmp_path / "raw_payload.jsonl"
    _write_jsonl(raw_payload_path, RAW_PAYLOADS)

    inputs_json_path = tmp_path / "inputs.json"
    inputs_json_path.write_text(
        json.dumps(
            {
                "data": [
                    {"session_id": "session-001", "payloads": RAW_PAYLOADS},
                    {"session_id": "session-002", "payloads": [RAW_PAYLOADS[0]]},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "single_turn": (single_turn_path, (3, 2, 3)),
        "multi_turn": (multi_turn_path, (2, 2, 3)),
        "raw_payload": (raw_payload_path, (2, 2, 2)),
        "inputs_json": (inputs_json_path, (3, 2, 3)),
    }


@pytest.mark.parametrize(
    "format_name",
    ["single_turn", "multi_turn", "raw_payload", "inputs_json"],
)
def test_run_sample_reports_deterministic_counts(
    format_name: str,
    format_cases: dict[str, tuple[Path, tuple[int, int, int]]],
) -> None:
    path, expected_counts = format_cases[format_name]

    first = dataset_load_python.run_sample(
        format_name=format_name,
        path=path,
        source=_local_source(path),
        options={},
        fixture_id=f"{format_name}-fixture",
        seed=42,
        model="test-model",
    )
    second = dataset_load_python.run_sample(
        format_name=format_name,
        path=path,
        source=_local_source(path),
        options={},
        fixture_id=f"{format_name}-fixture",
        seed=42,
        model="test-model",
    )

    assert first.implementation == "python"
    assert first.format == format_name
    assert first.fixture_id == f"{format_name}-fixture"
    assert (
        first.row_count,
        first.conversation_count,
        first.turn_count,
    ) == expected_counts
    if format_name in {"raw_payload", "inputs_json"}:
        assert first.total_input_tokens is None
    else:
        assert first.total_input_tokens is not None
        assert first.total_input_tokens > 0
    assert first.elapsed_ns > 0
    assert first.error is None
    assert (
        second.row_count,
        second.conversation_count,
        second.turn_count,
        second.total_input_tokens,
    ) == (
        first.row_count,
        first.conversation_count,
        first.turn_count,
        first.total_input_tokens,
    )


@pytest.mark.parametrize(
    ("format_name", "expected_counts"),
    [
        ("random_pool", (1, 1, 1)),
        ("mooncake_trace", (3, 3, 3)),
        ("bailian_trace", (2, 1, 2)),
        ("burst_gpt_trace", (2, 2, 2)),
        ("sagemaker_data_capture", (1, 1, 1)),
    ],
)
def test_run_sample_reports_new_format_counts_and_token_semantics(
    tmp_path: Path,
    format_name: str,
    expected_counts: tuple[int, int, int],
) -> None:
    cases = {case.format: case for case in generate_fixtures(tmp_path / "fixtures")}
    case = cases[format_name]

    sample = dataset_load_python.run_sample(
        format_name=format_name,
        path=case.path,
        source=case.source,
        options=case.options,
        fixture_id=case.fixture_id,
        seed=42,
        model="test-model",
    )

    assert sample.error is None
    assert (sample.row_count, sample.conversation_count, sample.turn_count) == (
        expected_counts
    )
    assert sample.total_input_tokens > 0
    assert sample.elapsed_ns > 0


def test_trace_loader_construction_matches_custom_composer_contract(
    tmp_path: Path,
) -> None:
    cases = {case.format: case for case in generate_fixtures(tmp_path)}
    tokenizer = dataset_load_python.Tokenizer.from_pretrained("builtin")
    run = dataset_load_python._benchmark_run(seed=42, model="test-model")

    mooncake = dataset_load_python._create_file_loader(
        "mooncake_trace", cases["mooncake_trace"].path, run, tokenizer, {}
    )
    bailian = dataset_load_python._create_file_loader(
        "bailian_trace", cases["bailian_trace"].path, run, tokenizer, {}
    )

    assert mooncake.prompt_generator.tokenizer is tokenizer
    assert bailian.prompt_generator.tokenizer is tokenizer
    assert bailian._block_size == 16


def test_count_extracted_tokens_prefers_chat_template_over_flat_text() -> None:
    class FakeInnerTokenizer:
        chat_template = "{{ messages }}"

        @staticmethod
        def apply_chat_template(
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            assert tokenize is True
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "hello"}]
            return [11, 12, 13, 14]

    class FakeTokenizer:
        def __init__(self) -> None:
            self._tokenizer = FakeInnerTokenizer()

        @staticmethod
        def encode(text: str) -> list[int]:
            return text.split()

    extracted = SimpleNamespace(
        pretokenised_token_count=2,
        texts=["flat text should not be used"],
        tool_texts=["tool schema"],
        messages=[{"role": "user", "content": "hello"}],
    )

    total = dataset_load_python._count_extracted_tokens(
        extracted,
        FakeTokenizer(),
        apply_chat_template=True,
    )

    assert total == 2 + 4 + 2


def test_chat_template_token_count_accepts_input_ids_mapping() -> None:
    class FakeInnerTokenizer:
        chat_template = "{{ messages }}"

        @staticmethod
        def apply_chat_template(
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> dict[str, list[int]]:
            assert tokenize is True
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "hello"}]
            return {"input_ids": [7, 8, 9]}

    class FakeTokenizer:
        def __init__(self) -> None:
            self._tokenizer = FakeInnerTokenizer()

    assert (
        dataset_load_python._chat_template_token_count(
            FakeTokenizer(),
            [{"role": "user", "content": "hello"}],
        )
        == 3
    )


def test_count_input_tokens_uses_chat_template_for_structured_turns() -> None:
    run = dataset_load_python._benchmark_run(seed=42, model="test-model")

    class FakeInnerTokenizer:
        chat_template = "{{ messages }}"

        @staticmethod
        def apply_chat_template(
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            assert tokenize is True
            assert add_generation_prompt is True
            assert messages == [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "ctx"},
                {"role": "user", "content": "hello"},
            ]
            return [1, 2, 3, 4, 5]

    class FakeTokenizer:
        def __init__(self) -> None:
            self._tokenizer = FakeInnerTokenizer()

        @staticmethod
        def encode(text: str) -> list[int]:
            return text.split()

    chat_endpoint = dataset_load_python.ChatEndpoint(ModelEndpointInfo.from_run(run))
    conversations = [
        Conversation(
            session_id="s1",
            system_message="sys",
            user_context_message="ctx",
            turns=[Turn(role="user", texts=[Text(contents=["hello"])])],
        )
    ]

    total = dataset_load_python._count_input_tokens(
        conversations,
        FakeTokenizer(),
        apply_chat_template=True,
        chat_endpoint=chat_endpoint,
    )

    assert total == 5


def test_main_initializes_rng_for_fresh_trace_adapter_process(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from aiperf.common import random_generator as rng

    cases = {case.format: case for case in generate_fixtures(tmp_path)}
    rng.reset()

    exit_code = dataset_load_python.main(
        [
            "--format",
            "mooncake_trace",
            "--path",
            str(cases["mooncake_trace"].path),
            "--options-json",
            "{}",
            "--source-json",
            _source_json(cases["mooncake_trace"].path),
            "--fixture-id",
            "fresh-process",
            "--seed",
            "42",
            "--model",
            "test-model",
        ]
    )

    record = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert record["error"] is None


def test_main_emits_exact_one_line_sample_schema_for_unknown_format(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    exit_code = dataset_load_python.main(
        [
            "--format",
            "unknown",
            "--path",
            str(tmp_path / "unused.jsonl"),
            "--options-json",
            "{}",
            "--source-json",
            _source_json(tmp_path / "unused.jsonl"),
            "--fixture-id",
            "unknown-fixture",
            "--seed",
            "42",
            "--model",
            "test-model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    record = json.loads(captured.out)
    assert set(record) == {
        "implementation",
        "format",
        "fixture_id",
        "row_count",
        "conversation_count",
        "turn_count",
        "total_input_tokens",
        "elapsed_ns",
        "error",
    }
    assert record == {
        "implementation": "python",
        "format": "unknown",
        "fixture_id": "unknown-fixture",
        "row_count": 0,
        "conversation_count": 0,
        "turn_count": 0,
        "total_input_tokens": None,
        "elapsed_ns": 0,
        "error": "unsupported dataset format: unknown",
    }


def test_tokenizer_initializes_before_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "single.jsonl"
    _write_jsonl(path, [{"text": "hello"}])
    events: list[str] = []
    real_from_pretrained = dataset_load_python.Tokenizer.from_pretrained
    real_clock = dataset_load_python.time.perf_counter_ns

    def tracked_from_pretrained(name: str):
        events.append(f"tokenizer:{name}")
        tokenizer = real_from_pretrained(name)
        real_encode = tokenizer.encode

        def tracked_encode(text: str, **kwargs: object):
            events.append(f"encode:{text}")
            return real_encode(text, **kwargs)

        tokenizer.encode = tracked_encode  # type: ignore[method-assign]
        return tokenizer

    def tracked_clock() -> int:
        events.append("clock")
        return real_clock()

    monkeypatch.setattr(
        dataset_load_python.Tokenizer,
        "from_pretrained",
        tracked_from_pretrained,
    )
    monkeypatch.setattr(dataset_load_python.time, "perf_counter_ns", tracked_clock)

    sample = dataset_load_python.run_sample(
        format_name="single_turn",
        path=path,
        source=_local_source(path),
        options={},
        fixture_id="timing",
        seed=42,
        model="test-model",
    )

    assert sample.error is None
    assert events[:3] == ["tokenizer:builtin", "encode:warm", "clock"]
    assert events[-1] == "clock"
    assert "clock" in events[2:]


def test_run_sample_uses_requested_tokenizer_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "single.jsonl"
    _write_jsonl(path, [{"text": "hello"}])
    names: list[str] = []
    real_from_pretrained = dataset_load_python.Tokenizer.from_pretrained

    def tracked_from_pretrained(name: str):
        names.append(name)
        return real_from_pretrained("builtin")

    monkeypatch.setattr(
        dataset_load_python.Tokenizer,
        "from_pretrained",
        tracked_from_pretrained,
    )

    sample = dataset_load_python.run_sample(
        format_name="single_turn",
        path=path,
        source=_local_source(path),
        options={},
        fixture_id="tokenizer-name",
        seed=42,
        model="test-model",
        tokenizer_name="openai-community/gpt2",
    )

    assert sample.error is None
    assert names == ["openai-community/gpt2"]


def test_main_emits_success_sample_schema(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "single.jsonl"
    _write_jsonl(path, [{"text": "alpha beta gamma"}])

    exit_code = dataset_load_python.main(
        [
            "--format",
            "single_turn",
            "--path",
            str(path),
            "--options-json",
            "{}",
            "--source-json",
            _source_json(path),
            "--fixture-id",
            "success-fixture",
            "--seed",
            "42",
            "--model",
            "test-model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    record = json.loads(captured.out)
    assert set(record) == {
        "implementation",
        "format",
        "fixture_id",
        "row_count",
        "conversation_count",
        "turn_count",
        "total_input_tokens",
        "elapsed_ns",
        "error",
    }
    assert record["implementation"] == "python"
    assert record["format"] == "single_turn"
    assert record["fixture_id"] == "success-fixture"
    assert record["row_count"] == 1
    assert record["conversation_count"] == 1
    assert record["turn_count"] == 1
    assert record["total_input_tokens"] > 0
    assert record["elapsed_ns"] > 0
    assert record["error"] is None


def test_main_rejects_nonempty_options_with_structured_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "single.jsonl"
    _write_jsonl(path, [{"text": "hello"}])

    exit_code = dataset_load_python.main(
        [
            "--format",
            "single_turn",
            "--path",
            str(path),
            "--options-json",
            '{"text_field":"text"}',
            "--source-json",
            _source_json(path),
            "--fixture-id",
            "options-fixture",
            "--seed",
            "42",
            "--model",
            "test-model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    record = json.loads(captured.out)
    assert "verified mapping" in record["error"]
    assert record["elapsed_ns"] == 0
    assert record["row_count"] == 0
