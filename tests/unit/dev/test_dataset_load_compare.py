# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python/Rust dataset-load comparison orchestrator."""

from __future__ import annotations

import subprocess
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest

from dev.benchmarks.dataset_load_compare import (
    ADAPTER_TIMEOUT_S,
    NON_EMPTY_OPTIONS_REASON,
    FormatCase,
    Sample,
    _run_adapter,
    build_report,
    generate_fixtures,
    main,
    parse_manifest,
    run_comparison,
    summarize_samples,
    validate_parity,
)


def _sample(
    implementation: str,
    *,
    fixture_id: str = "generated-single_turn",
    elapsed_ns: int = 100,
    row_count: int = 2,
    conversation_count: int = 2,
    turn_count: int = 3,
    total_input_tokens: int = 12,
    error: str | None = None,
) -> Sample:
    return Sample(
        implementation=implementation,
        format="single_turn",
        fixture_id=fixture_id,
        row_count=row_count,
        conversation_count=conversation_count,
        turn_count=turn_count,
        total_input_tokens=total_input_tokens,
        elapsed_ns=elapsed_ns,
        error=error,
    )


def test_sample_and_format_case_are_immutable(tmp_path: Path) -> None:
    sample = _sample("python")
    case = FormatCase("single_turn", tmp_path / "input.jsonl", "fixture", {})

    with pytest.raises(FrozenInstanceError):
        sample.elapsed_ns = 10  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        case.format = "other"  # type: ignore[misc]


def test_generate_fixtures_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_fixtures(tmp_path / "first")
    second = generate_fixtures(tmp_path / "second")

    assert [case.format for case in first] == [
        "single_turn",
        "multi_turn",
        "raw_payload",
        "inputs_json",
        "random_pool",
        "mooncake_trace",
        "bailian_trace",
        "burst_gpt_trace",
        "sagemaker_data_capture",
    ]
    assert [case.path.read_bytes() for case in first] == [
        case.path.read_bytes() for case in second
    ]
    assert [case.fixture_id for case in first] == [
        "generated-single_turn",
        "generated-multi_turn",
        "generated-raw_payload",
        "generated-inputs_json",
        "generated-random_pool",
        "generated-mooncake_trace",
        "generated-bailian_trace",
        "generated-burst_gpt_trace",
        "generated-sagemaker_data_capture",
    ]


def test_generate_fixtures_uses_approved_local_shapes(tmp_path: Path) -> None:
    cases = {case.format: case for case in generate_fixtures(tmp_path)}

    single_turn = [
        orjson.loads(line)
        for line in cases["single_turn"].path.read_bytes().splitlines()
    ]
    multi_turn = [
        orjson.loads(line)
        for line in cases["multi_turn"].path.read_bytes().splitlines()
    ]
    raw_payload = [
        orjson.loads(line)
        for line in cases["raw_payload"].path.read_bytes().splitlines()
    ]
    inputs_json = orjson.loads(cases["inputs_json"].path.read_bytes())
    random_pool = [
        orjson.loads(line)
        for line in cases["random_pool"].path.read_bytes().splitlines()
    ]
    mooncake = [
        orjson.loads(line)
        for line in cases["mooncake_trace"].path.read_bytes().splitlines()
    ]
    bailian = [
        orjson.loads(line)
        for line in cases["bailian_trace"].path.read_bytes().splitlines()
    ]
    burst_gpt = cases["burst_gpt_trace"].path.read_text(encoding="utf-8")
    sagemaker = [
        orjson.loads(line)
        for line in cases["sagemaker_data_capture"].path.read_bytes().splitlines()
    ]

    assert single_turn == [
        {"text": "alpha beta gamma"},
        {"session_id": "s-a", "text": "turn one"},
        {"session_id": "s-a", "text": "turn two"},
    ]
    assert multi_turn == [
        {"session_id": "m1", "turns": [{"text": "q1"}, {"text": "q2"}]},
        {"session_id": "m2", "turns": [{"text": "only"}]},
    ]
    assert raw_payload == [
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
    assert inputs_json == {
        "data": [
            {"session_id": "session-001", "payloads": raw_payload},
            {"session_id": "session-002", "payloads": [raw_payload[0]]},
        ]
    }
    assert random_pool == [{"text": "alpha beta gamma"}]
    assert mooncake == [
        {"timestamp": 0, "text_input": "alpha beta gamma", "output_length": 16},
        {"timestamp": 1, "text_input": "turn one", "output_length": 16},
        {"timestamp": 2, "text_input": "turn two", "output_length": 16},
    ]
    assert all("hash_ids" not in row for row in mooncake)
    assert bailian == [
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
    assert all("hash_ids" not in row for row in bailian)
    assert burst_gpt == (
        "Timestamp,Request tokens,Response tokens\n"
        "0,3,16\n"
        "1,2,16\n"
    )
    captured_input = orjson.loads(
        sagemaker[0]["captureData"]["endpointInput"]["data"]
    )
    captured_output = orjson.loads(
        sagemaker[0]["captureData"]["endpointOutput"]["data"]
    )
    assert sagemaker[0]["captureData"]["endpointInput"]["encoding"] == "JSON"
    assert captured_input["messages"] == [
        {"role": "user", "content": "alpha beta gamma"}
    ]
    assert "prompt_tokens" not in captured_output.get("usage", {})


def test_generate_fixtures_applies_row_and_token_controls(tmp_path: Path) -> None:
    cases = {
        case.format: case
        for case in generate_fixtures(tmp_path, rows=4, tokens_per_row=6)
    }

    assert len(cases["single_turn"].path.read_bytes().splitlines()) == 4
    assert len(cases["multi_turn"].path.read_bytes().splitlines()) == 4
    assert len(cases["raw_payload"].path.read_bytes().splitlines()) == 4
    assert len(cases["random_pool"].path.read_bytes().splitlines()) == 1
    assert len(cases["mooncake_trace"].path.read_bytes().splitlines()) == 4
    assert len(cases["bailian_trace"].path.read_bytes().splitlines()) == 4
    assert len(cases["burst_gpt_trace"].path.read_text().splitlines()) == 5
    assert len(cases["sagemaker_data_capture"].path.read_bytes().splitlines()) == 4
    inputs_json = orjson.loads(cases["inputs_json"].path.read_bytes())
    assert sum(len(session["payloads"]) for session in inputs_json["data"]) == 4

    single_turn = [
        orjson.loads(line)
        for line in cases["single_turn"].path.read_bytes().splitlines()
    ]
    assert all(len(row["text"].split()) == 6 for row in single_turn)
    multi_turn = [
        orjson.loads(line)
        for line in cases["multi_turn"].path.read_bytes().splitlines()
    ]
    assert all(len(row["turns"][0]["text"].split()) == 6 for row in multi_turn)
    raw_payload = [
        orjson.loads(line)
        for line in cases["raw_payload"].path.read_bytes().splitlines()
    ]
    assert all(
        len(row["messages"][0]["content"].split()) == 6 for row in raw_payload
    )
    assert all(
        len(payload["messages"][0]["content"].split()) == 6
        for session in inputs_json["data"]
        for payload in session["payloads"]
    )


def test_sagemaker_generated_timestamps_remain_valid_beyond_one_minute(
    tmp_path: Path,
) -> None:
    cases = {
        case.format: case
        for case in generate_fixtures(tmp_path, rows=61, tokens_per_row=1)
    }
    captures = [
        orjson.loads(line)
        for line in cases["sagemaker_data_capture"].path.read_bytes().splitlines()
    ]

    for capture in captures:
        datetime.fromisoformat(capture["eventMetadata"]["inferenceTime"])


@pytest.mark.integration
@pytest.mark.slow
def test_cross_language_generated_catalog_has_count_parity_and_positive_timing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--rows",
            "4",
            "--tokens-per-row",
            "6",
            "--warmups",
            "0",
            "--runs",
            "1",
            "--output",
            str(output),
        ]
    )

    report = orjson.loads(output.read_bytes())
    samples = [Sample.from_dict(sample) for sample in report["raw_samples"]]
    assert exit_code == 0
    assert report["failures"] == []
    assert set(report["formats"]) == {
        "single_turn",
        "multi_turn",
        "raw_payload",
        "inputs_json",
        "random_pool",
        "mooncake_trace",
        "bailian_trace",
        "burst_gpt_trace",
        "sagemaker_data_capture",
    }
    assert len(samples) == 18
    for offset in range(0, len(samples), 2):
        validate_parity(samples[offset], samples[offset + 1])
    assert all(sample.elapsed_ns > 0 for sample in samples)


def test_summarize_samples_uses_median_and_nearest_rank_p95() -> None:
    samples = [
        _sample("python", elapsed_ns=10),
        _sample("python", elapsed_ns=20),
        _sample("python", elapsed_ns=30),
        _sample("python", elapsed_ns=40),
        _sample("python", elapsed_ns=1_000),
    ]

    summary = summarize_samples(samples)

    assert summary["sample_count"] == 5
    assert summary["median_elapsed_ns"] == 30
    assert summary["p95_elapsed_ns"] == 1_000
    assert summary["median_rows_per_second"] == pytest.approx(2e9 / 30)
    assert summary["median_input_tokens_per_second"] == pytest.approx(12e9 / 30)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_count", 3),
        ("conversation_count", 3),
        ("turn_count", 4),
        ("total_input_tokens", 13),
    ],
)
def test_validate_parity_rejects_each_count_mismatch(field: str, value: int) -> None:
    python = _sample("python")
    rust_values = {
        "row_count": python.row_count,
        "conversation_count": python.conversation_count,
        "turn_count": python.turn_count,
        "total_input_tokens": python.total_input_tokens,
    }
    rust_values[field] = value
    rust = _sample("rust", **rust_values)

    with pytest.raises(ValueError, match=field):
        validate_parity(python, rust)


def test_parse_manifest_returns_supported_cases_and_explicit_skips(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        orjson.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "format": "single_turn",
                        "path": "fixture.jsonl",
                        "options": {},
                    },
                    {
                        "format": "multi_turn",
                        "path": "fixture.jsonl",
                        "options": {"text_field": "text"},
                    },
                    {"format": "sharegpt", "path": "fixture.jsonl", "options": {}},
                ],
            }
        )
    )

    cases, skips = parse_manifest(manifest)

    assert cases == [
        FormatCase(
            format="single_turn",
            path=fixture,
            fixture_id="manifest-0-single_turn",
            options={},
        )
    ]
    assert skips == [
        {
            "format": "multi_turn",
            "reason": NON_EMPTY_OPTIONS_REASON,
        },
        {
            "format": "sharegpt",
            "reason": (
                "public/Hugging Face datasets are skipped because equivalent "
                "generated local Python/Rust pipelines are not yet proven"
            ),
        }
    ]


def test_parse_manifest_gives_precise_generated_pipeline_skip_reasons(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        orjson.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {"format": name, "path": "fixture.jsonl", "options": {}}
                    for name in ("sharegpt", "hf_asr", "synthetic", "accuracy")
                ],
            }
        )
    )

    cases, skips = parse_manifest(manifest)

    assert cases == []
    assert skips == [
        {
            "format": "sharegpt",
            "reason": (
                "public/Hugging Face datasets are skipped because equivalent "
                "generated local Python/Rust pipelines are not yet proven"
            ),
        },
        {
            "format": "hf_asr",
            "reason": (
                "public/Hugging Face datasets are skipped because equivalent "
                "generated local Python/Rust pipelines are not yet proven"
            ),
        },
        {
            "format": "synthetic",
            "reason": (
                "synthetic datasets are skipped because equivalent generated "
                "local Python/Rust pipelines are not yet proven"
            ),
        },
        {
            "format": "accuracy",
            "reason": (
                "accuracy datasets are skipped because equivalent generated "
                "local Python/Rust pipelines are not yet proven"
            ),
        },
    ]


def test_build_report_serializes_skips_and_speedup() -> None:
    python = [_sample("python", elapsed_ns=200)]
    rust = [_sample("rust", elapsed_ns=100)]
    report = build_report(
        samples=python + rust,
        skips=[{"format": "sharegpt", "reason": "not equivalent"}],
        failures=[],
        options={"runs": 1},
    )

    encoded = orjson.dumps(report)
    decoded = orjson.loads(encoded)

    assert decoded["schema_version"] == 1
    assert decoded["skips"] == [{"format": "sharegpt", "reason": "not equivalent"}]
    assert decoded["formats"]["single_turn"]["rust_speedup"] == 2.0
    assert len(decoded["raw_samples"]) == 2
    assert decoded["environment"]["python_version"]
    assert decoded["environment"]["platform"]


def test_run_comparison_alternates_adapter_order_and_removes_warmups(
    tmp_path: Path,
) -> None:
    case = FormatCase(
        "single_turn", tmp_path / "fixture.jsonl", "fixture", {"key": "value"}
    )
    calls: list[str] = []

    def runner(command: list[str], **_: object) -> SimpleNamespace:
        implementation = command[0]
        calls.append(implementation)
        record = _sample(implementation, fixture_id="fixture", elapsed_ns=100)
        return SimpleNamespace(
            returncode=0,
            stdout=orjson.dumps(record.to_dict()).decode(),
            stderr="",
        )

    samples, failures = run_comparison(
        [case],
        warmups=1,
        runs=3,
        adapter_commands={"python": ["python"], "rust": ["rust"]},
        runner=runner,
        seed=42,
        model="test-model",
    )

    assert calls == ["python", "rust", "rust", "python"] * 2
    assert len(samples) == 6
    assert failures == []


def test_run_comparison_passes_the_complete_adapter_contract(tmp_path: Path) -> None:
    case = FormatCase(
        "single_turn", tmp_path / "fixture.jsonl", "fixture", {"key": "value"}
    )
    commands: list[list[str]] = []

    def runner(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        implementation = command[0]
        return SimpleNamespace(
            returncode=0,
            stdout=orjson.dumps(
                _sample(implementation, fixture_id="fixture").to_dict()
            ).decode(),
            stderr="",
        )

    run_comparison(
        [case],
        warmups=0,
        runs=1,
        adapter_commands={"python": ["python"], "rust": ["rust"]},
        runner=runner,
        seed=42,
        model="test-model",
    )

    command = commands[0]
    assert command[1:] == [
        "--format",
        "single_turn",
        "--path",
        str(case.path),
        "--options-json",
        '{"key":"value"}',
        "--fixture-id",
        "fixture",
        "--seed",
        "42",
        "--model",
        "test-model",
    ]


def test_main_returns_nonzero_when_no_format_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def runner(command: list[str], **_: object) -> SimpleNamespace:
        implementation = command[0]
        failed = _sample(implementation, elapsed_ns=0, error="adapter failed")
        return SimpleNamespace(
            returncode=1,
            stdout=orjson.dumps(failed.to_dict()).decode(),
            stderr="",
        )

    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--formats",
            "single_turn",
            "--warmups",
            "0",
            "--runs",
            "1",
            "--output",
            str(output),
        ],
        adapter_commands={"python": ["python"], "rust": ["rust"]},
        runner=runner,
    )

    assert exit_code != 0
    assert orjson.loads(output.read_bytes())["formats"] == {}
    assert "No format completed successfully" in capsys.readouterr().err


def test_build_report_excludes_format_on_count_mismatch_failure() -> None:
    matching_python = _sample("python", elapsed_ns=200)
    matching_rust = _sample("rust", elapsed_ns=100)
    mismatched_python = _sample("python", elapsed_ns=200, row_count=9)
    mismatched_rust = _sample("rust", elapsed_ns=100, row_count=2)
    failures = [
        {
            "format": "single_turn",
            "iteration": 0,
            "reason": "row_count mismatch: 9 != 2",
            "samples": [mismatched_python.to_dict(), mismatched_rust.to_dict()],
        }
    ]

    report = build_report(
        samples=[
            mismatched_python,
            mismatched_rust,
            matching_python,
            matching_rust,
        ],
        skips=[],
        failures=failures,
        options={"runs": 1},
    )

    assert "single_turn" not in report["formats"]
    assert report["failures"] == failures


def test_build_report_excludes_format_on_nonpositive_elapsed() -> None:
    python = _sample("python", elapsed_ns=0)
    rust = _sample("rust", elapsed_ns=100)

    report = build_report(
        samples=[python, rust],
        skips=[],
        failures=[],
        options={"runs": 1},
    )

    assert "single_turn" not in report["formats"]
    assert len(report["failures"]) == 1
    assert report["failures"][0]["format"] == "single_turn"
    assert "elapsed_ns" in str(report["failures"][0]["reason"])


def _adapter_case(tmp_path: Path) -> FormatCase:
    return FormatCase("single_turn", tmp_path / "fixture.jsonl", "fixture", {})


def test_run_adapter_rejects_bad_json(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="not-json\n", stderr="")

    sample = _run_adapter(
        "python",
        ["python"],
        _adapter_case(tmp_path),
        seed=42,
        model="test-model",
        runner=runner,
    )

    assert sample.error is not None
    assert "invalid adapter sample" in sample.error


def test_run_adapter_rejects_wrong_line_count(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> SimpleNamespace:
        line = orjson.dumps(_sample("python", fixture_id="fixture").to_dict()).decode()
        return SimpleNamespace(
            returncode=0,
            stdout=f"{line}\n{line}\n",
            stderr="",
        )

    sample = _run_adapter(
        "python",
        ["python"],
        _adapter_case(tmp_path),
        seed=42,
        model="test-model",
        runner=runner,
    )

    assert sample.error is not None
    assert "2 non-empty JSON lines" in sample.error


def test_run_adapter_rejects_identity_mismatch(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=orjson.dumps(
                _sample("rust", fixture_id="fixture").to_dict()
            ).decode(),
            stderr="",
        )

    sample = _run_adapter(
        "python",
        ["python"],
        _adapter_case(tmp_path),
        seed=42,
        model="test-model",
        runner=runner,
    )

    assert sample.error is not None
    assert "implementation field does not match" in sample.error


def test_run_adapter_rejects_nonzero_without_error_field(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=7,
            stdout=orjson.dumps(
                _sample("python", fixture_id="fixture").to_dict()
            ).decode(),
            stderr="boom",
        )

    sample = _run_adapter(
        "python",
        ["python"],
        _adapter_case(tmp_path),
        seed=42,
        model="test-model",
        runner=runner,
    )

    assert sample.error == "boom"


def test_run_adapter_handles_timeout_expired(tmp_path: Path) -> None:
    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert kwargs.get("timeout") == ADAPTER_TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd=command, timeout=ADAPTER_TIMEOUT_S)

    sample = _run_adapter(
        "python",
        ["python"],
        _adapter_case(tmp_path),
        seed=42,
        model="test-model",
        runner=runner,
    )

    assert sample.error is not None
    assert "timed out" in sample.error.lower()


def test_default_adapter_commands_always_builds_rust_example(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import dev.benchmarks.dataset_load_compare as compare

    repository_root = tmp_path / "repo"
    (repository_root / "dev/benchmarks").mkdir(parents=True)
    (repository_root / ".venv/bin").mkdir(parents=True)
    (repository_root / ".venv/bin/python").write_text("python\n", encoding="utf-8")
    binary = repository_root / "rust/target/release/examples/dataset_load_bench"
    binary.parent.mkdir(parents=True)
    binary.write_text("stale", encoding="utf-8")
    compare_path = repository_root / "dev/benchmarks/dataset_load_compare.py"
    compare_path.write_text("# stub\n", encoding="utf-8")
    builds: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], **kwargs: object) -> MagicMock:
        builds.append((list(command), Path(str(kwargs["cwd"]))))
        completed = MagicMock()
        completed.returncode = 0
        return completed

    monkeypatch.setattr(compare, "__file__", str(compare_path))
    monkeypatch.setattr(compare.subprocess, "run", fake_run)

    commands = compare._default_adapter_commands()

    assert builds == [
        (
            [
                "cargo",
                "build",
                "-p",
                "aiperf-runtime",
                "--release",
                "--example",
                "dataset_load_bench",
            ],
            repository_root / "rust",
        )
    ]
    assert commands["rust"] == [str(binary)]
    assert commands["python"] == [
        str(repository_root / ".venv/bin/python"),
        str(repository_root / "dev/benchmarks/dataset_load_python.py"),
    ]
