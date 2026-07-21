# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare equivalent Python and Rust dataset load/compose/tokenize paths."""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson

SCHEMA_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_MODEL = "test-model"
ADAPTER_TIMEOUT_S = 120
NON_EMPTY_OPTIONS_REASON = (
    "non-empty options are unsupported until cross-stack option mapping is verified"
)
SUPPORTED_FORMATS = (
    "single_turn",
    "multi_turn",
    "raw_payload",
    "inputs_json",
    "random_pool",
    "mooncake_trace",
    "bailian_trace",
    "burst_gpt_trace",
    "sagemaker_data_capture",
)
PARITY_FIELDS = (
    "row_count",
    "conversation_count",
    "turn_count",
    "total_input_tokens",
)
UNVERIFIED_FORMAT_REASON = "format is not in the verified Python/Rust intersection"
PUBLIC_HF_SKIP_REASON = (
    "public/Hugging Face datasets are skipped because equivalent generated local "
    "Python/Rust pipelines are not yet proven"
)
SYNTHETIC_SKIP_REASON = (
    "synthetic datasets are skipped because equivalent generated local Python/Rust "
    "pipelines are not yet proven"
)
ACCURACY_SKIP_REASON = (
    "accuracy datasets are skipped because equivalent generated local Python/Rust "
    "pipelines are not yet proven"
)
PUBLIC_HF_FORMATS = frozenset({"sharegpt", "hf_asr"})


@dataclass(frozen=True)
class Sample:
    """One adapter measurement and its semantic output counts."""

    implementation: str
    format: str
    fixture_id: str
    row_count: int
    conversation_count: int
    turn_count: int
    total_input_tokens: int | None
    elapsed_ns: int
    error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Sample:
        """Parse the shared adapter schema, rejecting missing or extra fields."""
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"invalid sample keys: missing={missing}, extra={extra}")
        try:
            raw_tokens = value["total_input_tokens"]
            total_input_tokens = (
                None if raw_tokens is None else int(raw_tokens)  # type: ignore[arg-type]
            )
            return cls(
                implementation=str(value["implementation"]),
                format=str(value["format"]),
                fixture_id=str(value["fixture_id"]),
                row_count=int(value["row_count"]),  # type: ignore[arg-type]
                conversation_count=int(value["conversation_count"]),  # type: ignore[arg-type]
                turn_count=int(value["turn_count"]),  # type: ignore[arg-type]
                total_input_tokens=total_input_tokens,
                elapsed_ns=int(value["elapsed_ns"]),  # type: ignore[arg-type]
                error=(None if value["error"] is None else str(value["error"])),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid sample field type: {error}") from error

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class FormatCase:
    """A format, input path, identity, and loader options to benchmark."""

    format: str
    path: Path
    fixture_id: str
    options: dict[str, object]


def _unsupported_format_reason(format_name: str) -> str:
    if format_name in PUBLIC_HF_FORMATS:
        return PUBLIC_HF_SKIP_REASON
    if format_name == "synthetic":
        return SYNTHETIC_SKIP_REASON
    if format_name == "accuracy":
        return ACCURACY_SKIP_REASON
    return UNVERIFIED_FORMAT_REASON


def _write_jsonl(path: Path, records: Sequence[object]) -> None:
    path.write_bytes(b"".join(orjson.dumps(record) + b"\n" for record in records))


def generate_fixtures(
    directory: Path,
    *,
    rows: int | None = None,
    tokens_per_row: int | None = None,
) -> list[FormatCase]:
    """Generate the deterministic, semantically verified fixture catalog."""
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
    else:
        if rows is None or rows <= 0:
            raise ValueError("rows must be positive")
        if tokens_per_row is None or tokens_per_row <= 0:
            raise ValueError("tokens_per_row must be positive")
        text = " ".join(["token"] * tokens_per_row)
        single_turn = [
            {"session_id": f"single-{index:06d}", "text": text}
            for index in range(rows)
        ]
        multi_turn = [
            {
                "session_id": f"multi-{index:06d}",
                "turns": [{"text": text}],
            }
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
        inputs_json = {
            "data": [
                {
                    "session_id": "session-001",
                    "payloads": raw_payload,
                }
            ]
        }
        # A one-row literal pool makes sampling deterministic across the Python
        # and Rust RNG implementations without claiming stream-level parity.
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
        burst_gpt_trace = [
            (index, tokens_per_row, 16) for index in range(rows)
        ]
        sagemaker_texts = [text] * rows

    sagemaker_data_capture = []
    for index, capture_text in enumerate(sagemaker_texts):
        captured_input = orjson.dumps(
            {
                "messages": [{"role": "user", "content": capture_text}],
                "max_tokens": 16,
            }
        ).decode()
        captured_output = orjson.dumps(
            {"usage": {"completion_tokens": 2}}
        ).decode()
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

    return [
        FormatCase(
            format=name,
            path=paths[name],
            fixture_id=f"generated-{name}",
            options={},
        )
        for name in SUPPORTED_FORMATS
    ]


def parse_manifest(
    manifest_path: Path,
) -> tuple[list[FormatCase], list[dict[str, str]]]:
    """Parse a schema-v1 manifest and explicitly skip unverified formats."""
    try:
        document = orjson.loads(manifest_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("manifest root must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be a list")

    cases: list[FormatCase] = []
    skips: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {index} must be an object")
        if set(entry) != {"format", "path", "options"}:
            raise ValueError(
                f"manifest entry {index} must contain format, path, and options"
            )
        format_name = entry["format"]
        input_path = entry["path"]
        options = entry["options"]
        if not isinstance(format_name, str) or not format_name:
            raise ValueError(f"manifest entry {index} has invalid format")
        if not isinstance(input_path, str) or not input_path:
            raise ValueError(f"manifest entry {index} has invalid path")
        if not isinstance(options, dict):
            raise ValueError(f"manifest entry {index} options must be an object")
        if format_name not in SUPPORTED_FORMATS:
            skips.append(
                {"format": format_name, "reason": _unsupported_format_reason(format_name)}
            )
            continue
        if options:
            skips.append({"format": format_name, "reason": NON_EMPTY_OPTIONS_REASON})
            continue
        path = Path(input_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        cases.append(
            FormatCase(
                format=format_name,
                path=path.resolve(),
                fixture_id=f"manifest-{index}-{format_name}",
                options=dict(options),
            )
        )
    return cases, skips


def summarize_samples(samples: Sequence[Sample]) -> dict[str, int | float]:
    """Summarize timings with median and nearest-rank p95 statistics."""
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    elapsed = [sample.elapsed_ns for sample in samples]
    if any(value <= 0 for value in elapsed):
        raise ValueError("successful samples must have positive elapsed_ns")
    sorted_elapsed = sorted(elapsed)
    p95_index = math.ceil(0.95 * len(sorted_elapsed)) - 1
    rows_per_second = [
        sample.row_count * 1_000_000_000 / sample.elapsed_ns for sample in samples
    ]
    tokens_per_second = [
        sample.total_input_tokens * 1_000_000_000 / sample.elapsed_ns
        for sample in samples
        if sample.total_input_tokens is not None
    ]
    return {
        "sample_count": len(samples),
        "median_elapsed_ns": statistics.median(elapsed),
        "p95_elapsed_ns": sorted_elapsed[p95_index],
        "median_rows_per_second": statistics.median(rows_per_second),
        "median_input_tokens_per_second": (
            None if not tokens_per_second else statistics.median(tokens_per_second)
        ),
    }


def validate_parity(left: Sample, right: Sample) -> None:
    """Require exact semantic-count parity between two samples."""
    if left.format != right.format:
        raise ValueError(f"format mismatch: {left.format!r} != {right.format!r}")
    if left.fixture_id != right.fixture_id:
        raise ValueError(
            f"fixture_id mismatch: {left.fixture_id!r} != {right.fixture_id!r}"
        )
    for field in PARITY_FIELDS:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value != right_value:
            raise ValueError(f"{field} mismatch: {left_value} != {right_value}")


def _adapter_arguments(case: FormatCase, *, seed: int, model: str) -> list[str]:
    return [
        "--format",
        case.format,
        "--path",
        str(case.path),
        "--options-json",
        orjson.dumps(case.options).decode(),
        "--fixture-id",
        case.fixture_id,
        "--seed",
        str(seed),
        "--model",
        model,
    ]


def _error_sample(implementation: str, case: FormatCase, message: str) -> Sample:
    return Sample(
        implementation=implementation,
        format=case.format,
        fixture_id=case.fixture_id,
        row_count=0,
        conversation_count=0,
        turn_count=0,
        total_input_tokens=None,
        elapsed_ns=0,
        error=message,
    )


def _run_adapter(
    implementation: str,
    command: Sequence[str],
    case: FormatCase,
    *,
    seed: int,
    model: str,
    runner: Callable[..., object],
) -> Sample:
    full_command = [*command, *_adapter_arguments(case, seed=seed, model=model)]
    try:
        completed = runner(
            full_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=ADAPTER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as error:
        return _error_sample(
            implementation,
            case,
            f"adapter timed out after {error.timeout}s",
        )
    except OSError as error:
        return _error_sample(implementation, case, str(error))

    stdout = str(getattr(completed, "stdout", ""))
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return _error_sample(
            implementation,
            case,
            f"adapter emitted {len(lines)} non-empty JSON lines",
        )
    try:
        decoded = orjson.loads(lines[0])
        if not isinstance(decoded, dict):
            raise ValueError("sample must be a JSON object")
        sample = Sample.from_dict(decoded)
    except (orjson.JSONDecodeError, ValueError) as error:
        return _error_sample(implementation, case, f"invalid adapter sample: {error}")
    if sample.implementation != implementation:
        return _error_sample(
            implementation,
            case,
            "adapter implementation field does not match invoked adapter",
        )
    if sample.format != case.format or sample.fixture_id != case.fixture_id:
        return _error_sample(
            implementation,
            case,
            "adapter sample does not identify the requested fixture",
        )
    return_code = int(getattr(completed, "returncode", 0))
    if return_code != 0 and sample.error is None:
        stderr = str(getattr(completed, "stderr", "")).strip()
        return _error_sample(
            implementation,
            case,
            stderr or f"adapter exited with status {return_code}",
        )
    return sample


def run_comparison(
    cases: Sequence[FormatCase],
    *,
    warmups: int,
    runs: int,
    adapter_commands: Mapping[str, Sequence[str]],
    runner: Callable[..., object] = subprocess.run,
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_MODEL,
) -> tuple[list[Sample], list[dict[str, object]]]:
    """Run adapters in alternating order and return measured samples only."""
    if warmups < 0:
        raise ValueError("warmups must be nonnegative")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if set(adapter_commands) != {"python", "rust"}:
        raise ValueError("adapter_commands must define python and rust")

    measured: list[Sample] = []
    failures: list[dict[str, object]] = []
    for case in cases:
        for iteration in range(warmups + runs):
            order = ("python", "rust") if iteration % 2 == 0 else ("rust", "python")
            pair: dict[str, Sample] = {}
            for implementation in order:
                pair[implementation] = _run_adapter(
                    implementation,
                    adapter_commands[implementation],
                    case,
                    seed=seed,
                    model=model,
                    runner=runner,
                )
            if iteration < warmups:
                continue
            current_samples = [pair["python"], pair["rust"]]
            measured.extend(current_samples)
            adapter_errors = [
                sample for sample in current_samples if sample.error is not None
            ]
            if adapter_errors:
                failures.append(
                    {
                        "format": case.format,
                        "iteration": iteration - warmups,
                        "reason": "adapter error",
                        "samples": [sample.to_dict() for sample in current_samples],
                    }
                )
                continue
            try:
                validate_parity(pair["python"], pair["rust"])
            except ValueError as error:
                failures.append(
                    {
                        "format": case.format,
                        "iteration": iteration - warmups,
                        "reason": str(error),
                        "samples": [sample.to_dict() for sample in current_samples],
                    }
                )
    return measured, failures


def build_report(
    *,
    samples: Sequence[Sample],
    skips: Sequence[Mapping[str, str]],
    failures: Sequence[Mapping[str, object]],
    options: Mapping[str, object],
    rust_binary_identity: str | None = None,
) -> dict[str, object]:
    """Build the versioned machine-readable benchmark report."""
    report_failures = [dict(failure) for failure in failures]
    failed_formats = {str(failure["format"]) for failure in report_failures}
    for sample in samples:
        if (
            sample.error is None
            and sample.elapsed_ns <= 0
            and sample.format not in failed_formats
        ):
            failed_formats.add(sample.format)
            report_failures.append(
                {
                    "format": sample.format,
                    "reason": "successful sample must have positive elapsed_ns",
                    "samples": [sample.to_dict()],
                }
            )
    successful_samples = [
        sample
        for sample in samples
        if sample.error is None and sample.format not in failed_formats
    ]
    formats: dict[str, object] = {}
    for format_name in sorted({sample.format for sample in successful_samples}):
        per_implementation = {
            implementation: [
                sample
                for sample in successful_samples
                if sample.format == format_name
                and sample.implementation == implementation
            ]
            for implementation in ("python", "rust")
        }
        if not all(per_implementation.values()):
            continue
        python_summary = summarize_samples(per_implementation["python"])
        rust_summary = summarize_samples(per_implementation["rust"])
        formats[format_name] = {
            "python": python_summary,
            "rust": rust_summary,
            "rust_speedup": (
                python_summary["median_elapsed_ns"] / rust_summary["median_elapsed_ns"]
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "options": dict(options),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "rust_binary_identity": rust_binary_identity,
        },
        "raw_samples": [sample.to_dict() for sample in samples],
        "formats": formats,
        "skips": [dict(skip) for skip in skips],
        "failures": report_failures,
    }


def _default_adapter_commands() -> dict[str, list[str]]:
    repository_root = Path(__file__).resolve().parents[2]
    rust_binary = (
        repository_root / "rust/target/release/examples/dataset_load_bench"
    )
    # Rebuild every invocation so a stale release example is never timed.
    subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            "aiperf-runtime",
            "--release",
            "--example",
            "dataset_load_bench",
        ],
        cwd=repository_root / "rust",
        check=True,
    )
    return {
        "python": [
            str(repository_root / ".venv/bin/python"),
            str(repository_root / "dev/benchmarks/dataset_load_python.py"),
        ],
        "rust": [str(rust_binary)],
    }


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip() for item in value.split(",") if item.strip()]
    if not formats:
        raise argparse.ArgumentTypeError("at least one format is required")
    return formats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--formats", type=_parse_formats)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tokens-per-row", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("dataset-load-comparison.json")
    )
    parser.add_argument("--keep-fixtures", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def _select_cases(
    cases: Sequence[FormatCase],
    requested: Sequence[str] | None,
    skips: list[dict[str, str]],
) -> list[FormatCase]:
    if requested is None:
        return list(cases)
    available = {case.format for case in cases}
    for format_name in requested:
        if format_name not in SUPPORTED_FORMATS:
            skips.append(
                {"format": format_name, "reason": _unsupported_format_reason(format_name)}
            )
        elif format_name not in available:
            skips.append(
                {
                    "format": format_name,
                    "reason": "requested format has no input in the manifest",
                }
            )
    requested_set = set(requested)
    return [case for case in cases if case.format in requested_set]


def _print_console_report(report: Mapping[str, object]) -> None:
    formats = report["formats"]
    assert isinstance(formats, dict)
    if formats:
        print(
            f"{'format':<16} {'python median':>14} {'rust median':>14} {'speedup':>10}"
        )
    for format_name, value in formats.items():
        assert isinstance(value, dict)
        python_summary = value["python"]
        rust_summary = value["rust"]
        assert isinstance(python_summary, dict)
        assert isinstance(rust_summary, dict)
        print(
            f"{format_name:<16} "
            f"{python_summary['median_elapsed_ns']:>14.0f} "
            f"{rust_summary['median_elapsed_ns']:>14.0f} "
            f"{value['rust_speedup']:>9.2f}x"
        )
    for skip in report["skips"]:  # type: ignore[union-attr]
        print(f"SKIP {skip['format']}: {skip['reason']}")
    for failure in report["failures"]:  # type: ignore[union-attr]
        print(
            f"FAIL {failure['format']}: {failure['reason']}",
            file=sys.stderr,
        )


def _execute(
    args: argparse.Namespace,
    *,
    fixture_directory: Path,
    adapter_commands: Mapping[str, Sequence[str]],
    runner: Callable[..., object],
) -> int:
    skips: list[dict[str, str]] = []
    if args.manifest is None:
        cases = generate_fixtures(
            fixture_directory,
            rows=args.rows,
            tokens_per_row=args.tokens_per_row,
        )
    else:
        cases, manifest_skips = parse_manifest(args.manifest)
        skips.extend(manifest_skips)
    cases = _select_cases(cases, args.formats, skips)

    samples, failures = run_comparison(
        cases,
        warmups=args.warmups,
        runs=args.runs,
        adapter_commands=adapter_commands,
        runner=runner,
        seed=args.seed,
        model=args.model,
    )
    report = build_report(
        samples=samples,
        skips=skips,
        failures=failures,
        options={
            "formats": args.formats,
            "rows": args.rows,
            "tokens_per_row": args.tokens_per_row,
            "warmups": args.warmups,
            "runs": args.runs,
            "manifest": (None if args.manifest is None else str(args.manifest)),
            "seed": args.seed,
            "model": args.model,
        },
        rust_binary_identity=" ".join(adapter_commands["rust"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2) + b"\n")
    _print_console_report(report)
    if not report["formats"]:
        print("No format completed successfully", file=sys.stderr)
        return 1
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_commands: Mapping[str, Sequence[str]] | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> int:
    """Run the comparison CLI and return its process exit status."""
    args = _parser().parse_args(argv)
    if args.rows <= 0:
        _parser().error("--rows must be positive")
    if args.tokens_per_row <= 0:
        _parser().error("--tokens-per-row must be positive")
    if args.warmups < 0:
        _parser().error("--warmups must be nonnegative")
    if args.runs <= 0:
        _parser().error("--runs must be positive")
    commands = (
        _default_adapter_commands() if adapter_commands is None else adapter_commands
    )

    if args.keep_fixtures:
        fixture_directory = Path(tempfile.mkdtemp(prefix="aiperf-dataset-load-"))
        print(f"Keeping fixtures in {fixture_directory}")
        return _execute(
            args,
            fixture_directory=fixture_directory,
            adapter_commands=commands,
            runner=runner,
        )
    with tempfile.TemporaryDirectory(prefix="aiperf-dataset-load-") as temporary:
        return _execute(
            args,
            fixture_directory=Path(temporary),
            adapter_commands=commands,
            runner=runner,
        )


if __name__ == "__main__":
    raise SystemExit(main())
