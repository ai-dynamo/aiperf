# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare equivalent Python and Rust dataset load/compose/tokenize paths.

See ``dev/benchmarks/README.md`` for tokenizer, chat-template, exact-ISL, and
synthetic-parity usage notes.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson

from dev.benchmarks.dataset_fixtures import generate_format_cases
from dev.benchmarks.dataset_format_catalog import (
    EXCLUDED_FORMATS,
    SUPPORTED_FORMATS,
    FormatCase,
    SourceEnvelope,
    documented_skip_for,
    parity_fields_for,
    profile_for,
    unsupported_format_reason,
)
from dev.benchmarks.dataset_public_cache import (
    OFFLINE_ENV,
    prefetch_public_cases,
    source_json_for_adapters,
)

SCHEMA_VERSION = 2
DEFAULT_SEED = 42
DEFAULT_MODEL = "test-model"
ADAPTER_TIMEOUT_S = 120
NON_EMPTY_OPTIONS_REASON = (
    "non-empty options are not in the verified cross-stack option mapping"
)


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


def generate_fixtures(
    directory: Path,
    *,
    rows: int | None = None,
    tokens_per_row: int | None = None,
    include_public: bool = False,
) -> list[FormatCase]:
    """Generate the deterministic, semantically verified fixture catalog."""
    return generate_format_cases(
        directory,
        rows=rows,
        tokens_per_row=tokens_per_row,
        include_public=include_public,
    )


def parse_manifest(
    manifest_path: Path,
) -> tuple[list[FormatCase], list[dict[str, str]]]:
    """Parse a schema-v2 manifest and explicitly skip unverified formats."""
    try:
        document = orjson.loads(manifest_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("manifest root must be an object")
    schema_version = document.get("schema_version")
    if schema_version not in (1, SCHEMA_VERSION):
        raise ValueError(f"manifest schema_version must be 1 or {SCHEMA_VERSION}")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be a list")

    cases: list[FormatCase] = []
    skips: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {index} must be an object")
        format_name = entry.get("format")
        if not isinstance(format_name, str) or not format_name:
            raise ValueError(f"manifest entry {index} has invalid format")
        if format_name in EXCLUDED_FORMATS or format_name not in SUPPORTED_FORMATS:
            skips.append(
                {
                    "format": format_name,
                    "reason": unsupported_format_reason(format_name),
                }
            )
            continue

        profile = profile_for(format_name)
        if profile is None:
            skips.append(
                {
                    "format": format_name,
                    "reason": unsupported_format_reason(format_name),
                }
            )
            continue

        options = entry.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"manifest entry {index} options must be an object")
        if options != profile.verified_options:
            skips.append({"format": format_name, "reason": NON_EMPTY_OPTIONS_REASON})
            continue
        if schema_version == 1 and profile.source_kind != "local_file":
            skips.append(
                {
                    "format": format_name,
                    "reason": (
                        f"{format_name} requires a schema-v2 source envelope; "
                        "legacy path-only manifests support local fixtures only"
                    ),
                }
            )
            continue

        if schema_version == SCHEMA_VERSION and "source" in entry:
            source_raw = entry["source"]
            if not isinstance(source_raw, dict):
                raise ValueError(f"manifest entry {index} source must be an object")
            source = SourceEnvelope.from_dict(source_raw)
            if source.kind == "local_file" and source.path:
                path = Path(source.path)
                if not path.is_absolute():
                    path = manifest_path.parent / path
                source = SourceEnvelope(kind="local_file", path=str(path.resolve()))
        else:
            input_path = entry.get("path")
            if not isinstance(input_path, str) or not input_path:
                raise ValueError(f"manifest entry {index} has invalid path")
            path = Path(input_path)
            if not path.is_absolute():
                path = manifest_path.parent / path
            source = SourceEnvelope(kind="local_file", path=str(path.resolve()))

        cases.append(
            FormatCase(
                format=format_name,
                fixture_id=f"manifest-{index}-{format_name}",
                options=dict(profile.verified_options),
                source=source,
            )
        )
    return cases, skips


def summarize_samples(samples: Sequence[Sample]) -> dict[str, int | float | None]:
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
    for field in parity_fields_for(left.format):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value != right_value:
            raise ValueError(f"{field} mismatch: {left_value} != {right_value}")


def _adapter_arguments(
    case: FormatCase,
    *,
    seed: int,
    model: str,
    tokenizer: str,
    apply_chat_template: bool,
    exact_isl: bool,
) -> list[str]:
    path = case.path
    arguments = [
        "--format",
        case.format,
        "--options-json",
        orjson.dumps(case.options).decode(),
        "--source-json",
        source_json_for_adapters(case),
        "--fixture-id",
        case.fixture_id,
        "--seed",
        str(seed),
        "--model",
        model,
        "--tokenizer",
        tokenizer,
    ]
    if apply_chat_template:
        arguments.append("--apply-chat-template")
    if exact_isl:
        arguments.append("--exact-isl")
    if path is not None:
        arguments.extend(["--path", str(path)])
    else:
        arguments.extend(["--path", ""])
    return arguments


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
    offline: bool,
    tokenizer: str = "builtin",
    apply_chat_template: bool = False,
    exact_isl: bool = False,
) -> Sample:
    full_command = [
        *command,
        *_adapter_arguments(
            case,
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            exact_isl=exact_isl,
        ),
    ]
    env = os.environ.copy()
    if case.source.kind == "inline_synthetic":
        # Synthetic cross-language comparisons should default both adapters to the
        # Python-compatible RNG lane so prompt sampling, prefix reuse, and timing
        # draws share one authored reference path unless the caller overrides it.
        env.setdefault("AIPERF_RNG_BACKEND", "python")
    if offline and case.source.kind == "public_cached":
        env.update(OFFLINE_ENV)
    try:
        completed = runner(
            full_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=ADAPTER_TIMEOUT_S,
            env=env,
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
    tokenizer: str = "builtin",
    apply_chat_template: bool = False,
    exact_isl: bool = False,
    offline_public: bool = True,
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
                    tokenizer=tokenizer,
                    apply_chat_template=apply_chat_template,
                    exact_isl=exact_isl,
                    runner=runner,
                    offline=offline_public,
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
    cases: Sequence[FormatCase],
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
        profile = profile_for(format_name)
        formats[format_name] = {
            "python": python_summary,
            "rust": rust_summary,
            "rust_speedup": (
                python_summary["median_elapsed_ns"] / rust_summary["median_elapsed_ns"]
            ),
            "public_aliases": list(profile.public_aliases) if profile else [],
            "source_kind": profile.source_kind if profile else None,
        }

    catalog = {
        case.format: {
            "fixture_id": case.fixture_id,
            "options": dict(case.options),
            "source": case.source.to_dict(),
        }
        for case in cases
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "options": dict(options),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "rust_binary_identity": rust_binary_identity,
        },
        "catalog": catalog,
        "raw_samples": [sample.to_dict() for sample in samples],
        "formats": formats,
        "skips": [dict(skip) for skip in skips],
        "failures": report_failures,
    }


def _default_adapter_commands() -> dict[str, list[str]]:
    repository_root = Path(__file__).resolve().parents[2]
    rust_binary = repository_root / "rust/target/release/examples/dataset_load_bench"
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
    parser.add_argument(
        "--skip-public-prefetch",
        action="store_true",
        help="Skip untimed public-source prefetch (local/synthetic cases only)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default="builtin")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--exact-isl", action="store_true")
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
        if format_name in EXCLUDED_FORMATS or format_name not in SUPPORTED_FORMATS:
            skips.append(
                {
                    "format": format_name,
                    "reason": unsupported_format_reason(format_name),
                }
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


def _partition_documented_skips(
    cases: Sequence[FormatCase],
    skips: list[dict[str, str]],
) -> list[FormatCase]:
    """Drop supported-but-unrunnable families, recording their documented reason.

    These families (for example streaming datasets that cannot be materialized
    offline) are removed before prefetch and timing so they neither hit the
    network nor produce a misleading failure.
    """
    runnable: list[FormatCase] = []
    for case in cases:
        reason = documented_skip_for(case.format)
        if reason is None:
            runnable.append(case)
        else:
            skips.append({"format": case.format, "reason": reason})
    return runnable


def _print_console_report(report: Mapping[str, object]) -> None:
    formats = report["formats"]
    assert isinstance(formats, dict)
    if formats:
        print(
            f"{'format':<24} {'python median':>14} {'rust median':>14} {'speedup':>10}"
        )
    for format_name, value in formats.items():
        assert isinstance(value, dict)
        python_summary = value["python"]
        rust_summary = value["rust"]
        assert isinstance(python_summary, dict)
        assert isinstance(rust_summary, dict)
        print(
            f"{format_name:<24} "
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
    include_public = not args.skip_public_prefetch
    if args.manifest is None:
        cases = generate_fixtures(
            fixture_directory,
            rows=args.rows,
            tokens_per_row=args.tokens_per_row,
            include_public=include_public,
        )
    else:
        cases, manifest_skips = parse_manifest(args.manifest)
        skips.extend(manifest_skips)
    cases = _select_cases(cases, args.formats, skips)
    cases = _partition_documented_skips(cases, skips)
    if include_public:
        cases, prefetch_skips = prefetch_public_cases(
            cases, seed=args.seed, row_limit=args.rows
        )
        skips.extend(prefetch_skips)

    samples, failures = run_comparison(
        cases,
        warmups=args.warmups,
        runs=args.runs,
        adapter_commands=adapter_commands,
        runner=runner,
        seed=args.seed,
        model=args.model,
        tokenizer=args.tokenizer,
        apply_chat_template=args.apply_chat_template,
        exact_isl=args.exact_isl,
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
            "skip_public_prefetch": args.skip_public_prefetch,
            "seed": args.seed,
            "model": args.model,
            "tokenizer": args.tokenizer,
            "apply_chat_template": args.apply_chat_template,
            "exact_isl": args.exact_isl,
        },
        cases=cases,
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
