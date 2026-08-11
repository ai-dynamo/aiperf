# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI plumbing and converter-guard tests for the baseten_trace replay knobs.

Covers the open-loop default flip (``--open-loop-replay`` /
``--no-open-loop-replay``), the ``--open-loop-strict`` / ``--omit-kv-hints`` /
``--force-min-tokens`` boolean flags, the converter guard rejecting the
baseten-only knobs (value and boolean) on non-baseten datasets, the
contradictory ``--open-loop-strict`` + ``--no-open-loop-replay`` rejection
(a ``FileDataset`` model validator, so YAML configs are covered too),
the resolver warning for baseten-only knobs on auto-detected non-baseten
datasets, and the baseten_trace rejections of ``--synthesis-speedup-ratio``,
the hash-reshaping synthesis multipliers, and loader-colliding
``--extra-inputs`` keys.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from pydantic import ValidationError
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags._converter_profiling import (
    _count_dataset_records,
    _first_record_has_timestamp,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf

_REPLAY_BOOL_FIELDS = (
    "open_loop_replay",
    "open_loop_strict",
    "omit_kv_hints",
    "force_min_tokens",
)


# Same cyclopts capture pattern as tests/unit/config/test_auto_plot_fields.py
# (not importable from there: tests/unit/config is not a package).
def _parse_cli_args(argv: list[str]) -> CLIConfig:
    """Parse ``argv`` through cyclopts into a ``CLIConfig`` (no execution)."""
    from cyclopts import App

    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:  # pragma: no cover - capture only
        captured["uc"] = cli_config

    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["uc"]


@pytest.fixture
def trace_parquet(tmp_path: Path) -> Path:
    """A real minimal baseten_trace parquet (resolution peeks at row 0)."""
    path = tmp_path / "trace.parquet"
    rows = [
        {
            "timestamp_start_unix_ms": 100,
            "prompt": "hello",
            "input_tokens": 3,
            "output_tokens": 4,
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.mark.parametrize("suffix", [".arrow", ".ipc"])
def test_arrow_ipc_profiling_metadata(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"trace{suffix}"
    table = pa.Table.from_pylist(
        [
            {
                "timestamp_start_unix_ms": 100,
                "prompt": "hello",
                "input_tokens": 3,
                "output_tokens": 4,
            }
        ]
    )
    with (
        pa.OSFile(str(path), "wb") as sink,
        ipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table)

    assert _first_record_has_timestamp(path) is True
    assert _count_dataset_records(path) == 1


def _truncated_gz(tmp_path: Path) -> Path:
    import gzip
    import json

    gz = tmp_path / "trunc.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for _ in range(50):
            f.write(json.dumps({"timestamp": 0, "input_length": 4}) + "\n")
    # Keep only the gzip magic bytes — the first record is guaranteed cut off.
    gz.write_bytes(gz.read_bytes()[:20])
    return gz


def test_first_record_has_timestamp_returns_false_on_truncated_gz(
    tmp_path: Path,
) -> None:
    """A truncated gzip must degrade to False, not raise EOFError."""
    assert _first_record_has_timestamp(_truncated_gz(tmp_path)) is False


def test_count_dataset_records_returns_zero_on_truncated_gz(tmp_path: Path) -> None:
    """A truncated gzip must degrade to 0, not raise EOFError."""
    assert _count_dataset_records(_truncated_gz(tmp_path)) == 0


def _baseten_argv(trace_parquet: Path, *extra: str) -> list[str]:
    return [
        "--url",
        "http://localhost:8000/test",
        "--model",
        "test-model",
        "--endpoint-type",
        "completions",
        "--input-file",
        str(trace_parquet),
        "--custom-dataset-type",
        "baseten_trace",
        *extra,
    ]


def _dataset_from_argv(argv: list[str]):
    return convert_cli_to_aiperf(_parse_cli_args(argv)).benchmark.datasets[0]


class TestReplayBoolFlagPlumbing:
    def test_cyclopts_unset_replay_bools_keep_defaults(
        self, trace_parquet: Path
    ) -> None:
        uc = _parse_cli_args(_baseten_argv(trace_parquet))
        assert uc.open_loop_replay is True
        assert uc.open_loop_strict is False
        assert uc.omit_kv_hints is False
        assert uc.force_min_tokens is True
        assert not set(_REPLAY_BOOL_FIELDS) & uc.model_fields_set

    @pytest.mark.parametrize(
        ("extra", "field", "expected"),
        [
            param((), "open_loop_replay", True, id="open-loop-default-true"),
            param(("--open-loop-replay",), "open_loop_replay", True, id="open-loop-explicit"),
            param(("--no-open-loop-replay",), "open_loop_replay", False, id="closed-loop-selected"),
            param((), "open_loop_strict", False, id="strict-default-false"),
            param(("--open-loop-strict",), "open_loop_strict", True, id="strict-on"),
            param((), "omit_kv_hints", False, id="omit-kv-default-false"),
            param(("--omit-kv-hints",), "omit_kv_hints", True, id="omit-kv-on"),
            param((), "force_min_tokens", True, id="force-min-default-true"),
            param(("--no-force-min-tokens",), "force_min_tokens", False, id="force-min-off"),
        ],
    )  # fmt: skip
    def test_replay_bool_flag_lands_on_file_dataset(
        self, trace_parquet: Path, extra: tuple[str, ...], field: str, expected: bool
    ) -> None:
        dataset = _dataset_from_argv(_baseten_argv(trace_parquet, *extra))
        assert getattr(dataset, field) is expected


_BASETEN_ONLY_FLAG_ARGV = [
    param(("--trace-session-sample-ratio", "0.5"), id="sample-ratio"),
    param(("--replay-speedup", "10"), id="replay-speedup"),
    param(("--max-idle-gap-cap-seconds", "5"), id="idle-gap-cap"),
    param(("--open-loop-replay",), id="open-loop"),
    param(("--no-open-loop-replay",), id="no-open-loop"),
    param(("--open-loop-strict",), id="strict"),
    param(("--omit-kv-hints",), id="omit-kv-hints"),
    param(("--force-min-tokens",), id="force-min"),
    param(("--no-force-min-tokens",), id="no-force-min"),
]


class TestBasetenOnlyFlagGuard:
    """Every baseten-only replay knob (value or boolean) gets the same guard."""

    @pytest.mark.parametrize("flag_argv", _BASETEN_ONLY_FLAG_ARGV)  # fmt: skip
    def test_flag_with_non_baseten_type_rejected(
        self, tmp_path: Path, flag_argv: tuple[str, ...]
    ) -> None:
        mc_jsonl = tmp_path / "mc.jsonl"
        mc_jsonl.touch()
        cli = _parse_cli_args(
            [
                "--url",
                "http://localhost:8000/test",
                "--model",
                "test-model",
                "--input-file",
                str(mc_jsonl),
                "--custom-dataset-type",
                "mooncake_trace",
                *flag_argv,
            ]
        )
        with pytest.raises(
            ValueError,
            match=f"{flag_argv[0]}(/--no-[a-z-]+)? is only supported by the "
            "baseten_trace loader, but --custom-dataset-type is mooncake_trace",
        ):
            build_dataset(cli)

    @pytest.mark.parametrize(
        "flag_argv",
        [
            *_BASETEN_ONLY_FLAG_ARGV,
            param(("--public-dataset", "sharegpt", "--open-loop-strict"), id="public-dataset-strict"),
        ],
    )  # fmt: skip
    def test_flag_without_baseten_input_clean_error(
        self, flag_argv: tuple[str, ...]
    ) -> None:
        """Clean guard error (not a raw pydantic ``extra_forbidden`` crash)."""
        cli = _parse_cli_args(
            ["--url", "http://localhost:8000/test", "--model", "test-model", *flag_argv]
        )
        with pytest.raises(
            ValueError,
            match="is only supported by the baseten_trace "
            "loader; provide --input-file and --custom-dataset-type baseten_trace",
        ):
            convert_cli_to_aiperf(cli)

    @pytest.mark.parametrize(
        "explicit_type",
        [param(True, id="explicit-baseten-type"), param(False, id="auto-detected-type")],
    )  # fmt: skip
    def test_value_flags_accepted_with_baseten_trace(
        self, trace_parquet: Path, explicit_type: bool
    ) -> None:
        argv = _baseten_argv(
            trace_parquet,
            "--trace-session-sample-ratio",
            "0.5",
            "--replay-speedup",
            "10",
            "--max-idle-gap-cap-seconds",
            "5",
        )
        if not explicit_type:
            idx = argv.index("--custom-dataset-type")
            del argv[idx : idx + 2]
        dataset = _dataset_from_argv(argv)
        assert dataset.trace_session_sample_ratio == 0.5
        assert dataset.replay_speedup == 10.0
        assert dataset.max_idle_gap_cap_seconds == 5.0


class TestOpenLoopContradictionGuard:
    @pytest.mark.parametrize(
        "extra",
        [
            param(("--no-open-loop-replay", "--open-loop-strict"), id="replay-first"),
            param(("--open-loop-strict", "--no-open-loop-replay"), id="strict-first"),
        ],
    )  # fmt: skip
    def test_strict_with_closed_loop_rejected(
        self, trace_parquet: Path, extra: tuple[str, ...]
    ) -> None:
        cli = _parse_cli_args(_baseten_argv(trace_parquet, *extra))
        with pytest.raises(
            ValueError,
            match="--open-loop-strict requires open-loop replay; remove "
            "--no-open-loop-replay",
        ):
            convert_cli_to_aiperf(cli)

    def test_direct_model_construction_rejected(self, trace_parquet: Path) -> None:
        """YAML-path bypass: the combo must be rejected at the model level,
        not just by the CLI converter guard."""
        from aiperf.config.dataset import FileDataset

        with pytest.raises(
            ValidationError,
            match="--open-loop-strict requires open-loop replay; remove "
            "--no-open-loop-replay",
        ):
            FileDataset(
                name="d",
                type="file",
                path=trace_parquet,
                open_loop_strict=True,
                open_loop_replay=False,
            )

    def test_yaml_dict_dataset_rejected(self, trace_parquet: Path) -> None:
        """Dict validation (the YAML ingestion path) hits the same validator."""
        from aiperf.config import BenchmarkConfig

        with pytest.raises(
            ValidationError,
            match="--open-loop-strict requires open-loop replay; remove "
            "--no-open-loop-replay",
        ):
            BenchmarkConfig(
                models=["test-model"],
                endpoint={"urls": ["http://localhost:8000/v1/completions"]},
                datasets=[
                    {
                        "name": "main",
                        "type": "file",
                        "path": str(trace_parquet),
                        "open_loop_strict": True,
                        "open_loop_replay": False,
                    }
                ],
                phases=[
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "duration": 1,
                        "concurrency": 1,
                    }
                ],
            )

    def test_strict_with_explicit_open_loop_accepted(self, trace_parquet: Path) -> None:
        dataset = _dataset_from_argv(
            _baseten_argv(trace_parquet, "--open-loop-replay", "--open-loop-strict")
        )
        assert dataset.open_loop_replay is True
        assert dataset.open_loop_strict is True


class TestMaxIdleGapCapMustBePositive:
    """0 (or negative) silently disabled the reflow; ``gt=0`` rejects it."""

    @pytest.mark.parametrize("bad", ["0", "-1"])  # fmt: skip
    def test_cli_rejects_non_positive(self, trace_parquet: Path, bad: str) -> None:
        from cyclopts.exceptions import ValidationError as CycloptsValidationError

        with pytest.raises(
            CycloptsValidationError, match="Input should be greater than 0"
        ):
            _parse_cli_args(
                _baseten_argv(trace_parquet, "--max-idle-gap-cap-seconds", bad)
            )

    def test_file_dataset_rejects_zero(self, trace_parquet: Path) -> None:
        from aiperf.config.dataset import FileDataset

        with pytest.raises(ValidationError, match="greater than 0"):
            FileDataset(
                name="d",
                type="file",
                path=trace_parquet,
                max_idle_gap_cap_seconds=0.0,
            )


@pytest.fixture
def mooncake_jsonl(tmp_path: Path) -> Path:
    """A mooncake-shaped JSONL that auto-detects as mooncake_trace."""
    path = tmp_path / "mc.jsonl"
    path.write_text(
        '{"input_length": 100, "output_length": 50, "hash_ids": [1, 2], "timestamp": 1000}\n'
        '{"input_length": 200, "output_length": 75, "hash_ids": [3], "timestamp": 2000}\n'
    )
    return path


def _resolve_file_dataset(tmp_path: Path, path: Path, **fields: object) -> None:
    from aiperf.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.config.resolution.resolvers import DatasetResolver

    cfg = BenchmarkConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/completions"]},
        datasets=[{"name": "main", "type": "file", "path": str(path), **fields}],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 1,
                "concurrency": 1,
            }
        ],
    )
    run = BenchmarkRun(
        benchmark_id="test-run", cfg=cfg, artifact_dir=tmp_path / "artifacts"
    )
    DatasetResolver().resolve(run)


class TestAutoDetectedNonBasetenWarning:
    """Baseten-only knobs on an auto-detected non-baseten dataset pass the
    convert-time guard (type is unset), so the resolver must warn."""

    @pytest.mark.parametrize(
        "fields",
        [
            param({"replay_speedup": 10.0}, id="replay-speedup"),
            param({"open_loop_replay": False}, id="closed-loop-bool"),
        ],
    )  # fmt: skip
    def test_warns_on_auto_detected_mooncake(
        self,
        tmp_path: Path,
        mooncake_jsonl: Path,
        fields: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.resolver"):
            _resolve_file_dataset(tmp_path, mooncake_jsonl, **fields)
        (field_name,) = fields
        assert any(
            "baseten_trace-only" in r.message
            and "mooncake_trace" in r.message
            and field_name in r.message
            for r in caplog.records
        )

    def test_no_warning_for_baseten_parquet(
        self,
        tmp_path: Path,
        trace_parquet: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.resolver"):
            _resolve_file_dataset(tmp_path, trace_parquet, replay_speedup=10.0)
        assert not any("baseten_trace-only" in r.message for r in caplog.records)

    def test_no_warning_without_baseten_fields(
        self,
        tmp_path: Path,
        mooncake_jsonl: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.resolver"):
            _resolve_file_dataset(tmp_path, mooncake_jsonl)
        assert not any("baseten_trace-only" in r.message for r in caplog.records)


def _mooncake_argv(mooncake_jsonl: Path, *extra: str) -> list[str]:
    return [
        "--url",
        "http://localhost:8000/test",
        "--model",
        "test-model",
        "--endpoint-type",
        "completions",
        "--input-file",
        str(mooncake_jsonl),
        "--custom-dataset-type",
        "mooncake_trace",
        *extra,
    ]


class TestSynthesisSpeedupBasetenGuard:
    """Synthesis speedup rescales raw timestamps before replay, compounding
    with --replay-speedup and desyncing the think-time/idle-gap math that
    divides by replay_speedup only; rejected on explicit baseten_trace."""

    @pytest.mark.parametrize(
        "extra",
        [
            param(("--synthesis-speedup-ratio", "10"), id="synthesis-alone"),
            param(("--synthesis-speedup-ratio", "0.5"), id="slowdown-ratio"),
            param(("--replay-speedup", "10", "--synthesis-speedup-ratio", "10"), id="compounds-with-replay-speedup"),
        ],
    )  # fmt: skip
    def test_rejected_on_baseten_trace(
        self, trace_parquet: Path, extra: tuple[str, ...]
    ) -> None:
        cli = _parse_cli_args(_baseten_argv(trace_parquet, *extra))
        with pytest.raises(
            ValueError,
            match="--synthesis-speedup-ratio is not supported with "
            "--custom-dataset-type baseten_trace; use --replay-speedup",
        ):
            build_dataset(cli)

    def test_accepted_on_mooncake_trace(self, mooncake_jsonl: Path) -> None:
        dataset = _dataset_from_argv(
            _mooncake_argv(mooncake_jsonl, "--synthesis-speedup-ratio", "10")
        )
        assert dataset.synthesis.speedup_ratio == 10.0

    def test_output_len_synthesis_field_accepted_on_baseten(
        self, trace_parquet: Path
    ) -> None:
        dataset = _dataset_from_argv(
            _baseten_argv(trace_parquet, "--synthesis-output-len-multiplier", "2.0")
        )
        assert dataset.synthesis.output_len_multiplier == 2.0


class TestSynthesisHashReshapingBasetenGuard:
    """Prompt-shaping synthesis reshapes hash_ids while the wire still sends
    the recorded prompt, desyncing the forwarded KV hints from the prompt;
    rejected on explicit baseten_trace."""

    @pytest.mark.parametrize(
        "extra",
        [
            param(("--synthesis-prefix-len-multiplier", "2.0"), id="prefix-len"),
            param(("--synthesis-prefix-root-multiplier", "2"), id="prefix-root"),
            param(("--synthesis-prompt-len-multiplier", "2.0"), id="prompt-len"),
        ],
    )  # fmt: skip
    def test_rejected_on_baseten_trace(
        self, trace_parquet: Path, extra: tuple[str, ...]
    ) -> None:
        cli = _parse_cli_args(_baseten_argv(trace_parquet, *extra))
        with pytest.raises(ValueError, match="hash_ids"):
            build_dataset(cli)

    def test_accepted_on_mooncake_trace(self, mooncake_jsonl: Path) -> None:
        dataset = _dataset_from_argv(
            _mooncake_argv(mooncake_jsonl, "--synthesis-prompt-len-multiplier", "2.0")
        )
        assert dataset.synthesis.prompt_len_multiplier == 2.0


_COLLIDING_EXTRA_INPUTS = [
    param("min_tokens:5", "--no-force-min-tokens", id="min-tokens"),
    param("hash_ids:999", "--omit-kv-hints", id="hash-ids"),
    param("block_size:64", "--omit-kv-hints", id="block-size"),
]


class TestExtraInputsLoaderCollisionGuard:
    """Keys the baseten_trace loader injects per-turn silently clobber user
    --extra-inputs values on the wire; rejected unless the matching opt-out
    flag disables the injection (max_tokens is user-wins, not guarded)."""

    @pytest.mark.parametrize(("extra_input", "optout"), _COLLIDING_EXTRA_INPUTS)  # fmt: skip
    def test_colliding_key_rejected_with_optout_hint(
        self, trace_parquet: Path, extra_input: str, optout: str
    ) -> None:
        key = extra_input.split(":", 1)[0]
        cli = _parse_cli_args(
            _baseten_argv(trace_parquet, "--extra-inputs", extra_input)
        )
        with pytest.raises(
            ValueError,
            match=f"--extra-inputs {key} is overwritten per-turn by the "
            f"baseten_trace loader; pass {optout} to send your value",
        ):
            build_dataset(cli)

    @pytest.mark.parametrize(("extra_input", "optout"), _COLLIDING_EXTRA_INPUTS)  # fmt: skip
    def test_accepted_with_optout_flag(
        self, trace_parquet: Path, extra_input: str, optout: str
    ) -> None:
        cfg = convert_cli_to_aiperf(
            _parse_cli_args(
                _baseten_argv(trace_parquet, "--extra-inputs", extra_input, optout)
            )
        )
        key, value = extra_input.split(":", 1)
        assert cfg.benchmark.endpoint.extra[key] == int(value)

    def test_max_tokens_not_rejected(self, trace_parquet: Path) -> None:
        cfg = convert_cli_to_aiperf(
            _parse_cli_args(
                _baseten_argv(trace_parquet, "--extra-inputs", "max_tokens:99")
            )
        )
        assert cfg.benchmark.endpoint.extra["max_tokens"] == 99

    def test_non_baseten_dataset_unaffected(self, mooncake_jsonl: Path) -> None:
        cfg = convert_cli_to_aiperf(
            _parse_cli_args(
                _mooncake_argv(mooncake_jsonl, "--extra-inputs", "min_tokens:5")
            )
        )
        assert cfg.benchmark.endpoint.extra["min_tokens"] == 5
