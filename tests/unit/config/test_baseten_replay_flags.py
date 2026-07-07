# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI plumbing and converter-guard tests for the baseten_trace replay knobs.

Covers the open-loop default flip (``--open-loop-replay`` /
``--no-open-loop-replay``), the ``--open-loop-strict`` / ``--omit-kv-hints`` /
``--force-min-tokens`` boolean flags, and the converter guard rejecting the
value-typed baseten-only knobs on non-baseten datasets.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
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
    @pytest.mark.parametrize(
        ("flag", "field", "expected"),
        [
            param("--open-loop-replay", "open_loop_replay", True, id="open-loop"),
            param("--no-open-loop-replay", "open_loop_replay", False, id="closed-loop"),
            param("--open-loop-strict", "open_loop_strict", True, id="strict"),
            param("--omit-kv-hints", "omit_kv_hints", True, id="omit-kv-hints"),
            param("--force-min-tokens", "force_min_tokens", True, id="force-min"),
            param("--no-force-min-tokens", "force_min_tokens", False, id="no-force-min"),
        ],
    )  # fmt: skip
    def test_cyclopts_parses_replay_bool_both_polarities(
        self, trace_parquet: Path, flag: str, field: str, expected: bool
    ) -> None:
        uc = _parse_cli_args(_baseten_argv(trace_parquet, flag))
        assert getattr(uc, field) is expected
        assert field in uc.model_fields_set

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


_BASETEN_ONLY_VALUE_FLAGS = [
    param(("--trace-session-sample-ratio", "0.5"), id="sample-ratio"),
    param(("--replay-speedup", "10"), id="replay-speedup"),
    param(("--max-idle-gap-cap-seconds", "5"), id="idle-gap-cap"),
]


class TestBasetenOnlyValueFlagGuard:
    @pytest.mark.parametrize("flag_argv", _BASETEN_ONLY_VALUE_FLAGS)  # fmt: skip
    def test_value_flag_with_non_baseten_type_rejected(
        self, tmp_path: Path, flag_argv: tuple[str, str]
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
            match=f"{flag_argv[0]} is only supported by the baseten_trace "
            "loader, but --custom-dataset-type is mooncake_trace",
        ):
            build_dataset(cli)

    @pytest.mark.parametrize("flag_argv", _BASETEN_ONLY_VALUE_FLAGS)  # fmt: skip
    def test_value_flag_without_input_file_rejected(
        self, flag_argv: tuple[str, str]
    ) -> None:
        cli = _parse_cli_args(
            ["--url", "http://localhost:8000/test", "--model", "test-model", *flag_argv]
        )
        with pytest.raises(
            ValueError,
            match=f"{flag_argv[0]} is only supported by the baseten_trace "
            "loader; provide --input-file and --custom-dataset-type baseten_trace",
        ):
            build_dataset(cli)

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
