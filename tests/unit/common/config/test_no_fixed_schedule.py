# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace-dataset auto-promotion to fixed_schedule and --no-fixed-schedule.

When a CLI invocation supplies a trace ``--custom-dataset-type`` whose
first record carries a ``timestamp`` field, the CLI->YAML converter
promotes the profiling phase to ``fixed_schedule`` and fills
``phase.requests`` from the dataset record count. ``--no-fixed-schedule``
suppresses the promotion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import PhaseType


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


def _write_trace_file(
    tmp_path: Path,
    records: list[dict],
    *,
    name: str = "trace.jsonl",
) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


class TestTraceAutoPromotion:
    def test_trace_with_timestamps_auto_promotes_to_fixed_schedule(self, tmp_path):
        """mooncake_trace + timestamps -> phase.type flips to fixed_schedule."""
        trace = _write_trace_file(
            tmp_path,
            [
                {"timestamp": 0, "input_length": 100, "output_length": 50},
                {"timestamp": 100, "input_length": 120, "output_length": 60},
                {"timestamp": 200, "input_length": 130, "output_length": 70},
            ],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
        )

        prof = build_profiling(cli)

        assert prof["type"] == PhaseType.FIXED_SCHEDULE
        # records=3 -> requests autofills to 3
        assert prof.get("requests") == 3

    def test_no_fixed_schedule_flag_suppresses_promotion(self, tmp_path):
        """--no-fixed-schedule keeps the user-selected timing mode."""
        trace = _write_trace_file(
            tmp_path,
            [
                {"timestamp": 0, "input_length": 100, "output_length": 50},
                {"timestamp": 100, "input_length": 120, "output_length": 60},
            ],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            disable_auto_fixed_schedule=True,
        )

        prof = build_profiling(cli)

        assert prof["type"] != PhaseType.FIXED_SCHEDULE
        # Falls back to the generic 10-requests default for unbounded runs.
        assert prof.get("requests") == 10

    def test_trace_without_timestamps_does_not_promote(self, tmp_path):
        """Trace dataset whose first record lacks ``timestamp`` keeps the type."""
        trace = _write_trace_file(
            tmp_path,
            [
                {"input_length": 100, "output_length": 50},
                {"input_length": 120, "output_length": 60},
            ],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
        )

        prof = build_profiling(cli)

        assert prof["type"] != PhaseType.FIXED_SCHEDULE

    def test_explicit_fixed_schedule_fills_requests_from_record_count(self, tmp_path):
        """``--fixed-schedule`` alone fills phase.requests from the file."""
        trace = _write_trace_file(
            tmp_path,
            [
                {"timestamp": 0},
                {"timestamp": 100},
                {"timestamp": 200},
                {"timestamp": 300},
                {"timestamp": 400},
            ],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            fixed_schedule=True,
        )

        prof = build_profiling(cli)

        assert prof["type"] == PhaseType.FIXED_SCHEDULE
        assert prof.get("requests") == 5

    def test_explicit_request_count_overrides_autofill(self, tmp_path):
        """When --request-count is explicit, it wins over the file count."""
        trace = _write_trace_file(
            tmp_path,
            [{"timestamp": 0}, {"timestamp": 100}, {"timestamp": 200}],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            fixed_schedule=True,
            request_count=2,
        )

        prof = build_profiling(cli)

        assert prof["type"] == PhaseType.FIXED_SCHEDULE
        assert prof["requests"] == 2

    def test_non_trace_dataset_never_auto_promotes(self, tmp_path):
        """``single_turn`` is not a trace type even with a ``timestamp`` field."""
        plain = _write_trace_file(
            tmp_path,
            [{"prompt": "hi", "timestamp": 0}, {"prompt": "yo", "timestamp": 1}],
        )
        cli = _make_cli(
            input_file=str(plain),
            custom_dataset_type="single_turn",
        )

        prof = build_profiling(cli)

        assert prof["type"] != PhaseType.FIXED_SCHEDULE


class TestSweepIncompatibleWithFixedSchedule:
    """Parameter sweeps + fixed_schedule (explicit or auto-promoted) must error.

    Ports v1 ``validate_sweep_incompatibilities``. Fixed schedule replays
    a single timing pattern, so a magic-list sweep across concurrency /
    request_rate / etc. is meaningless — refuse loudly rather than
    silently running variation 0 only.
    """

    def test_explicit_fixed_schedule_plus_magic_list_concurrency_raises(self, tmp_path):
        trace = _write_trace_file(tmp_path, [{"timestamp": 0}, {"timestamp": 100}])
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            fixed_schedule=True,
            concurrency=[1, 2, 4],
        )
        with pytest.raises(ValueError, match="Parameter sweeps.*fixed-schedule"):
            build_profiling(cli)

    def test_auto_promoted_trace_plus_magic_list_concurrency_raises(self, tmp_path):
        """Auto-promoted trace + sweep is the silent failure mode v1 errored on.

        ``--concurrency`` is on BasePhaseConfig so it survives the
        rate/users/smoothness conflict check inside the auto-promote
        block — the sweep guard is what catches this combo.
        """
        trace = _write_trace_file(
            tmp_path,
            [
                {"timestamp": 0, "input_length": 10},
                {"timestamp": 100, "input_length": 20},
            ],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            concurrency=[1, 2, 4],
        )
        with pytest.raises(ValueError, match="Parameter sweeps.*fixed-schedule"):
            build_profiling(cli)

    def test_request_rate_sweep_against_trace_errors_via_conflict_guard(self, tmp_path):
        """A magic-list request_rate trips the earlier rate/users/smoothness
        conflict guard inside the auto-promote block (separate from the
        sweep guard but the same end result: user gets a clear error)."""
        trace = _write_trace_file(
            tmp_path,
            [{"timestamp": 0}, {"timestamp": 100}],
        )
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            request_rate=[10.0, 20.0],
        )
        with pytest.raises(ValueError, match="incompatible with fixed_schedule"):
            build_profiling(cli)

    def test_no_fixed_schedule_flag_allows_sweep_with_trace(self, tmp_path):
        """--no-fixed-schedule suppresses auto-promote, so sweep is allowed."""
        trace = _write_trace_file(tmp_path, [{"timestamp": 0}, {"timestamp": 100}])
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            disable_auto_fixed_schedule=True,
            concurrency=[1, 2],
        )
        prof = build_profiling(cli)
        assert prof["type"] != PhaseType.FIXED_SCHEDULE
        assert prof["concurrency"] == [1, 2]

    def test_explicit_fixed_schedule_without_sweep_is_fine(self, tmp_path):
        trace = _write_trace_file(tmp_path, [{"timestamp": 0}, {"timestamp": 100}])
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            fixed_schedule=True,
        )
        prof = build_profiling(cli)
        assert prof["type"] == PhaseType.FIXED_SCHEDULE

    def test_single_element_list_is_not_a_sweep(self, tmp_path):
        """A magic-list field with one element is degenerate; allow it."""
        trace = _write_trace_file(tmp_path, [{"timestamp": 0}, {"timestamp": 100}])
        cli = _make_cli(
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            fixed_schedule=True,
            concurrency=[1],
        )
        prof = build_profiling(cli)
        assert prof["type"] == PhaseType.FIXED_SCHEDULE
