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
