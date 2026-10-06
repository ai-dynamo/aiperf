# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A request count that fixed schedule cannot honour must be announced.

Fixed-schedule replay sends the trace's own entries, so `--request-count` does
not bound the run. Discarding it in silence is the failure mode CLAUDE.md
forbids for flags under `--config` -- "never be silently ignored" -- and for a
benchmarking tool it means publishing a request count nobody asked for.
Measured before the fix: `--request-count 10` against a 31-row trace completed
31 requests with no warning anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli

WARNING_MARKER = "is not used under fixed-schedule"


def _trace(tmp_path: Path, rows: int = 31) -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "".join(
            json.dumps({"timestamp": i * 50, "input_length": 8, "output_length": 4})
            + "\n"
            for i in range(rows)
        )
    )
    return path


def _resolve(trace: Path, tmp_path: Path, **extra):
    return make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            tokenizer_name="m",
            input_file=str(trace),
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            artifact_directory=str(tmp_path / "artifacts"),
            **extra,
        )
    )


def test_explicit_request_count_under_fixed_schedule_warns(tmp_path, caplog) -> None:
    trace = _trace(tmp_path)
    with caplog.at_level("WARNING"):
        _resolve(trace, tmp_path, request_count=10)
    assert any(WARNING_MARKER in r.message for r in caplog.records), (
        "a request count the trace overrides must be announced, naming the flag"
    )


def test_the_warning_names_the_flag_and_the_remedy(tmp_path, caplog) -> None:
    """A warning that does not say what to do instead is only half useful."""
    trace = _trace(tmp_path)
    with caplog.at_level("WARNING"):
        _resolve(trace, tmp_path, request_count=10)
    text = " ".join(r.message for r in caplog.records)
    assert "--request-count" in text
    assert "--no-fixed-schedule" in text


@pytest.mark.parametrize(
    "extra, reason",
    [
        param({}, "no count was configured, so nothing is being overridden", id="no-request-count"),
        param(
            {"request_count": 10, "disable_auto_fixed_schedule": True},
            "the count is honoured when the run is not fixed-schedule",
            id="opted-out-of-fixed-schedule",
        ),
    ],
)  # fmt: skip
def test_no_warning_when_nothing_is_overridden(tmp_path, caplog, extra, reason) -> None:
    trace = _trace(tmp_path)
    with caplog.at_level("WARNING"):
        _resolve(trace, tmp_path, **extra)
    assert not any(WARNING_MARKER in r.message for r in caplog.records), reason
