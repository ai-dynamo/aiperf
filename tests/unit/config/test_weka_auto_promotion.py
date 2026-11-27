# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-promotion and timing validation must agree on what carries timing.

They did not. Validation consulted ``_implicit_timing_types`` -- loaders whose
timing is nested where a first-record probe cannot see it -- while
auto-promotion used a probe that only looked for a top-level ``timestamp`` key
and returned False for a directory. A weka_trace keeps timing in
``requests[].t`` and is documented as a *directory* of files, so it failed that
probe, never auto-promoted, and replayed as plain concurrency: a green run that
silently discarded the recorded timeline, which is the entire reason to replay
a trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from aiperf.config.dataset.resolver import _implicit_timing_types
from aiperf.config.flags._converter_profiling import _trace_carries_timing
from aiperf.plugin.enums import CustomDatasetType

BLOCK = 64


def _weka_trace(tmp_path: Path) -> Path:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "trace_0001.json").write_text(
        json.dumps(
            {
                "id": "demo",
                "models": ["m"],
                "block_size": BLOCK,
                "hash_id_scope": "local",
                "tool_tokens": 0,
                "system_tokens": 0,
                "requests": [
                    {
                        "t": 0.0,
                        "type": "n",
                        "model": "m",
                        "in": 4 * BLOCK,
                        "out": 16,
                        "hash_ids": [1, 2, 3, 4],
                        "stop": "end_turn",
                    }
                ],
            }
        )
    )
    return traces


def test_weka_trace_directory_is_recognised_as_carrying_timing(tmp_path) -> None:
    """The documented input shape: a directory, with timing nested per request."""
    assert _trace_carries_timing("weka_trace", _weka_trace(tmp_path)) is True


def test_weka_trace_single_file_is_recognised_too(tmp_path) -> None:
    traces = _weka_trace(tmp_path)
    assert _trace_carries_timing("weka_trace", traces / "trace_0001.json") is True


@pytest.mark.parametrize(
    "dataset_type",
    [
        param(str(t), id=str(t))
        for t in sorted(str(x) for x in _implicit_timing_types())
        if str(t) != "weka_trace"
    ],
)  # fmt: skip
def test_other_implicit_timing_types_keep_the_probe(
    dataset_type: str, tmp_path
) -> None:
    """Only weka_trace short-circuits; the rest still go through the probe.

    Sharing `_implicit_timing_types` here reads better and was the first
    attempt, but it silently changes the default timing mode for four other
    formats. Pinned so the tempting simplification is not made by accident.
    """
    assert _trace_carries_timing(dataset_type, tmp_path / "missing") is False


def test_an_untimed_dataset_still_falls_back_to_the_probe(tmp_path) -> None:
    """Scoping: a format with no implicit timing must not be promoted blindly."""
    plain = tmp_path / "plain.jsonl"
    plain.write_text(json.dumps({"text": "hello"}) + "\n")
    assert _trace_carries_timing("mooncake_trace", plain) is False


def test_a_timestamped_record_still_promotes(tmp_path) -> None:
    timed = tmp_path / "timed.jsonl"
    timed.write_text(json.dumps({"timestamp": 0, "text": "hello"}) + "\n")
    assert _trace_carries_timing("mooncake_trace", timed) is True


def _resolved_phase_type(dataset_type, path, **extra):
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            tokenizer_name="test-tokenizer",
            input_file=str(path),
            custom_dataset_type=dataset_type,
            artifact_directory=str(path if path.is_dir() else path.parent) + "/art",
            **extra,
        )
    )
    return str(run.cfg.phases[0].type)


def test_a_weka_directory_resolves_to_fixed_schedule(tmp_path) -> None:
    """Pins the wiring, not just the helper.

    Every other test here calls `_trace_carries_timing` directly, which leaves
    the promotion gate free to ignore it. This is the assertion that proves the
    headline behaviour: a documented weka invocation with no timing flags comes
    out as fixed schedule.
    """
    traces = _weka_trace(tmp_path)
    assert (
        _resolved_phase_type(CustomDatasetType.WEKA_TRACE, traces) == "fixed_schedule"
    )


@pytest.mark.parametrize(
    "flag",
    [
        param("ignore_trace_delays", id="ignore-trace-delays"),
        param("use_think_time_only", id="use-think-time-only"),
    ],
)  # fmt: skip
def test_timeline_opt_out_flags_prevent_promotion(tmp_path, flag: str) -> None:
    """Both flags say "do not replay the recorded timeline".

    Promoting anyway would accept the flag and then silently ignore it, because
    fixed schedule dispatches on recorded timestamps regardless.
    """
    traces = _weka_trace(tmp_path)
    resolved = _resolved_phase_type(
        CustomDatasetType.WEKA_TRACE, traces, **{flag: True}
    )
    assert resolved != "fixed_schedule"


def test_other_implicit_timing_types_are_not_promoted(tmp_path) -> None:
    """Compatibility boundary, deliberately pinned.

    `_implicit_timing_types` is the natural shared source of truth and reads
    better, but widening promotion to the whole set silently changes the default
    timing mode for burst_gpt_trace, sagemaker_data_capture, baseten_trace and
    tracelab -- and turns `burst_gpt_trace --request-rate 10` from a working
    command into a hard error, since the rate/fixed-schedule conflict check only
    fires once promotion is decided. Widening it is a per-format compatibility
    decision, not a side effect.
    """
    csv = tmp_path / "burst.csv"
    csv.write_text("Timestamp,Model,Request tokens,Response tokens\n0,m,10,5\n")
    assert (
        _resolved_phase_type(CustomDatasetType.BURST_GPT_TRACE, csv) != "fixed_schedule"
    )


def test_a_rate_flag_still_works_for_other_trace_types(tmp_path) -> None:
    """The regression that made the over-wide version unshippable."""
    csv = tmp_path / "burst.csv"
    csv.write_text("Timestamp,Model,Request tokens,Response tokens\n0,m,10,5\n")
    resolved = _resolved_phase_type(
        CustomDatasetType.BURST_GPT_TRACE, csv, request_rate=10.0
    )
    assert resolved != "fixed_schedule"


@pytest.mark.parametrize(
    "flag",
    [
        param("ignore_trace_delays", id="ignore-trace-delays"),
        param("use_think_time_only", id="use-think-time-only"),
    ],
)  # fmt: skip
def test_timeline_opt_outs_do_not_touch_other_trace_formats(
    tmp_path, flag: str
) -> None:
    """Scoping, pinned.

    Suppressing promotion for every format is a regression, not a safeguard: a
    50-row timestamped mooncake trace with either flag drops from
    fixed_schedule/50 to concurrency/10, silently replacing most of the
    workload. Only weka_trace's loader lets fixed schedule override these flags,
    so only there is there a silent no-op to prevent.
    """
    import json

    trace = tmp_path / "mooncake.jsonl"
    trace.write_text(
        "".join(
            json.dumps({"timestamp": i * 100, "input_length": 8, "output_length": 4})
            + "\n"
            for i in range(50)
        )
    )
    resolved = _resolved_phase_type(
        CustomDatasetType.MOONCAKE_TRACE, trace, **{flag: True}
    )
    assert resolved == "fixed_schedule"
