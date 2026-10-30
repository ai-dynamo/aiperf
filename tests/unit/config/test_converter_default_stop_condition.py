# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI default stop condition when no requests, sessions, or duration is given.

Benchmark suites (``full_pass_by_default`` loaders such as SPEED-Bench) run one
full pass (``sessions: all``); concurrency runs default to
``max(10, 2 x concurrency)``; everything else keeps the flat 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest
from pydantic import TypeAdapter, ValidationError
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.phases import FULL_PASS_SESSIONS, PhaseConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.plugin import plugins
from aiperf.timing.config import TimingConfig
from tests.unit.conftest import make_run_from_cli


def _make_cli(**overrides: Any) -> CLIConfig:
    return CLIConfig(
        url="http://localhost:8000", model_names=["test-model"], **overrides
    )


def _stop_fields(prof: dict[str, Any]) -> dict[str, Any]:
    return {k: prof[k] for k in ("requests", "sessions", "duration") if k in prof}


@pytest.mark.parametrize(
    "overrides,expected",
    [
        param({}, {"requests": 10}, id="no_flags"),
        param({"concurrency": 4}, {"requests": 10}, id="low_concurrency_floor"),
        param({"concurrency": 80}, {"requests": 160}, id="concurrency_scales"),
        param({"concurrency": [10, 20]}, {"requests": 40}, id="sweep_uses_largest_value"),
        param({"search_space": ["concurrency:64,128:int"]}, {"requests": 256}, id="search_uses_upper_bound"),
        param({"search_space": ["phases.profiling.concurrency:8,20.5"]}, {"requests": 42}, id="search_real_upper_bound_rounds_up"),
        param({"search_space": ["rate:1,50"]}, {"requests": 10}, id="search_rate_flat"),
        param({"search_space": ["phases.warmup.concurrency:1,2", "prefill_concurrency:1,4:int", "concurrency:8,32:int"]}, {"requests": 64}, id="search_skips_other_dimensions"),
        param({"search_space": ["concurrency:64:int"]}, {"requests": 10}, id="search_malformed_bounds_left_to_real_parser"),
        param({"request_rate": 5.0}, {"requests": 10}, id="request_rate_flat"),
        param({"request_rate": 5.0, "concurrency": 80}, {"requests": 10}, id="request_rate_with_concurrency_cap"),
        param({"public_dataset": "sharegpt", "concurrency": 80}, {"requests": 160}, id="non_suite_public_dataset"),
        param({"public_dataset": "speed_bench_coding", "concurrency": 80}, {"sessions": "all"}, id="speed_bench_category"),
        param({"public_dataset": "spec_al_mtbench", "request_rate": 5.0}, {"sessions": "all"}, id="spec_al_request_rate"),
    ],
)  # fmt: skip
def test_build_profiling_no_stop_condition_applies_default(
    overrides: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert _stop_fields(build_profiling(_make_cli(**overrides))) == expected


def test_build_profiling_custom_suite_dataset_runs_full_pass(tmp_path: Path) -> None:
    input_file = tmp_path / "qualitative.jsonl"
    input_file.write_text('{"question_id": "q", "messages": []}\n')
    cli = _make_cli(
        custom_dataset_type="speed_bench_qualitative",
        input_file=str(input_file),
        concurrency=80,
    )

    assert _stop_fields(build_profiling(cli)) == {"sessions": "all"}


@pytest.mark.parametrize(
    "overrides,expected",
    [
        param({"request_count": 25}, {"requests": 25}, id="request_count"),
        param({"conversation_num": 7}, {"sessions": 7}, id="num_conversations"),
        param({"benchmark_duration": 30.0}, {"duration": 30.0}, id="duration"),
    ],
)  # fmt: skip
def test_build_profiling_explicit_stop_condition_skips_default(
    overrides: dict[str, Any], expected: dict[str, Any]
) -> None:
    cli = _make_cli(public_dataset="speed_bench_coding", concurrency=80, **overrides)

    assert _stop_fields(build_profiling(cli)) == expected


def test_build_profiling_user_centric_suite_keeps_flat_default() -> None:
    cli = _make_cli(
        public_dataset="speed_bench_coding",
        user_centric_rate=5.0,
        num_users=4,
        conversation_turn_mean=2,
    )

    assert _stop_fields(build_profiling(cli)) == {"requests": 10}


def test_convert_cli_suite_dataset_maps_to_sessions_from_dataset() -> None:
    cli = _make_cli(public_dataset="speed_bench_coding", concurrency=80)

    profiling = convert_cli_to_aiperf(cli).benchmark.phases[-1]
    assert profiling.sessions == FULL_PASS_SESSIONS
    assert profiling.requests is None

    # Kubernetes service pods rebuild the run from its serialized JSON
    # (serialized as in aiperf.kubernetes.resources).
    serialized = make_run_from_cli(cli).model_dump(
        mode="json", exclude_none=True, exclude_unset=True
    )
    run = BenchmarkRun.model_validate(orjson.loads(orjson.dumps(serialized)))
    timing = TimingConfig.from_run(run)
    credit_phase = next(
        pc for pc in timing.phase_configs if pc.phase == CreditPhase.PROFILING
    )
    assert credit_phase.sessions_from_dataset is True
    assert credit_phase.expected_num_sessions is None
    assert credit_phase.total_expected_requests is None


@pytest.mark.parametrize(
    "sessions",
    [param("some", id="unknown_string"), param(0, id="zero")],
)  # fmt: skip
def test_phase_config_sessions_rejects_invalid_values(sessions: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(PhaseConfig).validate_python(
            {"name": "profiling", "type": "concurrency", "sessions": sessions}
        )


def test_phase_config_user_centric_rejects_sessions_all() -> None:
    with pytest.raises(ValidationError, match="not supported for user-centric"):
        TypeAdapter(PhaseConfig).validate_python(
            {
                "name": "profiling",
                "type": "user_centric",
                "rate": 5.0,
                "users": 4,
                "sessions": "all",
            }
        )


def test_full_pass_by_default_set_on_every_spec_decode_suite() -> None:
    public = {
        e.name
        for e in plugins.list_entries("public_dataset_loader")
        if e.name.startswith(("speed_bench_", "spec_al_")) or e.name == "spec_bench"
    }
    custom = {
        e.name
        for e in plugins.list_entries("custom_dataset_loader")
        if e.name.startswith("speed_bench_")
    }

    assert public and custom
    assert all(
        plugins.get_public_dataset_loader_metadata(n).full_pass_by_default
        for n in public
    )
    assert all(
        plugins.get_dataset_loader_metadata(n).full_pass_by_default for n in custom
    )
    assert not plugins.get_public_dataset_loader_metadata(
        "sharegpt"
    ).full_pass_by_default
