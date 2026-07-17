# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.config.config import BenchmarkConfig
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import _apply_phase_loadgen_overrides
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.sweep import expand_sweep
from aiperf.timing.config import TimingConfig

_ENDPOINT = {"urls": ["http://localhost:8000/v1/chat/completions"]}
_DATASETS = [
    {
        "name": "main",
        "type": "synthetic",
        "entries": 10,
        "prompts": {"isl": 16, "osl": 8},
    }
]
_PHASE = {"type": "concurrency", "requests": 2, "concurrency": 1}


def _cfg(phases: list[dict]) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": _DATASETS,
            "phases": phases,
        }
    )


def _envelope(phases: list[dict]) -> dict:
    return {
        "benchmark": {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": _DATASETS,
            "phases": phases,
        }
    }


def test_flat_phases_shorthand_gets_profiling_kind() -> None:
    cfg = _cfg({"type": "concurrency", "requests": 2, "concurrency": 1})

    assert len(cfg.phases) == 1
    assert cfg.phases[0].name == "profiling"
    assert cfg.phases[0].kind == "profiling"
    assert cfg.phases[0].exclude_from_results is False


def test_warmup_profiling_shorthand_gets_explicit_kinds() -> None:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": _ENDPOINT,
            "datasets": _DATASETS,
            "warmup": {"type": "concurrency", "requests": 1, "concurrency": 1},
            "profiling": {"type": "concurrency", "requests": 2, "concurrency": 1},
        }
    )

    assert [(p.name, p.kind, p.exclude_from_results) for p in cfg.phases] == [
        ("warmup", "warmup", True),
        ("profiling", "profiling", False),
    ]


def test_no_profiling_kind_fails_with_legacy_message() -> None:
    with pytest.raises(ValidationError, match="a 'profiling' phase is required"):
        _cfg([{"name": "setup", "kind": "warmup", **_PHASE}])


def test_profiling_kind_allows_noncanonical_name_as_only_results_phase() -> None:
    cfg = _cfg([{"name": "storm_1", "kind": "profiling", **_PHASE}])

    assert cfg.get_profiling_phases()[0].name == "storm_1"


def test_legacy_canonical_names_infer_kind() -> None:
    cfg = _cfg([{"name": "warmup", **_PHASE}, {"name": "profiling", **_PHASE}])

    assert [(p.name, p.kind, p.exclude_from_results) for p in cfg.phases] == [
        ("warmup", "warmup", True),
        ("profiling", "profiling", False),
    ]


def test_custom_name_without_kind_fails() -> None:
    with pytest.raises(ValidationError, match="kind is required"):
        _cfg([{"name": "storm_1", **_PHASE}])


def test_custom_names_with_multiple_profiling_phases_pass() -> None:
    cfg = _cfg(
        [
            {"name": "low_cancel_1", "kind": "profiling", **_PHASE},
            {"name": "storm_1", "kind": "profiling", **_PHASE},
            {"name": "recovery_1", "kind": "profiling", **_PHASE},
        ]
    )

    assert [p.name for p in cfg.get_profiling_phases()] == [
        "low_cancel_1",
        "storm_1",
        "recovery_1",
    ]


def test_three_plus_phases_can_mix_multiple_warmups_and_profiling_phases() -> None:
    cfg = _cfg(
        [
            {"name": "prime_cache", "kind": "warmup", **_PHASE},
            {"name": "stabilize", "kind": "warmup", **_PHASE},
            {"name": "low", "kind": "profiling", **_PHASE},
            {"name": "storm", "kind": "profiling", **_PHASE},
            {"name": "recover", "kind": "profiling", **_PHASE},
        ]
    )

    assert [p.name for p in cfg.get_warmup_phases()] == [
        "prime_cache",
        "stabilize",
    ]
    assert [p.exclude_from_results for p in cfg.get_warmup_phases()] == [True, True]
    assert [p.name for p in cfg.get_profiling_phases()] == [
        "low",
        "storm",
        "recover",
    ]
    assert [p.exclude_from_results for p in cfg.get_profiling_phases()] == [
        False,
        False,
        False,
    ]


def test_staged_cache_warmups_execute_in_order_without_profiling_indexes() -> None:
    cfg = _cfg(
        [
            {
                "name": "cold_cache_warmup",
                "kind": "warmup",
                "requests": 50,
                "concurrency": 4,
                "type": "concurrency",
            },
            {
                "name": "warm_cache_warmup",
                "kind": "warmup",
                "requests": 100,
                "concurrency": 16,
                "type": "concurrency",
            },
            {
                "name": "steady_state",
                "kind": "profiling",
                "duration": "30m",
                "concurrency": 128,
                "type": "concurrency",
            },
        ]
    )
    run = BenchmarkRun(
        benchmark_id="run", cfg=cfg, artifact_dir=Path("/tmp/aiperf-test-artifacts")
    )

    timing = TimingConfig.from_run(run)

    assert [p.phase_name for p in timing.phase_configs] == [
        "cold_cache_warmup",
        "warm_cache_warmup",
        "steady_state",
    ]
    assert [p.phase_kind for p in timing.phase_configs] == [
        "warmup",
        "warmup",
        "profiling",
    ]
    assert [p.phase_index for p in timing.phase_configs] == [0, 1, 2]
    assert [p.profiling_index for p in timing.phase_configs] == [None, None, 0]
    assert [p.exclude_from_results for p in cfg.phases] == [True, True, False]


def test_multiple_warmup_phases_without_profiling_still_fails() -> None:
    with pytest.raises(ValidationError, match="a 'profiling' phase is required"):
        _cfg(
            [
                {"name": "prime_cache", "kind": "warmup", **_PHASE},
                {"name": "stabilize", "kind": "warmup", **_PHASE},
            ]
        )


def test_case_insensitive_duplicate_names_fail() -> None:
    with pytest.raises(ValidationError, match="case-insensitively"):
        _cfg(
            [
                {"name": "Storm", "kind": "profiling", **_PHASE},
                {"name": "storm", "kind": "profiling", **_PHASE},
            ]
        )


@pytest.mark.parametrize(
    "name,kind", [("warmup", "profiling"), ("profiling", "warmup")]
)
def test_reserved_canonical_names_require_matching_kind(name: str, kind: str) -> None:
    with pytest.raises(ValidationError, match="reserved for kind"):
        _cfg([{"name": name, "kind": kind, **_PHASE}])


def test_kind_drives_exclude_from_results_validation() -> None:
    with pytest.raises(ValidationError, match="exclude_from_results must be True"):
        _cfg(
            [
                {
                    "name": "setup",
                    "kind": "warmup",
                    "exclude_from_results": False,
                    **_PHASE,
                },
                {"name": "main", "kind": "profiling", **_PHASE},
            ]
        )


def test_profiling_kind_cannot_be_explicitly_excluded_from_results() -> None:
    with pytest.raises(ValidationError, match="exclude_from_results must be False"):
        _cfg(
            [
                {
                    "name": "storm",
                    "kind": "profiling",
                    "exclude_from_results": True,
                    **_PHASE,
                },
            ]
        )


def test_strict_phase_name_regex_rejects_path_unsafe_names() -> None:
    with pytest.raises(ValidationError):
        _cfg([{"name": "storm.1", "kind": "profiling", **_PHASE}])


@pytest.mark.parametrize("name", ["NUL", "nul", "Com1", "lpt9", "AUX"])
def test_phase_name_rejects_windows_reserved_device_names(name: str) -> None:
    with pytest.raises(ValidationError, match="reserved by Windows"):
        _cfg([{"name": name, "kind": "profiling", **_PHASE}])


def test_phase_name_allows_reserved_name_neighbors() -> None:
    cfg = _cfg([{"name": "com10", "kind": "profiling", **_PHASE}])

    assert cfg.phases[0].name == "com10"


def test_sweep_path_resolves_phase_name_and_numeric_index() -> None:
    base = _envelope(
        [
            {"name": "warmup", "kind": "warmup", **_PHASE},
            {"name": "storm_1", "kind": "profiling", **_PHASE},
        ]
    )
    by_name = {
        **base,
        "sweep": {"type": "grid", "parameters": {"phases.storm_1.concurrency": [8]}},
    }
    by_index = {
        **base,
        "sweep": {"type": "grid", "parameters": {"phases.1.concurrency": [16]}},
    }

    assert expand_sweep(by_name)[0][0]["benchmark"]["phases"][1]["concurrency"] == 8
    assert expand_sweep(by_index)[0][0]["benchmark"]["phases"][1]["concurrency"] == 16


def test_legacy_phases_profiling_path_targets_unique_profiling_kind() -> None:
    base = _envelope([{"name": "storm_1", "kind": "profiling", **_PHASE}])
    base["sweep"] = {
        "type": "grid",
        "parameters": {"phases.profiling.concurrency": [4]},
    }

    assert expand_sweep(base)[0][0]["benchmark"]["phases"][0]["concurrency"] == 4


def test_legacy_phases_profiling_path_fails_when_ambiguous() -> None:
    base = _envelope(
        [
            {"name": "low", "kind": "profiling", **_PHASE},
            {"name": "storm", "kind": "profiling", **_PHASE},
        ]
    )
    base["sweep"] = {
        "type": "grid",
        "parameters": {"phases.profiling.concurrency": [4]},
    }

    with pytest.raises(ValueError) as exc_info:
        expand_sweep(base)

    message = str(exc_info.value)
    assert "phases.profiling.concurrency" in message
    assert "ambiguous" in message
    assert "2 profiling phases exist" in message
    assert "low, storm" in message
    assert "phases.low.concurrency" in message
    assert "phases.0.concurrency" in message


def test_cli_loadgen_overlays_unique_profiling_kind() -> None:
    merged = _envelope([{"name": "storm_1", "kind": "profiling", **_PHASE}])
    cli = CLIConfig(concurrency=7)

    _apply_phase_loadgen_overrides(merged, cli)

    assert merged["benchmark"]["phases"][0]["concurrency"] == 7


def test_cli_loadgen_overlay_fails_on_multiple_profiling_phases() -> None:
    merged = _envelope(
        [
            {"name": "low", "kind": "profiling", **_PHASE},
            {"name": "storm", "kind": "profiling", **_PHASE},
        ]
    )
    cli = CLIConfig(concurrency=7)

    with pytest.raises(Exception, match="2 profiling phases"):
        _apply_phase_loadgen_overrides(merged, cli)


def test_sweep_legacy_profiling_path_uses_kind_before_non_warmup_fallback() -> None:
    base = _envelope(
        [
            {"name": "setup", "kind": "warmup", **_PHASE},
            {"name": "storm_1", "kind": "profiling", **_PHASE},
        ]
    )
    base["sweep"] = {
        "type": "grid",
        "parameters": {"phases.profiling.concurrency": [9]},
    }

    expanded = expand_sweep(base)[0][0]["benchmark"]["phases"]

    assert expanded[0]["concurrency"] == 1
    assert expanded[1]["concurrency"] == 9


def test_cli_loadgen_overlay_ignores_warmup_kind_when_finding_profiling() -> None:
    merged = _envelope(
        [
            {"name": "setup", "kind": "warmup", **_PHASE},
            {"name": "storm_1", "kind": "profiling", **_PHASE},
        ]
    )
    cli = CLIConfig(concurrency=11)

    _apply_phase_loadgen_overrides(merged, cli)

    assert merged["benchmark"]["phases"][0]["concurrency"] == 1
    assert merged["benchmark"]["phases"][1]["concurrency"] == 11


def test_timing_config_preserves_order_indexes_and_phase_cancellation() -> None:
    cfg = _cfg(
        [
            {
                "name": "low",
                "kind": "profiling",
                "cancellation": {"rate": 5, "delay": 0},
                **_PHASE,
            },
            {
                "name": "storm",
                "kind": "profiling",
                "cancellation": {"rate": 50, "delay": 1},
                **_PHASE,
            },
            {"name": "setup", "kind": "warmup", **_PHASE},
        ]
    )
    run = BenchmarkRun(
        benchmark_id="run", cfg=cfg, artifact_dir=Path("/tmp/aiperf-test-artifacts")
    )

    timing = TimingConfig.from_run(run)

    assert [p.phase_name for p in timing.phase_configs] == ["low", "storm", "setup"]
    assert [p.phase_index for p in timing.phase_configs] == [0, 1, 2]
    assert [p.profiling_index for p in timing.phase_configs] == [0, 1, None]
    assert [p.request_cancellation.rate for p in timing.phase_configs] == [
        5.0,
        50.0,
        None,
    ]


def test_timing_config_inherits_profiling_cancellation_default_only() -> None:
    cfg = _cfg(
        [
            {
                "name": "setup",
                "kind": "warmup",
                "cancellation": {"rate": 99, "delay": 0},
                **_PHASE,
            },
            {
                "name": "cancel_phase",
                "kind": "profiling",
                "cancellation": {"rate": 25, "delay": 1},
                **_PHASE,
            },
            {"name": "inherit_phase", "kind": "profiling", **_PHASE},
            {"name": "warmup_omitted", "kind": "warmup", **_PHASE},
        ]
    )
    run = BenchmarkRun(
        benchmark_id="run", cfg=cfg, artifact_dir=Path("/tmp/aiperf-test-artifacts")
    )

    timing = TimingConfig.from_run(run)

    by_name = {phase.phase_name: phase for phase in timing.phase_configs}
    assert by_name["setup"].request_cancellation.rate == 99.0
    assert by_name["cancel_phase"].request_cancellation.rate == 25.0
    assert by_name["inherit_phase"].request_cancellation.rate == 25.0
    assert by_name["inherit_phase"].request_cancellation.delay == 1.0
    assert by_name["warmup_omitted"].request_cancellation.rate is None
    assert timing.request_cancellation.rate == 25.0


def test_timing_config_defaults_phase_metadata_for_legacy_programmatic_phases() -> None:
    cfg = _cfg([{"name": "profiling", **_PHASE}])
    run = BenchmarkRun(
        benchmark_id="run", cfg=cfg, artifact_dir=Path("/tmp/aiperf-test-artifacts")
    )

    timing = TimingConfig.from_run(run)

    assert [
        (p.phase_index, p.profiling_index, p.phase_name, p.phase_kind)
        for p in timing.phase_configs
    ] == [(0, 0, "profiling", "profiling")]
