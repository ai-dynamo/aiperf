# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance must survive the sweep orchestrator's dump/validate round-trip.

The sweep orchestrator serializes each ``BenchmarkRun`` with
``model_dump(mode="json", exclude_none=True)`` (``orchestrator/local_executor.py``)
and rehydrates it with ``model_validate`` in the child process
(``orchestrator/subprocess_runner.py``). Every dumped key comes back inside
``model_fields_set``, so any check that reads ``model_fields_set`` to mean
"the user authored this" misfires for fields carrying a non-None default.
Because the resolver chain runs only in the subprocess entry point
(``cli_runner/_single_run.py``), those checks ALWAYS execute post-round-trip
in a sweep.

These tests mirror the orchestrator boundary exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

import orjson
import pytest
from pydantic import BaseModel

from aiperf.config.accuracy import AccuracyConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


def _round_trip(model: ModelT) -> ModelT:
    """Reproduce the orchestrator's dump/validate boundary for ``model``."""
    dumped: dict[str, Any] = model.model_dump(mode="json", exclude_none=True)
    return type(model).model_validate(orjson.loads(orjson.dumps(dumped)))


class TestAccuracyConfigRoundTrip:
    """An accuracy block without a benchmark must survive the round-trip."""

    def test_empty_accuracy_config_round_trips(self) -> None:
        cfg = AccuracyConfig()
        assert _round_trip(cfg).benchmark is None

    def test_dump_omits_unset_boolean_knobs(self) -> None:
        dumped = AccuracyConfig().model_dump(mode="json", exclude_none=True)
        assert "verbose" not in dumped
        assert "enable_cot" not in dumped

    @pytest.mark.parametrize("field_name", ["verbose", "enable_cot"])
    def test_explicit_boolean_without_benchmark_still_rejected(
        self, field_name: str
    ) -> None:
        with pytest.raises(ValueError, match="--accuracy-benchmark"):
            AccuracyConfig(**{field_name: True})

    def test_benchmark_set_round_trips_with_verbose(self) -> None:
        cfg = AccuracyConfig(benchmark="mmlu", verbose=True)
        assert _round_trip(cfg).verbose is True


class TestPhaseDeadFlagsRemoved:
    """The two provenance flags with zero readers are gone."""

    @pytest.mark.parametrize(
        "attr",
        [
            "_failed_request_threshold_explicitly_set",
            "_burst_phase_starts_explicitly_set",
        ],
    )  # fmt: skip
    def test_dead_flag_absent(self, attr: str) -> None:
        from aiperf.config.phases import BasePhaseConfig

        assert attr not in BasePhaseConfig.__private_attributes__

    @pytest.mark.parametrize(
        "attr",
        [
            "_trajectory_start_min_ratio_explicitly_set",
            "_trajectory_start_max_ratio_explicitly_set",
            "_system_idle_gap_cap_seconds_explicitly_set",
        ],
    )  # fmt: skip
    def test_live_flag_retained(self, attr: str) -> None:
        from aiperf.config.phases import BasePhaseConfig

        assert attr in BasePhaseConfig.__private_attributes__


@pytest.fixture
def mooncake_jsonl(tmp_path: Path) -> Path:
    """A mooncake-shaped JSONL that auto-detects as mooncake_trace."""
    path = tmp_path / "mc.jsonl"
    path.write_text(
        '{"input_length": 100, "output_length": 50, "hash_ids": [1, 2], "timestamp": 1000}\n'
    )
    return path


def _resolve(tmp_path: Path, path: Path, *, round_trip: bool, **fields: Any) -> None:
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
    if round_trip:
        run = _round_trip(run)
    DatasetResolver().resolve(run)


class TestBasetenOnlyWarningRoundTrip:
    """Default-valued baseten-only knobs must not warn after the round-trip."""

    @pytest.mark.parametrize("round_trip", [False, True], ids=["direct", "round-trip"])
    def test_no_warning_when_all_knobs_at_defaults(
        self,
        tmp_path: Path,
        mooncake_jsonl: Path,
        round_trip: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.resolver"):
            _resolve(tmp_path, mooncake_jsonl, round_trip=round_trip)
        assert not any("baseten_trace-only" in r.message for r in caplog.records)

    @pytest.mark.parametrize("round_trip", [False, True], ids=["direct", "round-trip"])
    def test_still_warns_for_non_default_knob(
        self,
        tmp_path: Path,
        mooncake_jsonl: Path,
        round_trip: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="aiperf.config.dataset.resolver"):
            _resolve(
                tmp_path, mooncake_jsonl, round_trip=round_trip, open_loop_replay=False
            )
        assert any(
            "baseten_trace-only" in r.message and "open_loop_replay" in r.message
            for r in caplog.records
        )
