# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""benchmark_id lifecycle: None in config dumps, stamped at BenchmarkRun construction.

The old ``default_factory=uuid4`` meant every ``exclude_defaults`` dump
(``aiperf kube generate`` manifests, sweep-child specs) pinned one "unique"
id that all cluster-sweep children then shared. The field now defaults to
None and ``BenchmarkRun`` stamps its own run id into
``cfg.artifacts.benchmark_id`` at construction.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


def _make_run(benchmark_id: str, cfg: BenchmarkConfig) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_id=benchmark_id,
        cfg=cfg,
        artifact_dir=Path(f"/tmp/artifacts/{benchmark_id}"),
    )


class TestArtifactsConfigDefault:
    def test_benchmark_id_defaults_to_none(self) -> None:
        assert ArtifactsConfig().benchmark_id is None

    def test_exclude_defaults_dump_carries_no_benchmark_id(self) -> None:
        # The `aiperf kube generate` manifest dump shape.
        dumped = ArtifactsConfig().model_dump(
            mode="json", by_alias=True, exclude_defaults=True
        )
        assert "benchmarkId" not in dumped
        assert "benchmark_id" not in dumped

    def test_user_set_benchmark_id_survives_exclude_defaults_dump(self) -> None:
        dumped = ArtifactsConfig(benchmark_id="pinned-id").model_dump(
            mode="json", by_alias=True, exclude_defaults=True
        )
        assert dumped["benchmarkId"] == "pinned-id"

    def test_benchmark_config_dump_carries_no_benchmark_id(self) -> None:
        cfg = BenchmarkConfig.model_validate(_MINIMAL_CONFIG_KWARGS)
        dumped = cfg.model_dump(mode="json", by_alias=True, exclude_defaults=True)
        assert "benchmarkId" not in dumped.get("artifacts", {})


class TestBenchmarkRunStamping:
    def test_run_stamps_its_id_into_cfg_artifacts(self) -> None:
        cfg = BenchmarkConfig.model_validate(_MINIMAL_CONFIG_KWARGS)
        run = _make_run("aiperf-bench-7f2a", cfg)
        assert run.cfg.artifacts.benchmark_id == "aiperf-bench-7f2a"
        assert run.cfg.benchmark_id == "aiperf-bench-7f2a"

    def test_two_runs_from_same_config_get_distinct_effective_ids(self) -> None:
        cfg = BenchmarkConfig.model_validate(_MINIMAL_CONFIG_KWARGS)
        run_a = _make_run("run-a", cfg)
        run_b = _make_run("run-b", cfg)
        assert run_a.cfg.artifacts.benchmark_id == "run-a"
        assert run_b.cfg.artifacts.benchmark_id == "run-b"

    def test_stamping_does_not_mutate_caller_config(self) -> None:
        cfg = BenchmarkConfig.model_validate(_MINIMAL_CONFIG_KWARGS)
        _make_run("run-a", cfg)
        assert cfg.artifacts.benchmark_id is None

    def test_user_pinned_id_wins_over_run_id(self) -> None:
        cfg = BenchmarkConfig.model_validate(
            {**_MINIMAL_CONFIG_KWARGS, "artifacts": {"benchmark_id": "pinned-id"}}
        )
        run = _make_run("run-a", cfg)
        assert run.cfg.artifacts.benchmark_id == "pinned-id"

    def test_stamped_id_survives_subprocess_json_round_trip(self) -> None:
        cfg = BenchmarkConfig.model_validate(_MINIMAL_CONFIG_KWARGS)
        run = _make_run("run-a", cfg)
        payload = orjson.dumps(run.model_dump(mode="json", exclude_none=True))
        restored = BenchmarkRun.model_validate(orjson.loads(payload))
        assert restored.cfg.artifacts.benchmark_id == "run-a"
