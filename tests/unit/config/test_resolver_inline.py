# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.enums import DatasetFormat, DatasetType
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.plugin.enums import CustomDatasetType


def _build_run_inline(records, format_=DatasetFormat.SINGLE_TURN):
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {
                "urls": ["http://localhost:8000"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [
                {
                    "name": "default",
                    "type": DatasetType.FILE,
                    "format": format_,
                    "records": records,
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
            "runtime": {"ui": "simple"},
        }
    )
    return BenchmarkRun(benchmark_id="t", cfg=cfg, artifact_dir=cfg.artifacts.dir)


class TestResolverInline:
    def test_inline_skips_path_validation(self):
        records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        run = _build_run_inline(records)
        DatasetResolver().resolve(run)
        assert run.resolved.dataset_total_records.get("default") == 3

    def test_inline_multi_pool_total_count(self):
        records = {
            "queries": [
                {"text": "q1", "type": "random_pool"},
                {"text": "q2", "type": "random_pool"},
            ],
            "passages": [{"text": "p1", "type": "random_pool"}],
        }
        run = _build_run_inline(records, format_=DatasetFormat.RANDOM_POOL)
        DatasetResolver().resolve(run)
        assert run.resolved.dataset_total_records.get("default") == 3

    def test_inline_explicit_format_maps_type(self):
        records = [{"text": "a"}]
        run = _build_run_inline(records, format_=DatasetFormat.SINGLE_TURN)
        DatasetResolver().resolve(run)
        assert (
            run.resolved.dataset_types.get("default") == CustomDatasetType.SINGLE_TURN
        )

    def test_inline_has_timing_detected_from_first_record(self):
        records = [{"text": "a", "timestamp": 0}, {"text": "b", "timestamp": 100}]
        run = _build_run_inline(records)
        DatasetResolver().resolve(run)
        assert run.resolved.dataset_has_timing_data.get("default") is True

    def test_inline_no_timing_when_absent(self):
        records = [{"text": "a"}, {"text": "b"}]
        run = _build_run_inline(records)
        DatasetResolver().resolve(run)
        assert run.resolved.dataset_has_timing_data.get("default") is False
