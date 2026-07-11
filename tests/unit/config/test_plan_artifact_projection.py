# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact settings retained by Config-v2 benchmark plans."""

from aiperf.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan


def _config(artifacts: dict) -> AIPerfConfig:
    return AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["model"],
                "endpoint": {"urls": ["http://endpoint"]},
                "dataset": {"type": "synthetic"},
                "profiling": {
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 1,
                },
                "artifacts": artifacts,
            }
        }
    )


def test_plan_retains_custom_record_export_for_distribution_consumers() -> None:
    plan = build_benchmark_plan(
        _config(
            {
                "dir": "artifacts",
                "prefix": "search-samples",
                "records": ["jsonl"],
            }
        )
    )

    assert plan.export_level == "records"
    assert plan.export_jsonl_file == "search-samples.jsonl"


def test_plan_marks_disabled_records_as_summary_only() -> None:
    plan = build_benchmark_plan(
        _config({"dir": "artifacts", "records": False, "raw": False})
    )

    assert plan.export_level == "summary"
    assert plan.export_jsonl_file is None


def test_raw_level_still_retains_metrics_jsonl_path() -> None:
    plan = build_benchmark_plan(
        _config({"dir": "artifacts", "records": False, "raw": True})
    )

    assert plan.export_level == "raw"
    assert plan.export_jsonl_file == "profile_export.jsonl"
