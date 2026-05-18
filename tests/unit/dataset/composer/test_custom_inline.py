# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end composer test for inline FileDataset records."""

from __future__ import annotations

from aiperf.common.enums import DatasetFormat, DatasetType
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.composer.custom import CustomDatasetComposer


def _build_run_with_inline(records, format_=DatasetFormat.SINGLE_TURN):
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
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
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
    )


class TestCustomComposerInline:
    def test_inline_single_turn_creates_conversations(self):
        records = [
            {"text": "What is ML?"},
            {"text": "Explain GANs."},
            {"text": "Define AI."},
        ]
        run = _build_run_with_inline(records)
        composer = CustomDatasetComposer(run=run, tokenizer=None)
        conversations = composer.create_dataset()
        assert len(conversations) == 3

    def test_inline_random_pool_with_explicit_format(self):
        records = [
            {"text": "Q1", "type": "random_pool"},
            {"text": "Q2", "type": "random_pool"},
            {"text": "Q3", "type": "random_pool"},
        ]
        run = _build_run_with_inline(records, format_=DatasetFormat.RANDOM_POOL)
        composer = CustomDatasetComposer(run=run, tokenizer=None)
        conversations = composer.create_dataset()
        assert len(conversations) >= 1

    def test_inline_skips_file_existence_check(self):
        """Inline composer must not call check_file_exists - there's no path."""
        records = [{"text": "hello"}]
        run = _build_run_with_inline(records)
        composer = CustomDatasetComposer(run=run, tokenizer=None)
        # If check_file_exists were called, it would crash on a None/missing path.
        # Should reach loader creation cleanly.
        conversations = composer.create_dataset()
        assert len(conversations) == 1
