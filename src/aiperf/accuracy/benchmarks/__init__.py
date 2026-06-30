# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy benchmark implementations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HFSmokeSpec:
    """Declares what the network smoke test needs to verify a benchmark's HF dataset.

    Add one ``HF_SMOKE_SPEC`` module-level constant to every benchmark that
    loads from HuggingFace. The test in
    ``tests/unit/accuracy/test_hf_benchmark_datasets.py`` auto-discovers all
    specs so new benchmarks are covered automatically.
    """

    dataset: str
    split: str
    required_fields: list[str]
    config: str | None = None
    trust_remote_code: bool = False
