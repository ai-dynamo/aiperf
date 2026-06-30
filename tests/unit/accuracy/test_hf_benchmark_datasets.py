# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network-gated smoke tests: verify every HuggingFace benchmark dataset is
accessible with the repo name, config, split, and fields the benchmark module
expects.

These tests catch dataset renames / restructures (e.g. ``gsm8k`` ->
``openai/gsm8k``) before they reach production. They are excluded from the
default test suite — run explicitly with ``pytest -m network``.

For dynamic-config benchmarks (MMLU, BigBench, LCB) one representative config
is used; the rest of the config space is assumed stable if the schema holds.

``streaming=True`` is used throughout so the test only fetches row metadata —
no full dataset download.

Gated datasets (GPQA) are skipped automatically when no HuggingFace token is
present; they run in CI environments that set ``HF_TOKEN``.
"""

from __future__ import annotations

import pytest
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from pytest import param


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset,config,split,required_fields,trust_remote_code",
    [
        param("openai/gsm8k", "main", "test", ["question", "answer"], False, id="gsm8k"),
        param("HuggingFaceH4/MATH-500", None, "test", ["problem", "solution", "subject"], False, id="math500"),
        param("HuggingFaceH4/aime_2024", None, "train", ["problem", "answer"], False, id="aime24"),
        param("yentinglin/aime_2025", None, "train", ["problem", "answer"], False, id="aime25"),
        param("Maxwell-Jia/AIME_2024", None, "train", ["Problem", "Answer"], False, id="aime"),
        param("lighteval/mmlu", "abstract_algebra", "test", ["question", "choices", "answer"], False, id="mmlu"),
        param("Rowan/hellaswag", None, "train", ["activity_label", "label"], False, id="hellaswag"),
        param("lukaemon/bbh", "boolean_expressions", "test", ["input", "target"], False, id="bigbench"),
        param("Idavidrein/gpqa", "gpqa_diamond", "train", ["Question", "Correct Answer"], False, id="gpqa_diamond"),
        param(
            "livecodebench/code_generation_lite",
            "v4_v5",
            "test",
            ["question_id", "question_content"],
            True,
            id="lcb",
        ),
    ],
)  # fmt: skip
def test_hf_benchmark_dataset_is_accessible(
    dataset: str,
    config: str | None,
    split: str,
    required_fields: list[str],
    trust_remote_code: bool,
) -> None:
    """Dataset loads and first row contains all fields the benchmark expects."""
    args = (dataset,) + ((config,) if config is not None else ())
    try:
        ds = load_dataset(
            *args, split=split, streaming=True, trust_remote_code=trust_remote_code
        )
    except DatasetNotFoundError as e:
        if "gated dataset" in str(e):
            pytest.skip(f"{dataset!r} is gated — set HF_TOKEN to run this test")
        raise
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            # datasets>=4 dropped support for repo-level loading scripts; LCB still uses one.
            # TODO: fix LCB benchmark to load from the Parquet export instead.
            pytest.skip(
                f"{dataset!r} uses a loading script not supported by this datasets version: {e}"
            )
        raise
    row = next(iter(ds))
    missing = [f for f in required_fields if f not in row]
    assert not missing, (
        f"{dataset!r} (config={config!r}, split={split!r}) is missing fields: {missing}. "
        f"Available: {list(row.keys())}"
    )
