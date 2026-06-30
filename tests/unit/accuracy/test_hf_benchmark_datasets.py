# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network-gated smoke tests: verify every HuggingFace benchmark dataset is
accessible with the repo name, config, split, and fields the benchmark module
expects.

These tests catch dataset renames / restructures (e.g. ``gsm8k`` ->
``openai/gsm8k``) before they reach production. They are excluded from the
default test suite — run explicitly with ``pytest -m network``.

``streaming=True`` is used throughout so the test only fetches row metadata —
no full dataset download.

Gated datasets (e.g. GPQA) skip automatically when no HuggingFace token is
present; they run in CI environments that provide ``HF_TOKEN``.

Adding a new benchmark
----------------------
1. Add a ``param(...)`` entry to ``_BENCHMARK_DATASETS`` below.
2. ``test_all_hf_benchmarks_are_covered_by_network_tests`` will fail if you add a benchmark
   module with a ``DATASET_NAME`` constant but forget step 1 — the test tells
   you exactly which module is missing.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from pytest import param

import aiperf.accuracy.benchmarks as _benchmarks_pkg

_BENCHMARK_DATASETS = [
    param("openai/gsm8k", "main", "test", ["question", "answer"], False, id="gsm8k"),
    param(
        "HuggingFaceH4/MATH-500",
        None,
        "test",
        ["problem", "solution", "subject"],
        False,
        id="math_500",
    ),
    param(
        "HuggingFaceH4/aime_2024",
        None,
        "train",
        ["problem", "answer"],
        False,
        id="aime24",
    ),
    param(
        "yentinglin/aime_2025", None, "train", ["problem", "answer"], False, id="aime25"
    ),
    param(
        "Maxwell-Jia/AIME_2024", None, "train", ["Problem", "Answer"], False, id="aime"
    ),
    param(
        "lighteval/mmlu",
        "abstract_algebra",
        "test",
        ["question", "choices", "answer"],
        False,
        id="mmlu",
    ),
    param(
        "Rowan/hellaswag",
        None,
        "train",
        ["activity_label", "label"],
        False,
        id="hellaswag",
    ),
    param(
        "lukaemon/bbh",
        "boolean_expressions",
        "test",
        ["input", "target"],
        False,
        id="bigbench",
    ),
    param(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        "train",
        ["Question", "Correct Answer"],
        False,
        id="gpqa_diamond",
    ),
    param(
        "livecodebench/code_generation_lite",
        "v4_v5",
        "test",
        ["question_id", "question_content"],
        True,
        id="lcb_codegeneration",
    ),
]

# Module names of all benchmarks covered above (derived from param ids).
_COVERED_MODULE_NAMES = {p.id for p in _BENCHMARK_DATASETS}


def test_all_hf_benchmarks_are_covered_by_network_tests() -> None:
    """Fail if a benchmark module defines DATASET_NAME but has no network test entry.

    When adding a new HuggingFace-backed benchmark, add a param() to
    ``_BENCHMARK_DATASETS`` above with the same id as the module filename
    (without .py). This test will fail loudly until you do.
    """
    pkg_path = Path(_benchmarks_pkg.__file__).parent
    missing = []
    for info in pkgutil.iter_modules([str(pkg_path)]):
        mod = importlib.import_module(f"aiperf.accuracy.benchmarks.{info.name}")
        if hasattr(mod, "DATASET_NAME") and info.name not in _COVERED_MODULE_NAMES:
            missing.append(info.name)
    assert not missing, (
        f"Benchmark module(s) define DATASET_NAME but have no smoke test entry: {missing}. "
        f"Add a param() to _BENCHMARK_DATASETS in {__file__} with id='{missing[0]}'."
    )


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset,config,split,required_fields,trust_remote_code",
    _BENCHMARK_DATASETS,
)
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
            *args,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
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
                f"{dataset!r} uses a loading script unsupported by datasets>=4: {e}"
            )
        raise
    row = next(iter(ds))
    missing = [f for f in required_fields if f not in row]
    assert not missing, (
        f"{dataset!r} (config={config!r}, split={split!r}) is missing fields: {missing}. "
        f"Available: {list(row.keys())}"
    )
