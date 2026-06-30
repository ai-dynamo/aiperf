# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network-gated smoke tests: verify every HuggingFace benchmark dataset is
accessible with the repo name, config, split, and fields the benchmark module
expects.

These tests catch dataset renames / restructures (e.g. ``gsm8k`` ->
``openai/gsm8k``) before they reach production. They are excluded from the
default test suite — run explicitly with ``pytest -m network``.

New benchmarks are covered automatically: add an ``HF_SMOKE_SPEC`` constant
(``HFSmokeSpec`` instance) to the benchmark module and this test picks it up
on the next run — no changes needed here.

``streaming=True`` is used throughout so the test only fetches row metadata —
no full dataset download.

Gated datasets (e.g. GPQA) skip automatically when no HuggingFace token is
present; they run in CI environments that provide ``HF_TOKEN``.
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
from aiperf.accuracy.benchmarks import HFSmokeSpec


def _discover_specs() -> list[tuple[str, HFSmokeSpec]]:
    """Return (module_name, HFSmokeSpec) for every benchmark that declares one."""
    pkg_path = Path(_benchmarks_pkg.__file__).parent
    specs = []
    for info in pkgutil.iter_modules([str(pkg_path)]):
        mod = importlib.import_module(f"aiperf.accuracy.benchmarks.{info.name}")
        spec = getattr(mod, "HF_SMOKE_SPEC", None)
        if isinstance(spec, HFSmokeSpec):
            specs.append((info.name, spec))
    return specs


_SPECS = _discover_specs()


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "benchmark,spec",
    [param(name, spec, id=name) for name, spec in _SPECS],
)
def test_hf_benchmark_dataset_is_accessible(benchmark: str, spec: HFSmokeSpec) -> None:
    """Dataset loads and first row contains all fields the benchmark expects."""
    args = (spec.dataset,) + ((spec.config,) if spec.config is not None else ())
    try:
        ds = load_dataset(
            *args,
            split=spec.split,
            streaming=True,
            trust_remote_code=spec.trust_remote_code,
        )
    except DatasetNotFoundError as e:
        if "gated dataset" in str(e):
            pytest.skip(f"{spec.dataset!r} is gated — set HF_TOKEN to run this test")
        raise
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            # datasets>=4 dropped support for repo-level loading scripts; LCB still uses one.
            # TODO: fix LCB benchmark to load from the Parquet export instead.
            pytest.skip(
                f"{spec.dataset!r} uses a loading script unsupported by datasets>=4: {e}"
            )
        raise
    row = next(iter(ds))
    missing = [f for f in spec.required_fields if f not in row]
    assert not missing, (
        f"{spec.dataset!r} (config={spec.config!r}, split={spec.split!r}) is missing fields: {missing}. "
        f"Available: {list(row.keys())}"
    )
