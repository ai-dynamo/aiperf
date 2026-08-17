# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``resolve_effective_root_seed`` precedence: explicit seed, ambient manager, fresh entropy."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common import random_generator as rng
from aiperf.dataset.graph.adapters.shared.content import resolve_effective_root_seed


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        param(1234, 1234, id="explicit-seed-wins-over-ambient"),
        param(None, 777, id="none-falls-back-to-ambient-manager-seed"),
    ],
)  # fmt: skip
def test_resolve_effective_root_seed_prefers_explicit_then_ambient(
    requested: int | None, expected: int
) -> None:
    """An explicit seed wins; otherwise the seeded ambient manager's seed is reused."""
    rng.reset()
    rng.init(777)
    assert resolve_effective_root_seed(requested) == expected


@pytest.mark.parametrize(
    "init_manager",
    [
        param(False, id="no-ambient-manager"),
        param(True, id="ambient-manager-initialized-unseeded"),
    ],
)  # fmt: skip
def test_resolve_effective_root_seed_generates_per_run_seed_when_unseeded(
    init_manager: bool,
) -> None:
    """With no usable ambient seed, each resolution mints a fresh OS-entropy int."""
    # Distinct unseeded runs must differ, while the resolved int keeps ONE run
    # internally consistent.
    rng.reset()
    if init_manager:
        rng.init(None)

    first = resolve_effective_root_seed(None)
    second = resolve_effective_root_seed(None)

    assert isinstance(first, int)
    assert isinstance(second, int)
    assert first != second
