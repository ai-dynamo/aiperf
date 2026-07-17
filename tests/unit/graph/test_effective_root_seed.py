# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common import random_generator as rng
from aiperf.dataset.graph.adapters.shared.content import (
    resolve_effective_root_seed,
)


def test_resolve_effective_root_seed_explicit_seed_wins() -> None:
    rng.reset()
    rng.init(777)
    assert resolve_effective_root_seed(1234) == 1234


def test_resolve_effective_root_seed_none_uses_ambient_manager_seed() -> None:
    rng.reset()
    rng.init(777)
    assert resolve_effective_root_seed(None) == 777


def test_resolve_effective_root_seed_no_manager_generates_per_run_seed() -> None:
    rng.reset()
    first = resolve_effective_root_seed(None)
    second = resolve_effective_root_seed(None)
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert first != second


def test_resolve_effective_root_seed_unseeded_manager_generates_per_run_seed() -> None:
    # Ambient manager exists but was initialized with None (unseeded run):
    # each resolution generates a fresh OS-entropy seed, so distinct unseeded
    # runs differ while the resolved int keeps ONE run internally consistent.
    rng.reset()
    rng.init(None)
    first = resolve_effective_root_seed(None)
    second = resolve_effective_root_seed(None)
    assert first != second
