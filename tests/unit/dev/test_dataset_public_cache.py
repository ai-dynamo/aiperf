# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dev.benchmarks.dataset_format_catalog import profile_for
from dev.benchmarks.dataset_public_cache import effective_streaming


def test_snapshot_cache_pins_load_non_streaming() -> None:
    profile = profile_for("exgentic")
    assert profile is not None
    entry = {"streaming": True}
    assert effective_streaming(profile, entry) is False


def test_non_snapshot_pins_preserve_catalog_streaming_flag() -> None:
    profile = profile_for("mmvu")
    assert profile is not None
    assert effective_streaming(profile, {"streaming": False}) is False
    assert effective_streaming(profile, {"streaming": True}) is True
