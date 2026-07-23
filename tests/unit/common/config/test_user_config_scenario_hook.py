# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v2 scenario hook + the underscore explicit-set flags it depends on."""

from __future__ import annotations

from aiperf.common.enums import CacheBustTarget
from aiperf.config.dataset.config import FileDataset, PublicDataset
from aiperf.config.dataset.content import CacheBustConfig


class TestExplicitlySetFlags:
    """The underscore flags ``apply_scenario`` defensively reads."""

    def test_cache_bust_target_explicit_flag_when_passed(self) -> None:
        cfg = CacheBustConfig(target=CacheBustTarget.SYSTEM_PREFIX)
        assert cfg._target_explicitly_set is True

    def test_cache_bust_target_explicit_flag_when_omitted(self) -> None:
        cfg = CacheBustConfig()
        assert cfg._target_explicitly_set is False
        assert cfg.target == CacheBustTarget.NONE

    def test_use_think_time_only_explicit_flag_when_passed_file_dataset(self) -> None:
        cfg = FileDataset(
            name="main", type="file", path="/fake/trace.jsonl", use_think_time_only=True
        )
        assert cfg._use_think_time_only_explicitly_set is True

    def test_use_think_time_only_explicit_flag_when_omitted_file_dataset(self) -> None:
        cfg = FileDataset(name="main", type="file", path="/fake/trace.jsonl")
        assert cfg._use_think_time_only_explicitly_set is False

    def test_use_think_time_only_explicit_flag_when_passed_public_dataset(self) -> None:
        cfg = PublicDataset(
            name="main", type="public", dataset="sharegpt", use_think_time_only=True
        )
        assert cfg._use_think_time_only_explicitly_set is True

    def test_use_think_time_only_explicit_flag_when_omitted_public_dataset(
        self,
    ) -> None:
        cfg = PublicDataset(name="main", type="public", dataset="sharegpt")
        assert cfg._use_think_time_only_explicitly_set is False
