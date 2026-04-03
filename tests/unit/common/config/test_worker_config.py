# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for WorkersConfig validation."""

from __future__ import annotations

import pytest

from aiperf.common.config.worker_config import WorkersConfig


class TestDirectWorkerValidation:
    """Validate --workers-direct and --workers-max mutual exclusivity."""

    @pytest.mark.parametrize(
        ("direct", "max_workers", "should_pass"),
        [
            pytest.param(True, None, True, id="direct-only"),
            pytest.param(False, None, True, id="no-direct-only"),
            pytest.param(None, 4, True, id="max-only"),
            pytest.param(False, 4, True, id="no-direct-with-max"),
            pytest.param(None, None, True, id="neither"),
            pytest.param(True, 4, False, id="direct-with-max"),
            pytest.param(True, 1, False, id="direct-with-max-1"),
        ],
    )
    def test_direct_and_max_compatibility(
        self,
        direct: bool | None,
        max_workers: int | None,
        should_pass: bool,
    ) -> None:
        kwargs: dict = {}
        if direct is not None:
            kwargs["direct"] = direct
        if max_workers is not None:
            kwargs["max"] = max_workers

        if should_pass:
            config = WorkersConfig(**kwargs)
            assert config.direct == direct
            assert config.max == max_workers
        else:
            with pytest.raises(
                ValueError,
                match="--workers-direct and --workers-max cannot be used together",
            ):
                WorkersConfig(**kwargs)

    def test_default_values(self) -> None:
        config = WorkersConfig()
        assert config.direct is None
        assert config.max is None
        assert config.min is None
