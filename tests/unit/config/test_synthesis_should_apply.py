# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared ``synthesis_should_apply`` gate.

Every "did the user configure trace synthesis?" site funnels through this one
helper, so drift here silently no-ops a lone multiplier. Tests use real
``SynthesisConfig`` objects (never MagicMock, which auto-creates attribute
paths and hides missing-field drift).
"""

import pytest
from pytest import param

from aiperf.config.dataset.trace import SynthesisConfig, synthesis_should_apply


class TestSynthesisShouldApply:
    """Gate returns True only when a transform multiplier differs from default."""

    def test_none_returns_false(self):
        assert synthesis_should_apply(None) is False

    def test_all_defaults_returns_false(self):
        assert synthesis_should_apply(SynthesisConfig()) is False

    @pytest.mark.parametrize(
        "overrides",
        [
            param({"speedup_ratio": 2.0}, id="speedup_ratio"),
            param({"prefix_len_multiplier": 2.0}, id="prefix_len_multiplier"),
            param({"prefix_root_multiplier": 2}, id="prefix_root_multiplier"),
            param({"prompt_len_multiplier": 2.0}, id="prompt_len_multiplier"),
            param({"output_len_multiplier": 2.0}, id="output_len_multiplier"),
        ],
    )  # fmt: skip
    def test_lone_transform_triggers(self, overrides):
        assert synthesis_should_apply(SynthesisConfig(**overrides)) is True

    @pytest.mark.parametrize(
        "overrides",
        [
            param({"max_isl": 4096}, id="max_isl"),
            param({"max_osl": 512}, id="max_osl"),
        ],
    )  # fmt: skip
    def test_filters_and_caps_do_not_trigger_apply_gate(self, overrides):
        """max_isl / max_osl are filters/caps, not transforms — apply gate stays off."""
        assert synthesis_should_apply(SynthesisConfig(**overrides)) is False
