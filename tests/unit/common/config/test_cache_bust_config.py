# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``CacheBustConfig`` on ``PromptConfig``.

Coverage:

- Default ``target`` is ``CacheBustTarget.NONE``.
- ``PromptConfig`` constructs with the default empty ``CacheBustConfig``.
- Every ``CacheBustTarget`` value (other than NONE) round-trips through
  ``CacheBustConfig`` construction.
- Invalid string values are rejected.
- ``model_dump_json`` -> ``model_validate_json`` round-trip preserves the
  configured target.
- Nested ``cache_bust`` block under ``PromptConfig`` survives a JSON round-trip
  via the camelCase alias generator.
"""

from __future__ import annotations

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.enums import CacheBustTarget
from aiperf.config.dataset.content import CacheBustConfig, PromptConfig


class TestCacheBustConfigDefaults:
    def test_default_target_is_none(self) -> None:
        cfg = CacheBustConfig()
        assert cfg.target == CacheBustTarget.NONE

    def test_prompt_config_has_default_cache_bust_block(self) -> None:
        prompt = PromptConfig()
        assert isinstance(prompt.cache_bust, CacheBustConfig)
        assert prompt.cache_bust.target == CacheBustTarget.NONE


class TestCacheBustConfigValid:
    @pytest.mark.parametrize(
        "target",
        [
            CacheBustTarget.NONE,
            CacheBustTarget.SYSTEM_PREFIX,
            CacheBustTarget.SYSTEM_SUFFIX,
            CacheBustTarget.FIRST_TURN_PREFIX,
            CacheBustTarget.FIRST_TURN_SUFFIX,
        ],
    )
    def test_construct_from_enum_member(self, target: CacheBustTarget) -> None:
        cfg = CacheBustConfig(target=target)
        assert cfg.target == target

    @pytest.mark.parametrize(
        "target_str,expected",
        [
            ("none", CacheBustTarget.NONE),
            ("system_prefix", CacheBustTarget.SYSTEM_PREFIX),
            ("system_suffix", CacheBustTarget.SYSTEM_SUFFIX),
            ("first_turn_prefix", CacheBustTarget.FIRST_TURN_PREFIX),
            ("first_turn_suffix", CacheBustTarget.FIRST_TURN_SUFFIX),
        ],
    )
    def test_construct_from_string(
        self, target_str: str, expected: CacheBustTarget
    ) -> None:
        cfg = CacheBustConfig(target=target_str)
        assert cfg.target == expected


class TestCacheBustConfigInvalid:
    def test_unknown_target_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CacheBustConfig(target="not_a_real_target")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CacheBustConfig(target="none", banana=True)

    def test_none_value_rejected(self) -> None:
        # CacheBustTarget.NONE is a string member; ``None`` itself is not valid.
        with pytest.raises(ValidationError):
            CacheBustConfig(target=None)


class TestCacheBustConfigRoundTrip:
    @pytest.mark.parametrize(
        "target",
        [
            CacheBustTarget.NONE,
            CacheBustTarget.SYSTEM_PREFIX,
            CacheBustTarget.FIRST_TURN_SUFFIX,
        ],
    )
    def test_json_round_trip(self, target: CacheBustTarget) -> None:
        original = CacheBustConfig(target=target)
        restored = CacheBustConfig.model_validate_json(original.model_dump_json())
        assert restored == original
        assert restored.target == target

    def test_nested_under_prompt_config_round_trip(self) -> None:
        prompt = PromptConfig(
            cache_bust=CacheBustConfig(target=CacheBustTarget.FIRST_TURN_PREFIX),
        )
        restored = PromptConfig.model_validate_json(prompt.model_dump_json())
        assert restored.cache_bust.target == CacheBustTarget.FIRST_TURN_PREFIX

    def test_nested_camel_case_alias_round_trip(self) -> None:
        # BaseConfig uses camelCase aliases for K8s CRD shape; the nested
        # block must round-trip via ``cacheBust``.
        prompt = PromptConfig(
            cache_bust=CacheBustConfig(target=CacheBustTarget.SYSTEM_SUFFIX),
        )
        dumped = prompt.model_dump(by_alias=True)
        assert "cacheBust" in dumped
        # The nested enum dumps as its string value.
        assert dumped["cacheBust"]["target"] == CacheBustTarget.SYSTEM_SUFFIX

        # Re-parse via JSON (mimicking K8s CRD ingest).
        restored = PromptConfig.model_validate(orjson.loads(orjson.dumps(dumped)))
        assert restored.cache_bust.target == CacheBustTarget.SYSTEM_SUFFIX
