# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``deepseek_v32`` config-alias compatibility shim.

DeepSeek-V3.2-Exp ships ``model_type: "deepseek_v32"`` with no ``auto_map``.
On transformers releases without native support, the ``AutoConfig`` lookup that
``AutoTokenizer`` performs internally aborts tokenizer loading. The shim
registers a ``DeepseekV3Config`` alias so loading proceeds; once transformers
ships native support it must stay a no-op.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import (
    _DEEPSEEK_V32_MODEL_TYPE,
    _ensure_deepseek_v32_config_registered,
)


class _FakeDeepseekV3Config:
    """Stand-in for ``transformers.DeepseekV3Config`` that supports subclassing."""

    model_type = "deepseek_v3"


@pytest.fixture
def isolated_transformers_registry() -> Iterator[None]:
    """Snapshot and restore the real transformers config registry.

    Tests that mutate the global ``CONFIG_MAPPING`` (registering or removing
    ``deepseek_v32``) must not leak that state into other tests, since the
    registry is process-global and unaffected by the singleton-reset fixtures.
    """
    from transformers.models.auto.configuration_auto import (
        CONFIG_MAPPING,
        CONFIG_MAPPING_NAMES,
    )

    saved_names = dict(CONFIG_MAPPING_NAMES)
    saved_extra = dict(CONFIG_MAPPING._extra_content)
    try:
        yield
    finally:
        CONFIG_MAPPING_NAMES.clear()
        CONFIG_MAPPING_NAMES.update(saved_names)
        CONFIG_MAPPING._extra_content.clear()
        CONFIG_MAPPING._extra_content.update(saved_extra)


class TestEnsureDeepseekV32ConfigRegistered:
    def test_registers_alias_when_model_type_absent(self) -> None:
        auto_config = MagicMock()
        with (
            patch("transformers.AutoConfig", auto_config),
            patch("transformers.DeepseekV3Config", _FakeDeepseekV3Config),
            patch(
                "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
                {},  # deepseek_v32 not present
            ),
        ):
            _ensure_deepseek_v32_config_registered()

        auto_config.register.assert_called_once()
        args, kwargs = auto_config.register.call_args
        assert args[0] == _DEEPSEEK_V32_MODEL_TYPE
        registered_cls = args[1]
        assert issubclass(registered_cls, _FakeDeepseekV3Config)
        assert registered_cls.model_type == _DEEPSEEK_V32_MODEL_TYPE
        assert kwargs.get("exist_ok") is True

    def test_no_op_when_model_type_already_present(self) -> None:
        auto_config = MagicMock()
        with (
            patch("transformers.AutoConfig", auto_config),
            patch("transformers.DeepseekV3Config", _FakeDeepseekV3Config),
            patch(
                "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
                {_DEEPSEEK_V32_MODEL_TYPE: object()},  # native support present
            ),
        ):
            _ensure_deepseek_v32_config_registered()

        auto_config.register.assert_not_called()

    def test_swallows_errors_and_does_not_raise(self) -> None:
        # Simulate an old/renamed transformers where DeepseekV3Config is absent:
        # the shim must degrade silently so loading reaches its normal error path.
        with (
            patch(
                "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
                {},
            ),
            patch("transformers.AutoConfig", MagicMock()),
            patch(
                "transformers.DeepseekV3Config",
                new=None,
                create=True,
            ),
        ):
            # DeepseekV3Config = None -> subclassing raises TypeError,
            # which the shim must swallow.
            _ensure_deepseek_v32_config_registered()

    def test_real_transformers_round_trip(
        self, isolated_transformers_registry: None
    ) -> None:
        # End-to-end against the installed transformers: force the
        # "no native support" state, then verify the shim makes AutoConfig
        # resolve deepseek_v32 to a DeepseekV3Config subclass.
        from transformers import AutoConfig, DeepseekV3Config
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )

        CONFIG_MAPPING._extra_content.pop(_DEEPSEEK_V32_MODEL_TYPE, None)
        CONFIG_MAPPING_NAMES.pop(_DEEPSEEK_V32_MODEL_TYPE, None)
        assert _DEEPSEEK_V32_MODEL_TYPE not in CONFIG_MAPPING

        with pytest.raises(ValueError):
            AutoConfig.for_model(_DEEPSEEK_V32_MODEL_TYPE)

        _ensure_deepseek_v32_config_registered()

        assert _DEEPSEEK_V32_MODEL_TYPE in CONFIG_MAPPING
        config = AutoConfig.for_model(_DEEPSEEK_V32_MODEL_TYPE)
        assert isinstance(config, DeepseekV3Config)
        assert config.model_type == _DEEPSEEK_V32_MODEL_TYPE

        # Second call is idempotent (exist_ok=True), no raise.
        _ensure_deepseek_v32_config_registered()


class TestLoadFromHubRegistersDeepseekV32:
    def test_load_from_hub_invokes_registration_hook(self) -> None:
        # The shim must run before AutoTokenizer.from_pretrained on every load
        # path. Drive the cache-warm branch (simplest: no alias resolution).
        sentinel = object()
        with (
            patch(
                "aiperf.common.tokenizer._ensure_deepseek_v32_config_registered"
            ) as ensure_mock,
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch("aiperf.common.tokenizer._is_hf_cached", return_value=True),
            patch(
                "aiperf.common.tokenizer.Tokenizer._build_with_kwargs",
                return_value=sentinel,
            ),
            patch("transformers.AutoTokenizer"),
        ):
            from aiperf.common.tokenizer import Tokenizer

            result = Tokenizer._load_from_hub(
                "deepseek-ai/DeepSeek-V3.2-Exp",
                trust_remote_code=False,
                revision="main",
                resolve_alias=True,
            )

        assert result is sentinel
        ensure_mock.assert_called_once()
