# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``RecipeUserConfigView`` Protocol surface.

The Protocol declares the read-only structural shape that recipes are
allowed to depend on from ``UserConfig``. We verify both that a real v1
``UserConfig`` is structurally compatible and that
``SearchRecipeContext.user_config`` actually carries the documented attribute
chain at runtime.
"""

from __future__ import annotations

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.search_recipes._base import SearchRecipeContext
from aiperf.search_recipes._user_config_view import (
    RecipeUserConfigView,
    _EndpointView,
)


def _make_user_config(streaming: bool) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(model_names=["m"], streaming=streaming),
    )


@pytest.mark.parametrize("streaming", [True, False])
def test_user_config_is_structurally_compatible_with_view(
    streaming: bool,
) -> None:
    uc = _make_user_config(streaming=streaming)
    assert isinstance(uc, RecipeUserConfigView)
    assert isinstance(uc.endpoint, _EndpointView)


def test_search_recipe_context_accepts_real_user_config() -> None:
    uc = _make_user_config(streaming=True)
    ctx = SearchRecipeContext(user_config=uc)
    assert ctx.user_config.benchmark.endpoint is not None
    assert ctx.user_config.benchmark.endpoint.streaming is True


def test_search_recipe_context_user_config_attribute_chain() -> None:
    uc = _make_user_config(streaming=False)
    ctx = SearchRecipeContext(user_config=uc, sla_targets={"ttft_sla_ms": 200.0})
    # Recipes read this dotted path today; if it ever stops working, recipes
    # break — pin the contract here.
    assert ctx.user_config.benchmark.endpoint.streaming is False
    assert ctx.sla_targets["ttft_sla_ms"] == 200.0
