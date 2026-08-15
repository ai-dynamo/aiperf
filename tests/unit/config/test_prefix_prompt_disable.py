# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``--num-prefix-prompts 0`` / ``--prefix-prompt-length 0`` disable prefixes.

The CLI fields accept ``ge=0`` and document "Set to 0 to disable prefix
prompts", but the converter forwarded the zero into ``PrefixPromptConfig``
whose ``pool_size``/``length`` are ``ge=1``, so the run died with a validation
error instead of disabling the feature.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config


def _resolve(**kwargs: object):  # noqa: ANN202 - resolved AIPerfConfig
    seed = CLIConfig(model_names=["m"], url="http://localhost:8000", **kwargs)
    return resolve_config(CLIConfig(**seed.model_dump(exclude_unset=True)), None)


def _prefix_prompts(cfg):  # noqa: ANN001, ANN202
    return cfg.benchmark.datasets[0].prefix_prompts


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"prompt_prefix_pool_size": 0}, id="pool-size-zero"),
        param({"prompt_prefix_length": 0}, id="length-zero"),
        param(
            {"prompt_prefix_pool_size": 0, "prompt_prefix_length": 0},
            id="both-zero",
        ),
        param(
            {"prompt_prefix_pool_size": 4, "prompt_prefix_length": 0},
            id="nonzero-pool-zero-length",
        ),
        param(
            {"prompt_prefix_pool_size": 0, "prompt_prefix_length": 16},
            id="zero-pool-nonzero-length",
        ),
    ],
)  # fmt: skip
def test_build_prefix_prompts_zero_value_disables_prefix_prompts(
    kwargs: dict[str, int],
) -> None:
    """A zero pool size or length disables prefixes instead of raising."""
    prefix = _prefix_prompts(_resolve(**kwargs))
    assert prefix is None or (prefix.pool_size is None and prefix.length is None)


def test_build_prefix_prompts_nonzero_values_build_pool_config() -> None:
    """The normal (non-zero) path still builds the pool config unchanged."""
    prefix = _prefix_prompts(
        _resolve(prompt_prefix_pool_size=3, prompt_prefix_length=10)
    )
    assert prefix is not None
    assert prefix.pool_size == 3
    assert prefix.length == 10


def test_build_prefix_prompts_zero_pool_keeps_shared_system_length() -> None:
    """Disabling the pool leaves the two-part prefix fields intact."""
    prefix = _prefix_prompts(
        _resolve(prompt_prefix_pool_size=0, prompt_prefix_shared_system_length=32)
    )
    assert prefix is not None
    assert prefix.pool_size is None
    assert prefix.shared_system_length == 32
