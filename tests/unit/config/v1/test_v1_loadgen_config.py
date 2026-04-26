# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 LoadGeneratorConfig DTO contract: validator-free CLI input layer.

The v1 package is the cyclopts-facing input layer. AIPerfConfig is the single
validation gate. Adding `@field_validator` or `@model_validator` here is a
contract violation - move the validation into AIPerfConfig instead.
"""

import inspect

from aiperf.config.v1._loadgen import LoadGeneratorConfig


def test_loadgen_config_carries_warmup_fields():
    cfg = LoadGeneratorConfig.model_validate(
        {
            "concurrency": 100,
            "request_count": 1000,
            "warmup_concurrency": 10,
            "warmup_request_count": 50,
        }
    )
    assert cfg.concurrency == 100
    assert cfg.request_count == 1000
    assert cfg.warmup_concurrency == 10
    assert cfg.warmup_request_count == 50


def test_loadgen_config_has_no_validators():
    bad = [
        m
        for m in inspect.getmembers(LoadGeneratorConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not bad, f"LoadGeneratorConfig must have NO validators (found: {bad})"
