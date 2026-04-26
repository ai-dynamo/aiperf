# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 DTO contract: NO validators on UserConfig or ServiceConfig.

The v1 package is a CLI-only input layer. AIPerfConfig is the single validation
gate. Adding a `@field_validator` or `@model_validator` to a v1 class is a
contract violation — fix it by moving the validation into AIPerfConfig.
"""

import inspect

from aiperf.config.v1 import ServiceConfig, UserConfig


def test_user_config_has_no_validators():
    decorators = [
        m
        for m in inspect.getmembers(UserConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not decorators, f"UserConfig must have NO validators (found: {decorators})"


def test_service_config_has_no_validators():
    decorators = [
        m
        for m in inspect.getmembers(ServiceConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not decorators, (
        f"ServiceConfig must have NO validators (found: {decorators})"
    )


def test_user_config_imports_from_v1_package():
    from aiperf.config.v1.user_config import UserConfig as UC

    assert UC is UserConfig
