# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseConfig smoke tests.

The legacy ``serialize_to_yaml`` / ``_preprocess_value`` /
``_is_a_nested_config`` / ``_should_add_field_to_template`` machinery on
BaseConfig was removed during the v2 refactor (see
``aiperf.config.base`` - it's now a tiny ``pydantic.BaseModel`` subclass
with camelCase alias support, nothing more). Template generation moved
to ``aiperf.config.templates`` / ``aiperf.config._cli_runner_templates``.

The few surviving BaseConfig assertions live as smoke tests below; the
older private-method tests have been removed.
"""

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.base import BaseConfig


class _NestedConfig(AIPerfBaseModel):
    field1: str
    field2: int


class _BaseTestConfig(BaseConfig):
    nested: _NestedConfig
    verbose: bool = False


def test_basetestconfig_round_trips_through_json() -> None:
    config = _BaseTestConfig(
        nested=_NestedConfig(field1="value1", field2=42),
        verbose=True,
    )
    restored = _BaseTestConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_baseconfig_camelcase_alias_population() -> None:
    """BaseConfig accepts camelCase keys (alias_generator) and snake_case names."""

    class Config(BaseConfig):
        my_field: int = 0

    assert Config.model_validate({"my_field": 5}).my_field == 5
    assert Config.model_validate({"myField": 5}).my_field == 5
