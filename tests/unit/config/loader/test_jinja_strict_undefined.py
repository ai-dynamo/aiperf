# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict mode: undefined jinja2 variables raise ConfigurationError naming the variable."""

from __future__ import annotations

import pytest

from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.jinja import render_jinja2_templates


def test_undefined_variable_raises_configuration_error() -> None:
    data = {"foo": "{{ undefined_var }}"}
    with pytest.raises(ConfigurationError) as exc_info:
        render_jinja2_templates(data, context={})
    assert "undefined_var" in str(exc_info.value.message)


def test_defined_variable_renders_normally() -> None:
    data = {"foo": "{{ defined }}"}
    result = render_jinja2_templates(data, context={"defined": 42})
    assert result == {"foo": 42}
