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


def test_artifacts_user_files_subtree_skipped_during_render() -> None:
    """artifacts.user_files content is rendered at run-time, not load-time.

    Pins the ``prefix + "."`` boundary in ``_path_is_skipped``: a sibling under
    ``artifacts`` whose name shares a prefix with ``user_files`` (e.g.
    ``artifacts.user_files_index``) MUST still render normally.
    """
    data = {
        "artifacts": {
            "sibling": "{{ defined }}",  # sibling under artifacts: must render
            "user_files": [
                {
                    "path": "x.json",
                    "format": "json",
                    "content": {"k": "{{ undefined }}"},
                }
            ],
        },
        "other": "{{ defined }}",
    }
    result = render_jinja2_templates(data, context={"defined": "yes"})
    # Top-level non-skipped renders:
    assert result["other"] == "yes"
    # Sibling under artifacts also renders (subtree skip is anchored, not substring):
    assert result["artifacts"]["sibling"] == "yes"
    # user_files content survived verbatim — no exception, no transformation:
    assert result["artifacts"]["user_files"][0]["content"]["k"] == "{{ undefined }}"
