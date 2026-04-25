# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variables block must persist on the resolved config so run-time renderers can use it."""

from aiperf.config import AIPerfConfig
from aiperf.config.loader import load_config_from_string
from aiperf.config.loader.jinja import expand_config_dict

_BASE_YAML = """
models:
  - test/model
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
datasets:
  default:
    type: synthetic
    entries: 100
    prompts:
      isl: 128
      osl: 64
phases:
  default:
    type: concurrency
    requests: 10
    concurrency: 1
"""


_BASE_DICT: dict = {
    "models": ["test/model"],
    "endpoint": {"type": "chat", "urls": ["http://localhost:8000"]},
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


def test_variables_block_persists_on_resolved_config():
    yaml_str = (
        """
variables:
  isl: 1024
  osl: 512
"""
        + _BASE_YAML
    )
    config = load_config_from_string(yaml_str)
    assert config.variables == {"isl": 1024, "osl": 512}


def test_variables_default_empty_when_not_declared():
    config = load_config_from_string(_BASE_YAML)
    assert config.variables == {}


def test_variables_block_persists_through_expand_config_dict():
    """K8s/CRD ingestion path: expand_config_dict must keep variables intact.

    Mirrors test_variables_block_persists_on_resolved_config but exercises the
    operator-side dict pipeline used in spec_converter.py rather than the YAML
    string pipeline used by the CLI.
    """
    data = {"variables": {"isl": 1024, "osl": 512}, **_BASE_DICT}
    expanded = expand_config_dict(data)

    assert "variables" in expanded
    assert expanded["variables"] == {"isl": 1024, "osl": 512}

    config = AIPerfConfig.model_validate(expanded)
    assert config.variables == {"isl": 1024, "osl": 512}
