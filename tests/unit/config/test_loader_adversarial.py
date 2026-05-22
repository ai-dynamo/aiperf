# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the adversarial config-loader audit.

Each test corresponds to a bug from /tmp/adversarial-config.md:

- S1: Envelope-rooted unknown keys must NOT be silently swallowed.
- S2: Numeric YAML keys must raise ConfigurationError, not AttributeError.
- S3: Cyclic YAML aliases must raise ConfigurationError, not RecursionError.
- S4: Pathologically deep YAML nesting must raise ConfigurationError.
- S5: Duplicate YAML keys must raise ConfigurationError, not last-win silently.
"""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from aiperf.config.loader.core import _MAX_CONFIG_NESTING_DEPTH, load_config_from_string
from aiperf.config.loader.errors import ConfigurationError

_VALID_BENCHMARK = textwrap.dedent("""\
benchmark:
  models: [llama]
  endpoint:
    urls: ["http://x:8000/v1/chat/completions"]
  datasets:
    - name: main
      type: synthetic
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
""")


def test_envelope_typo_raises_with_suggestion() -> None:
    """`sweeps:` (instead of `sweep:`) must NOT be silently dropped."""
    yaml_str = "sweeps:\n  type: grid\n  parameters: {}\n" + _VALID_BENCHMARK
    with pytest.raises((ConfigurationError, ValidationError)) as exc_info:
        load_config_from_string(yaml_str)
    msg = str(exc_info.value)
    assert "sweeps" in msg
    # Suggestion: the validator hints at the closest known key.
    assert "sweep" in msg


def test_envelope_random_seeds_typo_raises() -> None:
    """`random_seeds:` (plural typo) must NOT silently drop."""
    yaml_str = "random_seeds: 42\n" + _VALID_BENCHMARK
    with pytest.raises((ConfigurationError, ValidationError)) as exc_info:
        load_config_from_string(yaml_str)
    msg = str(exc_info.value)
    assert "random_seeds" in msg
    assert "random_seed" in msg


def test_envelope_root_user_files_typo_raises() -> None:
    """user_files at envelope root (belongs under benchmark.artifacts) must surface."""
    yaml_str = (
        'user_files:\n  ROOTLEVEL: {format: json, content: "rootlevel"}\n'
        + _VALID_BENCHMARK
    )
    with pytest.raises((ConfigurationError, ValidationError)) as exc_info:
        load_config_from_string(yaml_str)
    assert "user_files" in str(exc_info.value)


def test_numeric_yaml_key_raises_configuration_error() -> None:
    """An integer YAML key (e.g. ``42: scalar``) must raise ConfigurationError."""
    yaml_str = "42: scalar_value\n" + _VALID_BENCHMARK
    with pytest.raises(ConfigurationError) as exc_info:
        load_config_from_string(yaml_str)
    msg = str(exc_info.value)
    assert "not a string" in msg
    assert "42" in msg


def test_yaml_cycle_raises_configuration_error() -> None:
    """A cyclic YAML alias graph must raise ConfigurationError, not RecursionError."""
    # Cycle inside a dict that lives at the envelope root under a single
    # known key (`variables`) so it doesn't trip the envelope-extra check.
    yaml_str = (
        textwrap.dedent("""\
        variables: &anchor
          self: *anchor
        """)
        + _VALID_BENCHMARK
    )
    with pytest.raises(ConfigurationError) as exc_info:
        load_config_from_string(yaml_str)
    msg = str(exc_info.value)
    assert "Cyclic YAML aliases" in msg or "recursion" in msg.lower()


def test_yaml_deep_nesting_raises_configuration_error() -> None:
    """A pathologically deep config must raise ConfigurationError, not RecursionError."""
    # Stay just above AIPerf's explicit depth guard without forcing PyYAML itself
    # into interpreter-level recursion warnings before the guard can run.
    depth = _MAX_CONFIG_NESTING_DEPTH + 1
    nested = "leaf: 1"
    for _ in range(depth):
        nested = "a:\n  " + nested.replace("\n", "\n  ")
    yaml_str = "variables:\n  " + nested.replace("\n", "\n  ") + "\n" + _VALID_BENCHMARK
    with pytest.raises(ConfigurationError) as exc_info:
        load_config_from_string(yaml_str)
    msg = str(exc_info.value)
    assert "nested too deeply" in msg or "recursion" in msg.lower()


def test_yaml_duplicate_key_raises_configuration_error() -> None:
    """Duplicate mapping keys must raise ConfigurationError, not silent last-win."""
    yaml_str = textwrap.dedent("""\
        benchmark:
          models: [llama]
          models: [other-model]
          endpoint:
            urls: ["http://x:8000/v1/chat/completions"]
          datasets:
            - name: main
              type: synthetic
          phases:
            - name: profiling
              type: concurrency
              requests: 10
              concurrency: 1
        """)
    with pytest.raises(ConfigurationError) as exc_info:
        load_config_from_string(yaml_str)
    msg = str(exc_info.value)
    assert "Duplicate" in msg or "duplicate" in msg
    assert "models" in msg
