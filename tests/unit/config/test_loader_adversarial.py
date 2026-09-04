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

import sys
import textwrap
from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.loader import core
from aiperf.config.loader.core import (
    _MAX_CONFIG_NESTING_DEPTH,
    load_config_dict_with_raw_envelope,
    load_config_from_mapping,
    load_config_from_string,
)
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


def test_self_referential_mapping_raises_configuration_error() -> None:
    """An in-memory cycle must raise ConfigurationError, not RecursionError."""
    data: dict[str, object] = {"benchmark": {}}
    data["benchmark"] = data

    with pytest.raises(ConfigurationError, match="Cyclic"):
        load_config_from_mapping(data)


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


def _spy_on_flat_shape_migration(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record whether ``_auto_migrate_flat_shape`` ran, preserving behavior."""
    calls: list[bool] = []
    original = core._auto_migrate_flat_shape

    def _spy(data: dict[str, object], file_path: object) -> None:
        calls.append(True)
        original(data, file_path)  # type: ignore[arg-type]

    monkeypatch.setattr(core, "_auto_migrate_flat_shape", _spy)
    return calls


def test_config_dict_deep_nesting_rejected_before_flat_shape_migration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--config`` path must run the depth guard before any mutation pass.

    ``load_config_dict_with_raw_envelope`` is the CLI-facing entry point, so it
    must mirror ``load_config_from_mapping``'s early ``_detect_cycles_or_depth``
    guard rather than relying on the later guard inside
    ``_expand_capture_pre_jinja``.
    """
    calls = _spy_on_flat_shape_migration(monkeypatch)
    depth = _MAX_CONFIG_NESTING_DEPTH + 1
    # Flat (pre-envelope) shape so `_auto_migrate_flat_shape` would have work
    # to do if it were reached first.
    config_file = tmp_path / "deep.yaml"
    config_file.write_text("models: " + "[" * depth + "1" + "]" * depth + "\n")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config_dict_with_raw_envelope(config_file)

    assert "nested too deeply" in str(exc_info.value)
    assert calls == [], "depth guard must run before flat-shape migration"


def test_config_dict_cycle_rejected_before_flat_shape_migration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cyclic mapping must be rejected before any mutation pass touches it.

    PyYAML's own constructor rejects cyclic anchors today, so the cyclic dict is
    injected at the parse boundary to pin the loader-side ordering invariant
    independently of the parser's behavior.
    """
    calls = _spy_on_flat_shape_migration(monkeypatch)
    cyclic: dict[str, object] = {"models": ["llama"]}
    cyclic["self"] = cyclic
    monkeypatch.setattr(
        core, "_parse_yaml_mapping", lambda _content, _file_path: cyclic
    )
    config_file = tmp_path / "cyclic.yaml"
    config_file.write_text("models: [llama]\n")

    with pytest.raises(ConfigurationError, match="Cyclic"):
        load_config_dict_with_raw_envelope(config_file)

    assert calls == [], "cycle guard must run before flat-shape migration"


@pytest.fixture
def pinned_recursion_limit() -> Iterator[None]:
    """Pin the interpreter recursion limit for the duration of one test.

    The deepcopy-ordering regression only manifests below a certain ceiling, and
    pytest plugins may have raised the ambient limit. Set and restore it here so
    the pin cannot leak into other tests sharing the same xdist worker.
    """
    original = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        yield
    finally:
        sys.setrecursionlimit(original)


def _nest(depth: int) -> dict[str, object]:
    """Build a mapping nested ``depth`` levels under a valid envelope key."""
    node: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        node = {"a": node}
    return {"variables": node}


@pytest.mark.parametrize(
    "depth",
    [
        param(500, id="depth-500"),
        param(900, id="depth-900"),
    ],
)  # fmt: skip
def test_mapping_deeper_than_deepcopy_ceiling_raises_configuration_error(
    depth: int, pinned_recursion_limit: None
) -> None:
    """The depth guard must run BEFORE ``load_config_from_mapping`` deep-copies.

    ``copy.deepcopy`` is itself recursive, so guarding a copy rather than the
    original lets the copy blow the stack first and surface a bare
    ``RecursionError`` — exactly the failure the guard exists to prevent.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        load_config_from_mapping(_nest(depth), file_path="deep.yaml")

    assert "nested too deeply" in str(exc_info.value)
    assert exc_info.value.file_path == "deep.yaml"


def test_cyclic_mapping_reports_file_path() -> None:
    """A cyclic in-memory mapping raises ConfigurationError with its file_path."""
    data: dict[str, object] = {"benchmark": {}}
    data["benchmark"] = data

    with pytest.raises(ConfigurationError) as exc_info:
        load_config_from_mapping(data, file_path="cyclic.yaml")

    assert "Cyclic" in str(exc_info.value)
    assert exc_info.value.file_path == "cyclic.yaml"


@pytest.mark.parametrize(
    "depth",
    [
        param(500, id="depth-500"),
        param(900, id="depth-900"),
    ],
)  # fmt: skip
def test_k8s_config_envelope_guards_before_its_own_deepcopy(
    depth: int, pinned_recursion_limit: None
) -> None:
    """``build_config_envelope`` must guard before its per-field deepcopy.

    It is the Kubernetes CR ingestion path's first touch of a raw spec
    (``spec_converter.build_config_envelope`` -> ``load_config_from_mapping``),
    and it deep-copies each projected field itself, so fixing the loader's
    ordering alone would still leave a ``RecursionError`` reachable from an
    ``AIPerfJob``/``AIPerfSweep`` spec.
    """
    from aiperf.kubernetes.spec_converter import build_config_envelope

    with pytest.raises(ConfigurationError, match="nested too deeply"):
        build_config_envelope(_nest(depth))
