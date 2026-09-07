# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``--random-seed`` under ``--config`` must reach the envelope seed.

``build_dataset`` writes ``cli.random_seed`` onto the dataset block, but
``AIPerfConfig.random_seed`` -- the envelope-level seed that
``resolve_run_seed`` feeds into ``rng.init(...)`` for every child service
process -- is a distinct field that ``build_cli_overrides`` never wrote in
the YAML+CLI path. The dataset seed only drives ``SessionIDGenerator``;
every other ``rng.derive(...)`` consumer (synthetic prompt content,
image/audio/video generation, per-conversation turn count and delay) reads
off the envelope seed instead, so it stayed non-reproducible despite
``--random-seed`` appearing to take effect.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config


def _cli(**kwargs: object) -> CLIConfig:
    """Build a CLIConfig with only the supplied fields in model_fields_set."""
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


def _write_minimal_yaml(tmp_path: Path) -> Path:
    yaml_content = """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml_content)
    return cfg_path


def test_random_seed_flag_sets_the_envelope_seed(tmp_path: Path) -> None:
    """--random-seed under --config must reach AIPerfConfig.random_seed,
    not just the dataset's own seed field."""
    yaml_path = _write_minimal_yaml(tmp_path)
    cli = _cli(random_seed=42)
    cfg = resolve_config(cli, yaml_path)
    assert cfg.random_seed == 42


def test_random_seed_flag_still_reaches_the_dataset_seed(tmp_path: Path) -> None:
    """The existing dataset-seed routing must not regress while fixing the
    envelope gap -- SessionIDGenerator takes the dataset seed with
    precedence, so both fields must carry the flag's value."""
    yaml_path = _write_minimal_yaml(tmp_path)
    cli = _cli(random_seed=42)
    cfg = resolve_config(cli, yaml_path)
    assert cfg.benchmark.datasets[0].random_seed == 42


def test_random_seed_flag_absent_leaves_envelope_seed_unset(tmp_path: Path) -> None:
    """An unset --random-seed must not clobber the YAML with a materialized
    default; the envelope seed stays whatever the YAML (or its own default)
    supplies."""
    yaml_path = _write_minimal_yaml(tmp_path)
    cli = _cli()
    assert "random_seed" not in cli.model_fields_set
    cfg = resolve_config(cli, yaml_path)
    assert cfg.random_seed is None
