# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: batch-size CLI flags must override YAML-supplied random_pool datasets.

In the YAML+CLI path, ``_apply_input_overrides`` only routed ``headers`` and
``extra_inputs``; all other INPUT_FIELDS members were silently discarded.  For
the four batch-size fields this was newly reachable after this PR added them to
``FileDataset``.  ``_apply_random_pool_batch_size_overrides`` closes the gap.

All tests drive ``resolve_config`` with a real YAML file, not ``convert_cli_to_aiperf``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.dataset.config import FileDataset
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config


def _write_random_pool_yaml(
    tmp_path: Path, pool_path: Path, **extra_dataset_fields: object
) -> Path:
    """Write a minimal YAML config with a random_pool file dataset.

    Extra keyword arguments are serialised as YAML dataset fields (one per line,
    two-space indent) so individual tests can pre-set batch sizes in the YAML.
    """
    extra_lines = "".join(
        f"    {key}: {value}\n" for key, value in extra_dataset_fields.items()
    )
    yaml_content = f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: random_pool
    path: {pool_path}
{extra_lines}\
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    cfg_path = tmp_path / "random_pool.yaml"
    cfg_path.write_text(yaml_content)
    return cfg_path


def _cli(**kwargs: object) -> CLIConfig:
    """Build a CLIConfig with only the supplied fields in model_fields_set."""
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


def _dataset(cfg: AIPerfConfig) -> FileDataset:
    return cfg.benchmark.datasets[0]


# ---------------------------------------------------------------------------
# Core override semantics
# ---------------------------------------------------------------------------


def test_prompt_batch_size_cli_overrides_yaml_silence(tmp_path: Path) -> None:
    """CLI --prompt-batch-size wins when YAML has no batch-size set."""
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_path = _write_random_pool_yaml(tmp_path, pool)
    cli = _cli(prompt_batch_size=7)
    cfg = resolve_config(cli, yaml_path)
    assert _dataset(cfg).prompt_batch_size == 7


def test_prompt_batch_size_cli_wins_over_yaml(tmp_path: Path) -> None:
    """CLI --prompt-batch-size overrides a value already set in the YAML."""
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_path = _write_random_pool_yaml(tmp_path, pool, prompt_batch_size=10)
    cli = _cli(prompt_batch_size=3)
    cfg = resolve_config(cli, yaml_path)
    assert _dataset(cfg).prompt_batch_size == 3


def test_yaml_batch_size_survives_when_cli_not_set(tmp_path: Path) -> None:
    """YAML batch-size must survive when the CLI flag is absent.

    This is the regression guard for the 'gate on model_fields_set' requirement:
    if the check were truthiness-based the YAML value would be clobbered on
    every run that omits the flag.
    """
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_path = _write_random_pool_yaml(tmp_path, pool, prompt_batch_size=10)
    cli = CLIConfig()  # no batch-size flags set at all
    cfg = resolve_config(cli, yaml_path)
    assert _dataset(cfg).prompt_batch_size == 10


def test_image_batch_size_zero_resolves_to_zero(tmp_path: Path) -> None:
    """image_batch_size=0 (disable images) must not be treated as unset or clamped to 1."""
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_path = _write_random_pool_yaml(tmp_path, pool)
    cli = _cli(image_batch_size=0)
    cfg = resolve_config(cli, yaml_path)
    assert _dataset(cfg).image_batch_size == 0


def test_all_four_modalities_distinct_values(tmp_path: Path) -> None:
    """All four batch-size fields are written independently; a field-order swap fails."""
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_path = _write_random_pool_yaml(tmp_path, pool)
    cli = _cli(
        prompt_batch_size=2, image_batch_size=3, audio_batch_size=5, video_batch_size=7
    )
    cfg = resolve_config(cli, yaml_path)
    ds = _dataset(cfg)
    assert ds.prompt_batch_size == 2
    assert ds.image_batch_size == 3
    assert ds.audio_batch_size == 5
    assert ds.video_batch_size == 7


def test_camel_case_yaml_field_overridden_by_cli_flag_does_not_crash(
    tmp_path: Path,
) -> None:
    """CLI --prompt-batch-size against a YAML that already spells the field in
    its camelCase alias (promptBatchSize -- the shipped template idiom, see
    e.g. templates/embeddings.yaml's batchSize) must override cleanly, not
    crash with extra_forbidden.

    Regression test: the override previously always wrote the snake_case key
    (dataset["prompt_batch_size"] = ...) without removing a pre-existing
    camelCase key. FileDataset has alias_generator=to_camel, extra="forbid",
    so both keys ending up present made Pydantic bind from the alias and
    reject the snake_case key as extra.
    """
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_content = f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: random_pool
    path: {pool}
    promptBatchSize: 10
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    yaml_path = tmp_path / "camel.yaml"
    yaml_path.write_text(yaml_content)
    cli = _cli(prompt_batch_size=3)
    cfg = resolve_config(cli, yaml_path)
    assert _dataset(cfg).prompt_batch_size == 3


# ---------------------------------------------------------------------------
# Non-random_pool dataset: friendly error, not raw Pydantic trace
# ---------------------------------------------------------------------------


def test_batch_size_on_non_random_pool_yaml_raises_friendly_error(
    tmp_path: Path,
) -> None:
    """Batch-size CLI flag on a mooncake_trace YAML dataset must raise a clear ValueError.

    The FileDataset model validator would fire anyway, but its message does not
    name the CLI flag.  The override helper must intercept first and raise a
    message that tells the user what to do.
    """
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_content = f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: mooncake_trace
    path: {pool}
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    yaml_path = tmp_path / "trace.yaml"
    yaml_path.write_text(yaml_content)
    cli = _cli(prompt_batch_size=4)
    with pytest.raises(ValueError, match="random_pool"):
        resolve_config(cli, yaml_path)


def test_batch_size_on_yaml_with_omitted_format_reports_effective_default(
    tmp_path: Path,
) -> None:
    """The rejection message must report the effective default format
    ('single_turn'), not a bare 'None', when the YAML omits `format:` entirely.

    Regression test: dataset.get("format") reads the raw pre-validation YAML
    dict, so an omitted format key read back as None even though
    FileDataset.format defaults to DatasetFormat.SINGLE_TURN -- "got format:
    None" reads like a parse failure rather than "you didn't set format, so
    it's using the default".
    """
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_content = f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    path: {pool}
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    yaml_path = tmp_path / "no_format.yaml"
    yaml_path.write_text(yaml_content)
    cli = _cli(prompt_batch_size=4)
    with pytest.raises(ValueError, match=re.escape("got format: single_turn")):
        resolve_config(cli, yaml_path)


# ---------------------------------------------------------------------------
# Synthetic / public datasets: batch-size flags are legitimate elsewhere and
# must not be rejected by the random_pool-specific override helper.
# ---------------------------------------------------------------------------


def _write_synthetic_yaml(tmp_path: Path, **extra_prompts_fields: object) -> Path:
    """Write a minimal YAML config with a synthetic dataset."""
    extra_lines = "".join(
        f"      {key}: {value}\n" for key, value in extra_prompts_fields.items()
    )
    yaml_content = f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    prompts:
{extra_lines}\
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    cfg_path = tmp_path / "synthetic.yaml"
    cfg_path.write_text(yaml_content)
    return cfg_path


@pytest.mark.parametrize(
    "batch_kwarg",
    [
        pytest.param({"prompt_batch_size": 4}, id="text"),
        pytest.param({"image_batch_size": 2}, id="image"),
        pytest.param({"audio_batch_size": 2}, id="audio"),
        pytest.param({"video_batch_size": 2}, id="video"),
    ],
)
def test_batch_size_flag_on_synthetic_yaml_does_not_raise(
    tmp_path: Path, batch_kwarg: dict
) -> None:
    """Batch-size CLI flags must not raise on a synthetic YAML dataset.

    Regression test: synthetic datasets have no ``format`` field at all, so
    ``dataset.get("format")`` returned ``None`` and the ``!= RANDOM_POOL`` check
    incorrectly raised for perfectly legitimate synthetic/embeddings batch-size
    usage.
    """
    yaml_path = _write_synthetic_yaml(tmp_path, batch_size=1)
    cli = _cli(**batch_kwarg)
    cfg = resolve_config(cli, yaml_path)
    assert cfg.benchmark.datasets[0].type == "synthetic"


def test_prompt_batch_size_flag_on_synthetic_yaml_is_silently_dropped(
    tmp_path: Path,
) -> None:
    """Pins the known (pre-existing, out-of-scope-to-fix-here) gap: nothing in
    the YAML+CLI path routes --prompt-batch-size onto a synthetic dataset's
    prompts.batch_size, unlike the CLI-only path (no --config) where the same
    flag applies correctly. The YAML value survives untouched; the explicit
    CLI flag is silently ignored, inverting normal CLI-overrides-YAML
    precedence. If this starts failing, the gap has been closed -- update
    this test to assert the override actually took effect.
    """
    yaml_path = _write_synthetic_yaml(tmp_path, batch_size=1)
    cli = _cli(prompt_batch_size=4)
    cfg = resolve_config(cli, yaml_path)
    assert cfg.benchmark.datasets[0].prompts.batch_size == 1


def test_image_batch_size_flag_on_synthetic_yaml_is_silently_dropped(
    tmp_path: Path,
) -> None:
    """Companion to the text case: --image-batch-size against a synthetic YAML
    dataset with no `images:` block doesn't even create one -- the flag has
    zero effect end to end, not just a value mismatch.
    """
    yaml_path = _write_synthetic_yaml(tmp_path, batch_size=1)
    cli = _cli(image_batch_size=2)
    cfg = resolve_config(cli, yaml_path)
    assert cfg.benchmark.datasets[0].images is None


def test_batch_size_flag_applies_without_cli_custom_dataset_type(
    tmp_path: Path,
) -> None:
    """Batch-size flags must resolve via YAML `format: random_pool` alone, with
    no `--custom-dataset-type` CLI flag set at all.

    Regression test for a doc claim ("only valid with --custom-dataset-type
    random_pool") that was false: the resolver keys off `dataset["format"]`
    in the merged YAML dict, never `cli.custom_dataset_type` -- the YAML
    `format:` key is the same knob, reachable without touching the CLI flag.
    """
    pool = tmp_path / "pool.jsonl"
    pool.touch()
    yaml_path = _write_random_pool_yaml(tmp_path, pool)
    cli = _cli(prompt_batch_size=4)
    assert "custom_dataset_type" not in cli.model_fields_set
    cfg = resolve_config(cli, yaml_path)
    assert _dataset(cfg).prompt_batch_size == 4
