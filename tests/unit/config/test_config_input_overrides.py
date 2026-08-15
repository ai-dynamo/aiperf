# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset-shaping CLI flags must override a YAML-supplied dataset.

Rather than hand-writing a routing function per field class, the YAML+CLI
path reuses ``build_dataset`` -- the same builder the CLI-only path uses --
with the dataset type supplied by the YAML instead of inferred from flags.

Two properties matter and are pinned here:

1. an explicitly-set flag reaches the resolved dataset, and
2. a flag the user did NOT set never overwrites the YAML value, including
   the defaults ``build_dataset`` materializes for the CLI-only path
   (``images.batch_size``, ``turns.stddev``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError


def cli(**kwargs: object) -> CLIConfig:
    """Build a CLIConfig whose model_fields_set is exactly ``kwargs``."""
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


def dataset(cfg: AIPerfConfig):
    return cfg.benchmark.datasets[0]


@pytest.fixture
def synthetic_yaml(tmp_path: Path) -> Path:
    """Synthetic dataset with values a CLI flag could clobber."""
    cfg = tmp_path / "synthetic.yaml"
    cfg.write_text(
        """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    random_seed: 11
    entries: 16
    prompts:
      isl:
        mean: 32
      osl:
        mean: 8
    turns:
      mean: 3
      stddev: 2
    images:
      width:
        mean: 64
      batch_size: 4
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    return cfg


@pytest.fixture
def random_pool_yaml(tmp_path: Path) -> Path:
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"text": "hi"}\n')
    cfg = tmp_path / "random_pool.yaml"
    cfg.write_text(
        f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: random_pool
    path: {pool}
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    return cfg


# ---------------------------------------------------------------------------
# Flags reach the resolved dataset
# ---------------------------------------------------------------------------


def test_random_seed_overrides_yaml(synthetic_yaml: Path) -> None:
    assert (
        dataset(resolve_config(cli(random_seed=99), synthetic_yaml)).random_seed == 99
    )


def test_image_width_mean_overrides_yaml(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(image_width_mean=128), synthetic_yaml))
    assert resolved.images.width.mean == 128


def test_scalar_isl_overrides_yaml(synthetic_yaml: Path) -> None:
    """The scalar form must now route, not just the magic-list form."""
    resolved = dataset(
        resolve_config(cli(prompt_input_tokens_mean=256), synthetic_yaml)
    )
    assert resolved.prompts.isl.mean == 256


def test_prefix_pool_size_overrides_yaml(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(prompt_prefix_pool_size=7), synthetic_yaml))
    assert resolved.prefix_prompts.pool_size == 7


def test_conversation_turn_mean_overrides_yaml(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(conversation_turn_mean=5), synthetic_yaml))
    assert resolved.turns.mean == 5


def test_audio_length_mean_overrides_yaml(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(audio_length_mean=2.5), synthetic_yaml))
    assert resolved.audio.length.mean == 2.5


def test_video_duration_overrides_yaml(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(video_duration=9), synthetic_yaml))
    assert resolved.video.duration == 9


def test_rankings_passages_mean_overrides_yaml(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(rankings_passages_mean=6), synthetic_yaml))
    assert resolved.rankings.passages.mean == 6


# ---------------------------------------------------------------------------
# Unset flags must never clobber YAML
# ---------------------------------------------------------------------------


def test_unset_fields_leave_yaml_untouched(synthetic_yaml: Path) -> None:
    resolved = dataset(resolve_config(cli(), synthetic_yaml))
    assert resolved.random_seed == 11
    assert resolved.prompts.isl.mean == 32
    assert resolved.turns.stddev == 2
    assert resolved.images.batch_size == 4


def test_image_width_flag_does_not_reset_yaml_batch_size(synthetic_yaml: Path) -> None:
    """build_dataset materializes images.batch_size=1 for the CLI-only path.

    Leaking that default into the override would silently reset a YAML batch
    size the user never mentioned -- the exact class of bug this work exists
    to remove.
    """
    resolved = dataset(resolve_config(cli(image_width_mean=128), synthetic_yaml))
    assert resolved.images.width.mean == 128
    assert resolved.images.batch_size == 4


def test_turn_mean_flag_does_not_reset_yaml_turn_stddev(synthetic_yaml: Path) -> None:
    """Likewise for the turns.stddev=0 default."""
    resolved = dataset(resolve_config(cli(conversation_turn_mean=5), synthetic_yaml))
    assert resolved.turns.mean == 5
    assert resolved.turns.stddev == 2


def test_isl_flag_does_not_reset_yaml_osl(synthetic_yaml: Path) -> None:
    resolved = dataset(
        resolve_config(cli(prompt_input_tokens_mean=256), synthetic_yaml)
    )
    assert resolved.prompts.osl.mean == 8


# ---------------------------------------------------------------------------
# The YAML owns the dataset type and source
# ---------------------------------------------------------------------------


def test_synthetic_only_flag_on_file_dataset_is_rejected(
    random_pool_yaml: Path,
) -> None:
    """A synthetic-shaping flag against a YAML file dataset must say so.

    The declared type comes from the YAML, so the converter's file-dataset
    guard has to fire on it rather than on --input-file being absent.
    """
    with pytest.raises((ValueError, ConfigurationError)) as excinfo:
        resolve_config(cli(prompt_prefix_pool_size=4), random_pool_yaml)
    assert "prefix" in str(excinfo.value).lower()


def test_random_pool_batch_size_still_routes(random_pool_yaml: Path) -> None:
    """Regression guard for PR #1274 behavior after the refactor."""
    resolved = dataset(resolve_config(cli(prompt_batch_size=7), random_pool_yaml))
    assert resolved.prompt_batch_size == 7


def test_input_file_flag_remains_rejected(synthetic_yaml: Path, tmp_path: Path) -> None:
    """--input-file would swap the dataset source the YAML declared."""
    other = tmp_path / "other.jsonl"
    other.write_text('{"text": "hi"}\n')
    with pytest.raises(ConfigurationError, match=r"--input-file"):
        resolve_config(CLIConfig(input_file=str(other)), synthetic_yaml)


def test_public_dataset_flag_remains_rejected(synthetic_yaml: Path) -> None:
    """--public-dataset would swap the dataset type the YAML declared."""
    with pytest.raises(ConfigurationError, match=r"--public-dataset"):
        resolve_config(cli(public_dataset="sharegpt"), synthetic_yaml)
