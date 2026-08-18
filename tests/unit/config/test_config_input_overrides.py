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


def test_zero_batch_size_overrides_yaml(tmp_path: Path) -> None:
    """``--prompt-batch-size 0`` must win over a non-zero YAML value.

    Zero disables the text modality on random_pool (FileDataset allows ge=0
    as of the base branch). It is also falsy, so an override path gating on
    truthiness rather than ``model_fields_set`` would drop it and leave the
    YAML batch size in place -- text still enabled, contrary to the request.
    """
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"text": "hi"}\n')
    cfg = tmp_path / "rp.yaml"
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
    prompt_batch_size: 4
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    assert dataset(resolve_config(cli(prompt_batch_size=0), cfg)).prompt_batch_size == 0
    # ...and an unset flag still leaves the YAML value alone.
    assert dataset(resolve_config(cli(), cfg)).prompt_batch_size == 4


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


# ---------------------------------------------------------------------------
# --num-conversations: must match CLI-only semantics, not error
# ---------------------------------------------------------------------------


def test_conversation_num_sets_entries_and_sessions(synthetic_yaml: Path) -> None:
    """``--num-conversations`` must do under --config what it does without it.

    On the CLI-only path the flag sets both ``dataset.entries`` and the
    profiling phase's ``sessions``. Under --config it produced neither, and
    the resolver rejected it with "no effect on a dataset of type
    'synthetic'" -- factually wrong, on one of the most-used flags in the
    tool.
    """
    resolved = resolve_config(cli(conversation_num=13), synthetic_yaml)
    assert dataset(resolved).entries == 13
    profiling = [p for p in resolved.benchmark.phases if p.name == "profiling"]
    assert profiling and profiling[0].sessions == 13


def test_conversation_num_dataset_entries_sets_entries(synthetic_yaml: Path) -> None:
    """The explicit entry-count flag wins for ``entries``."""
    resolved = resolve_config(cli(conversation_num_dataset_entries=21), synthetic_yaml)
    assert dataset(resolved).entries == 21


def test_request_count_does_not_become_dataset_entries(synthetic_yaml: Path) -> None:
    """--request-count must not silently reset a YAML ``entries``.

    _resolve_entries falls back to request_count on the CLI-only path, where
    it builds a dataset from nothing. As an override that fallback would
    overwrite the config file's entry count from an unrelated loadgen flag.
    """
    resolved = resolve_config(cli(request_count=50), synthetic_yaml)
    assert dataset(resolved).entries == 16  # the YAML value


# ---------------------------------------------------------------------------
# Per-flag reconciliation: a flag is not excused by its neighbours
# ---------------------------------------------------------------------------


def test_inert_flag_paired_with_routable_one_still_errors(
    synthetic_yaml: Path,
) -> None:
    """An unroutable dataset flag must be caught even in company.

    The guard used to fire only when build_dataset produced nothing at all,
    so pairing an inert flag with a working one left the result non-empty
    and the inert flag was discarded exactly as before this work. Every
    multi-flag command line touching the dataset escaped the guarantee.
    """
    with pytest.raises(ConfigurationError, match=r"--synthesis-max-isl"):
        resolve_config(
            cli(prompt_input_tokens_mean=128, synthesis_max_isl=512), synthetic_yaml
        )


def test_inert_media_flag_paired_on_file_dataset_errors(
    random_pool_yaml: Path,
) -> None:
    """Same, for the media flags against a file dataset."""
    with pytest.raises(ConfigurationError, match=r"--image-width-mean"):
        resolve_config(cli(prompt_batch_size=4, image_width_mean=128), random_pool_yaml)


def test_two_routable_flags_together_are_fine(synthetic_yaml: Path) -> None:
    """The reconciliation must not reject flags that do take effect."""
    resolved = dataset(
        resolve_config(
            cli(prompt_input_tokens_mean=128, image_width_mean=64), synthetic_yaml
        )
    )
    assert resolved.prompts.isl.mean == 128
    assert resolved.images.width.mean == 64


def test_dataset_flag_outside_input_fields_is_reconciled(
    synthetic_yaml: Path,
) -> None:
    """--inter-turn-delay-cap-seconds is dataset-carried but not INPUT_FIELDS.

    Fields outside INPUT_FIELDS were never considered by the guard at all,
    so this one no-oped silently on a synthetic dataset.
    """
    with pytest.raises(ConfigurationError, match=r"--inter-turn-delay-cap-seconds"):
        resolve_config(cli(inter_turn_delay_cap_seconds=3.0), synthetic_yaml)


# ---------------------------------------------------------------------------
# YAML-declared baseten_trace: the guards must read the declared identity
# ---------------------------------------------------------------------------


@pytest.fixture
def baseten_yaml(tmp_path: Path) -> Path:
    """A file dataset that declares format: baseten_trace in the YAML."""
    trace = tmp_path / "baseten.jsonl"
    trace.write_text('{"text": "hi"}\n')
    cfg = tmp_path / "baseten.yaml"
    cfg.write_text(
        f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: baseten_trace
    path: {trace}
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    return cfg


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("replay_speedup", 2.0),
        ("max_idle_gap_cap_seconds", 3.0),
        ("trace_session_sample_ratio", 0.5),
        ("open_loop_replay", True),
        ("omit_kv_hints", True),
    ],
)
def test_baseten_only_flags_accepted_on_yaml_declared_baseten_trace(
    baseten_yaml: Path, field: str, value: object
) -> None:
    """These are baseten_trace-only, and the YAML declares baseten_trace.

    The guard inferred the loader from --input-file / --custom-dataset-type,
    which the --config path rejects outright because the config file owns
    the dataset source and format. So a supported combination failed before
    the run started.
    """
    resolved = dataset(resolve_config(cli(**{field: value}), baseten_yaml))
    assert getattr(resolved, field) == value


def test_baseten_only_flag_still_rejected_on_other_trace_format(
    random_pool_yaml: Path,
) -> None:
    """The guard must still fire when the YAML declares a different loader."""
    with pytest.raises(ValueError, match="baseten_trace"):
        resolve_config(cli(replay_speedup=2.0), random_pool_yaml)


def test_baseten_extra_input_collision_rejected_from_yaml_format(
    baseten_yaml: Path,
) -> None:
    """Loader-injected extras collide whether the format came from YAML or CLI.

    The collision guard keyed off cli.custom_dataset_type, so under --config
    it never ran and the user's value was clobbered on the wire instead.
    """
    with pytest.raises(ValueError, match="min_tokens"):
        resolve_config(cli(extra_inputs=["min_tokens:5"]), baseten_yaml)


def test_synthetic_only_flag_message_is_actionable_under_config(
    random_pool_yaml: Path,
) -> None:
    """The advice must be something a --config user can act on.

    The CLI-only wording says to remove --input-file / --public-dataset. On
    this path the user passed neither, and both are rejected outright, so
    that advice is unfollowable. Point at the config file instead -- the same
    correction already made for --dataset-filter one function over.
    """
    with pytest.raises(ValueError) as excinfo:
        resolve_config(cli(prompt_prefix_pool_size=4), random_pool_yaml)
    message = str(excinfo.value)
    assert "--input-file" not in message
    assert "--public-dataset" not in message
    assert "config file" in message


def test_batch_size_message_is_actionable_under_config(tmp_path: Path) -> None:
    """Same for the batch-size variant of the guard."""
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"text": "hi"}\n')
    cfg = tmp_path / "mooncake.yaml"
    cfg.write_text(
        f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: mooncake_trace
    path: {trace}
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_config(cli(image_batch_size=2), cfg)
    message = str(excinfo.value)
    assert "--input-file" not in message
    assert "config file" in message
