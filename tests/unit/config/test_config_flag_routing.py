# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI flags combined with ``--config`` must never be silently ignored.

``resolve_config`` historically routed only a small subset of ``CLIConfig``
fields into the YAML-supplied base config; every other explicitly-set flag was
silently discarded. A user passing ``-f base.yaml --random-seed 42`` got a run
with a different seed than they asked for, with no diagnostic.

These tests pin the guarantee that replaces that behavior: a flag that cannot
be routed under ``--config`` raises a ``ConfigurationError`` naming the flag,
rather than being dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags._section_fields import (
    ACCURACY_FIELDS,
    ENDPOINT_FIELDS,
    INPUT_FIELDS,
    LOADGEN_FIELDS,
    OUTPUT_FIELDS,
    SWEEPING_FIELDS,
    TOKENIZER_FIELDS,
)
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError

ALL_SECTIONS = (
    ACCURACY_FIELDS
    | ENDPOINT_FIELDS
    | INPUT_FIELDS
    | LOADGEN_FIELDS
    | OUTPUT_FIELDS
    | SWEEPING_FIELDS
    | TOKENIZER_FIELDS
)


@pytest.fixture
def base_yaml(tmp_path: Path) -> Path:
    """A minimal, valid YAML config with a synthetic dataset."""
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        """\
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
    )
    return cfg


def cli(**kwargs: object) -> CLIConfig:
    """Build a CLIConfig whose model_fields_set is exactly ``kwargs``."""
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unrouted flags must be loud
# ---------------------------------------------------------------------------


def test_unrouted_input_flag_raises_naming_the_flag(
    base_yaml: Path, tmp_path: Path
) -> None:
    """--input-file would swap the dataset the YAML declared, so it errors."""
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"text": "hi"}\n')
    with pytest.raises(ConfigurationError, match=r"--input-file"):
        resolve_config(CLIConfig(input_file=str(pool)), base_yaml)


def test_unrouted_sweeping_flag_raises(base_yaml: Path) -> None:
    """Sweep bounds resolve cleanly and change nothing, so they must error.

    (The warmup flags this used to check are routed now.)
    """
    with pytest.raises(ConfigurationError, match=r"--concurrency-min"):
        resolve_config(cli(concurrency_min=2), base_yaml)


def test_unrouted_endpoint_flag_raises(base_yaml: Path) -> None:
    """reset-kv-cache flags are unrouted under --config and must error."""
    with pytest.raises(ConfigurationError, match=r"--reset-kv-cache"):
        resolve_config(cli(reset_kv_cache=True), base_yaml)


def test_error_names_every_offending_flag(base_yaml: Path) -> None:
    """All unrouted flags are reported at once, not one per run."""
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_config(cli(concurrency_min=2, reset_kv_cache=True), base_yaml)
    message = str(excinfo.value)
    assert "--concurrency-min" in message
    assert "--reset-kv-cache" in message


def test_error_mentions_config_flag_as_the_cause(base_yaml: Path) -> None:
    """The message must tell users WHY the flag was rejected."""
    with pytest.raises(ConfigurationError, match=r"--config"):
        resolve_config(cli(concurrency_min=2), base_yaml)


# ---------------------------------------------------------------------------
# Routed flags keep working
# ---------------------------------------------------------------------------


def test_routed_endpoint_flag_still_applies(base_yaml: Path) -> None:
    """--streaming is routed; it must override YAML without raising."""
    resolved = resolve_config(cli(streaming=True), base_yaml)
    assert resolved.benchmark.endpoint.streaming is True


def test_routed_output_flag_still_applies(base_yaml: Path, tmp_path: Path) -> None:
    """--artifact-dir is routed through build_artifacts; must not raise."""
    target = tmp_path / "artifacts"
    resolved = resolve_config(cli(artifact_directory=target), base_yaml)
    assert Path(resolved.benchmark.artifacts.dir) == target


def test_no_flags_with_config_is_fine(base_yaml: Path) -> None:
    """A bare --config invocation must not trip the guard."""
    resolved = resolve_config(cli(), base_yaml)
    assert resolved.benchmark.models.items[0].name == "test-model"


# ---------------------------------------------------------------------------
# Magic lists: routed as sweep parameters, but only when list-shaped
# ---------------------------------------------------------------------------


def test_magic_list_isl_is_routed_as_sweep_parameter(base_yaml: Path) -> None:
    """A list-shaped --isl becomes a sweep parameter, so it must not raise."""
    resolved = resolve_config(cli(prompt_input_tokens_mean=[128, 256]), base_yaml)
    assert resolved.sweep is not None
    assert resolved.sweep.parameters["datasets.default.prompts.isl.mean"] == [128, 256]


def test_scalar_isl_routes_onto_the_dataset(base_yaml: Path) -> None:
    """The scalar form reaches the dataset block rather than a sweep.

    The magic-list form becomes a sweep parameter; the scalar form is routed
    by _apply_dataset_overrides. Both must take effect -- neither may be
    silently dropped.
    """
    resolved = resolve_config(cli(prompt_input_tokens_mean=128), base_yaml)
    assert resolved.benchmark.datasets[0].prompts.isl.mean == 128


# ---------------------------------------------------------------------------
# The CLI-only path is untouched
# ---------------------------------------------------------------------------


def test_unrouted_flag_without_config_is_allowed() -> None:
    """Without --config the converter handles every flag; no guard applies."""
    resolved = resolve_config(
        cli(
            random_seed=1234,
            model_names=["m"],
            urls=["http://localhost:8000"],
        ),
        None,
    )
    assert resolved.benchmark.datasets[0].random_seed == 1234


# ---------------------------------------------------------------------------
# Totality: no field may be silently unclassified
# ---------------------------------------------------------------------------


def test_every_cli_config_field_is_classified() -> None:
    """EVERY CLIConfig field must be classified, not just sectioned ones.

    The section frozensets are opt-in: adding a field to ``CLIConfig``
    requires nobody to touch ``_section_fields.py``, and
    ``test_section_fields_partition_cli_config`` only checks that section
    entries exist on CLIConfig -- never the reverse. So an unsectioned field
    was invisible to the guard and silently dropped under ``--config``, which
    is how 17 fields (all mlflow_*, --otel-url, network_latency_*, --scenario)
    came to be dropped without anyone noticing.

    Keying this on CLIConfig.model_fields instead makes classification
    mandatory: a newly-added flag belongs to no set and fails here, at
    authoring time.
    """
    from aiperf.config.flags._config_flag_routing import (
        EXEMPT_FROM_CONFIG_ROUTING,
        ROUTED_UNDER_CONFIG,
        UNROUTED_UNDER_CONFIG,
    )

    unclassified = (
        frozenset(CLIConfig.model_fields)
        - ROUTED_UNDER_CONFIG
        - UNROUTED_UNDER_CONFIG
        - EXEMPT_FROM_CONFIG_ROUTING
    )
    assert not unclassified, (
        f"CLI fields are unclassified for the --config path: "
        f"{sorted(unclassified)}. Every flag must be routed, listed in "
        f"UNROUTED_UNDER_CONFIG so users get a loud error, or exempted in "
        f"EXEMPT_FROM_CONFIG_ROUTING with a reason."
    )


def test_exempt_fields_are_not_also_classified() -> None:
    """An exemption must not overlap the routed or unrouted sets."""
    from aiperf.config.flags._config_flag_routing import (
        EXEMPT_FROM_CONFIG_ROUTING,
        ROUTED_UNDER_CONFIG,
        UNROUTED_UNDER_CONFIG,
    )

    overlap = EXEMPT_FROM_CONFIG_ROUTING & (ROUTED_UNDER_CONFIG | UNROUTED_UNDER_CONFIG)
    assert not overlap, f"exempt fields also classified: {sorted(overlap)}"


def test_every_section_field_is_classified() -> None:
    """Each section field is either routed under --config or known-unrouted.

    This is the regression gate. A newly-added CLI flag that nobody wired into
    the resolver lands in neither set and fails here, at authoring time,
    instead of being silently dropped for users.
    """
    from aiperf.config.flags._config_flag_routing import (
        ROUTED_UNDER_CONFIG,
        UNROUTED_UNDER_CONFIG,
    )

    unclassified = ALL_SECTIONS - ROUTED_UNDER_CONFIG - UNROUTED_UNDER_CONFIG
    assert not unclassified, (
        f"CLI fields are neither routed nor listed as unrouted under --config: "
        f"{sorted(unclassified)}. Route them in resolver.py, or add them to "
        f"UNROUTED_UNDER_CONFIG so users get a loud error instead of silence."
    )


def test_routed_and_unrouted_are_disjoint() -> None:
    """A field cannot be both routed and rejected."""
    from aiperf.config.flags._config_flag_routing import (
        ROUTED_UNDER_CONFIG,
        UNROUTED_UNDER_CONFIG,
    )

    assert not (ROUTED_UNDER_CONFIG & UNROUTED_UNDER_CONFIG)


def test_every_unrouted_field_has_a_real_flag_name() -> None:
    """Error messages must name actual CLI flags, not guessed kebab-case."""
    from aiperf.config.flags._config_flag_routing import (
        UNROUTED_UNDER_CONFIG,
        flag_names_for,
    )

    missing = sorted(f for f in UNROUTED_UNDER_CONFIG if not flag_names_for(f))
    assert not missing, f"No CLIParameter flag name found for: {missing}"


# ---------------------------------------------------------------------------
# Error messages must name a flag the user can find in their shell history
# ---------------------------------------------------------------------------


def test_error_lists_every_spelling_of_a_multi_alias_flag(base_yaml: Path) -> None:
    """Naming only the first declared spelling sends users looking for a flag
    they never typed.

    ``--sweep-variant`` was reported as ``--variant``. Since the resolver
    cannot see which spelling was typed, it names them all.
    """
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_config(cli(sweep_variants=["concurrency=2"]), base_yaml)
    message = str(excinfo.value)
    assert "--variant" in message
    assert "--sweep-variant" in message
