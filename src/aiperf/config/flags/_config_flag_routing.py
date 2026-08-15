# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which CLI flags actually reach ``AIPerfConfig`` when ``--config`` is used.

``resolve_config`` merges a YAML base with explicitly-set CLI flags, but it
only knows how to route a subset of ``CLIConfig`` into the merged dict. Any
flag outside that subset used to be discarded without a word: a user running
``aiperf profile -f base.yaml --random-seed 42`` got a different seed than
they asked for and no diagnostic. For a benchmarking tool that silently
corrupts published numbers.

This module makes the boundary explicit and enforceable:

``ROUTED_UNDER_CONFIG``
    Fields the resolver genuinely routes. Derived from the same maps the
    resolver and converters use, so it cannot drift from them.

``UNROUTED_UNDER_CONFIG``
    Fields known to be dropped. Written out by hand on purpose -- a computed
    complement would make ``test_every_section_field_is_classified`` vacuous.
    A newly-added CLI flag lands in neither set and fails that test at
    authoring time, which is the whole point.

:func:`reject_unrouted_cli_flags` turns the drop into a ``ConfigurationError``
naming the offending flags. Repairing the routing (so entries move from
``UNROUTED`` to ``ROUTED``) is separate work; until then users get a loud
error instead of a wrong benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.flags._section_fields import (
    ACCURACY_FIELDS,
    ENDPOINT_FIELDS,
    INPUT_FIELDS,
    LOADGEN_FIELDS,
    OUTPUT_FIELDS,
    SWEEPING_FIELDS,
    TOKENIZER_FIELDS,
)

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig

ALL_SECTION_FIELDS: frozenset[str] = (
    ACCURACY_FIELDS
    | ENDPOINT_FIELDS
    | INPUT_FIELDS
    | LOADGEN_FIELDS
    | OUTPUT_FIELDS
    | SWEEPING_FIELDS
    | TOKENIZER_FIELDS
)


# Flags that would replace the dataset the config file declared rather than
# shape it: its source file, its public-dataset identity, its format. The YAML
# owns those, and build_dataset drops the corresponding keys in override mode,
# so the flags must keep erroring instead of appearing to work.
DATASET_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "input_file",
        "public_dataset",
        "custom_dataset_type",
        "hf_weka_dataset",
    }
)

# INPUT_FIELDS members that build_dataset does not carry -- they land
# somewhere other than the dataset block (SLOs, phase timing), so routing
# them is separate work and they stay loud for now.
_INPUT_NOT_ON_DATASET: frozenset[str] = frozenset(
    {
        "goodput",
        "fixed_schedule",
        "fixed_schedule_auto_offset",
        "fixed_schedule_end_offset",
        "fixed_schedule_start_offset",
    }
)


# INPUT fields that _apply_dataset_overrides carries onto the dataset block.
# Used to decide whether a config file lacking a dataset is an error for this
# invocation or simply nothing to do.
DATASET_OVERRIDE_FIELDS: frozenset[str] = (
    INPUT_FIELDS - DATASET_SOURCE_FIELDS - _INPUT_NOT_ON_DATASET
) - {"headers", "extra_inputs"}


def _build_routed_under_config() -> frozenset[str]:
    """Derive the routed set from the resolver's own routing tables.

    Imports are local: ``resolver`` imports this module's reject helper
    lazily, and reaching back into it at module scope would close the cycle.
    Deriving rather than restating means adding an entry to
    ``_ENDPOINT_FIELD_MAP`` or ``_LOADGEN_PHASE_FIELD_MAP`` automatically
    counts as routed here.
    """
    from aiperf.config.flags._converter_endpoint import _ENDPOINT_FIELD_MAP
    from aiperf.config.flags._converter_profiling import _AGENTIC_REPLAY_ROUTES
    from aiperf.config.flags.resolver import _LOADGEN_PHASE_FIELD_MAP

    # _apply_endpoint_overrides: the field map, plus --url and the
    # --model-names/--model-selection-strategy pair that lands on `models`.
    endpoint = set(_ENDPOINT_FIELD_MAP) | {
        "urls",
        "model_names",
        "model_selection_strategy",
    }

    # _apply_input_overrides routes headers/extra onto the endpoint block;
    # every other dataset-shaping field goes through _apply_dataset_overrides,
    # which delegates to build_dataset in override mode.
    #
    # The exceptions are the fields that would replace the dataset the config
    # file declared rather than shape it -- source, type, and format. The YAML
    # owns those, so the flags stay rejected (see DATASET_SOURCE_FIELDS).
    inputs = (INPUT_FIELDS - DATASET_SOURCE_FIELDS) - _INPUT_NOT_ON_DATASET

    # _apply_phase_loadgen_overrides: the phase field map, the rate-series
    # pair, and the AGENTIC_REPLAY fields shared with BasePhaseConfig.
    loadgen = (
        {attr for attr, _ in _LOADGEN_PHASE_FIELD_MAP}
        | {"request_rate_series", "arrival_pattern"}
        | (set(_AGENTIC_REPLAY_ROUTES) & LOADGEN_FIELDS)
    )

    # Whole-section builders consumed by build_cli_overrides: build_artifacts
    # + resolve_auto_plot, build_tokenizer, build_accuracy.
    whole_sections = OUTPUT_FIELDS | TOKENIZER_FIELDS | ACCURACY_FIELDS

    # SWEEPING is treated as routed in full. build_sweep / expand_search_recipe
    # consume the section, but only conditionally -- e.g. --concurrency-min /
    # --concurrency-max / --concurrency-steps under --config verifiably
    # produce no sweep at all, while --ttft-sla-ms does take effect alongside
    # --search-recipe. Separating genuine drops from "needs a companion flag"
    # requires a combination audit that is out of scope here, and erroring on
    # the whole section would break valid recipe invocations. Tracked as
    # follow-up work; see AIP-1133.
    sweeping = set(SWEEPING_FIELDS)

    return frozenset(endpoint | inputs | loadgen | whole_sections | sweeping)


ROUTED_UNDER_CONFIG: frozenset[str] = _build_routed_under_config()


def _build_magic_list_only_fields() -> frozenset[str]:
    """Dataset fields routed under ``--config`` only in their list form.

    ``promote_benchmark_magic_lists`` hoists a list-shaped ``--isl 128 256``
    into a ``sweep`` parameter, so those invocations resolve correctly. The
    scalar ``--isl 128`` has no such path and is dropped. Treating the field
    as routed outright would re-open the silent drop for scalars; rejecting it
    outright would break working magic-list runs. So the guard decides per
    value, and these are the fields it decides on.
    """
    from aiperf.config.flags.converter import _CLI_DATASET_MAGIC_LIST_PATHS

    return frozenset(attr for attr, _ in _CLI_DATASET_MAGIC_LIST_PATHS)


MAGIC_LIST_ONLY_UNDER_CONFIG: frozenset[str] = _build_magic_list_only_fields()

# Fields silently discarded when --config is supplied. Listed explicitly so
# that a new CLI flag belongs to neither set and trips the classification
# test. Entries move to ROUTED_UNDER_CONFIG as routing is repaired.
UNROUTED_UNDER_CONFIG: frozenset[str] = frozenset(
    {
        # ----- endpoint -----
        "reset_kv_cache",
        "reset_kv_cache_path",
        "reset_kv_cache_timeout_seconds",
        "server_profiler",
        "server_profiler_start_path",
        "server_profiler_stop_path",
        "server_profiler_timeout_seconds",
        # ----- input: dataset identity, owned by the config file -----
        # Every other INPUT field now routes through _apply_dataset_overrides;
        # these would swap the dataset the YAML declared rather than shape it.
        *DATASET_SOURCE_FIELDS,
        # ----- input: not carried on the dataset block -----
        *_INPUT_NOT_ON_DATASET,
        # ----- loadgen: warmup phase -----
        "warmup_arrival_pattern",
        "warmup_concurrency",
        "warmup_concurrency_ramp_duration",
        "warmup_duration",
        "warmup_grace_period",
        "warmup_num_sessions",
        "warmup_prefill_concurrency",
        "warmup_prefill_concurrency_ramp_duration",
        "warmup_request_count",
        "warmup_request_rate",
        "warmup_request_rate_ramp_duration",
        # ----- loadgen: ramps, pacing, cancellation -----
        "arrival_smoothness",
        "concurrency_ramp_duration",
        "prefill_concurrency_ramp_duration",
        "request_rate_ramp_duration",
        "request_cancellation_delay",
        "request_cancellation_rate",
        "trace_idle_gap_cap_seconds",
    }
)


def flag_names_for(field: str) -> tuple[str, ...]:
    """Return the CLI flag spellings for a ``CLIConfig`` field.

    Reads the ``CLIParameter`` annotation metadata rather than kebab-casing
    the attribute, because several fields do not follow from their name --
    ``conversation_num`` is ``--num-conversations``, ``warmup_request_count``
    is also ``--num-warmup-requests``. Error messages have to name flags the
    user can actually type.

    Returns an empty tuple when the field carries no ``CLIParameter``.
    """
    from aiperf.config.cli_parameter import CLIParameter
    from aiperf.config.flags import CLIConfig

    info = CLIConfig.model_fields.get(field)
    if info is None:
        return ()
    for meta in info.metadata:
        if isinstance(meta, CLIParameter) and meta.name:
            return tuple(meta.name)
    return ()


def _describe(field: str) -> str:
    """Render a field as its primary flag spelling, falling back to the name."""
    names = flag_names_for(field)
    return names[0] if names else field


def reject_unrouted_cli_flags(cli: CLIConfig) -> None:
    """Raise when explicitly-set CLI flags cannot be routed under ``--config``.

    Gating is on ``cli.model_fields_set`` -- a flag the user did not pass can
    never trigger this, and a flag passed with a falsy value (``--concurrency 0``)
    still counts as set.

    Args:
        cli: the parsed ``CLIConfig`` for this invocation.

    Raises:
        ConfigurationError: naming every offending flag at once, so a user
            fixing a long command line learns about all of them in one run.
    """
    from aiperf.config.loader.errors import ConfigurationError

    unrouted = (cli.model_fields_set & ALL_SECTION_FIELDS) - ROUTED_UNDER_CONFIG
    # Magic-list fields resolve correctly in their list form; only the scalar
    # form falls through unrouted.
    unrouted -= {
        field
        for field in unrouted & MAGIC_LIST_ONLY_UNDER_CONFIG
        if isinstance(getattr(cli, field, None), list)
    }
    if not unrouted:
        return

    flags = ", ".join(sorted(_describe(field) for field in unrouted))
    raise ConfigurationError(
        f"These CLI flags cannot be combined with --config and would "
        f"otherwise be ignored: {flags}. Set the equivalent values in the "
        f"YAML config file, or drop --config and pass the run entirely on "
        f"the command line."
    )
