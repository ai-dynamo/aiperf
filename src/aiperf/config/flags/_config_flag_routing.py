# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which CLI flags actually reach ``AIPerfConfig`` when ``--config`` is used.

``resolve_config`` merges a YAML base with explicitly-set CLI flags, but it
only knows how to route a subset of ``CLIConfig`` into the merged dict. Any
flag outside that subset used to be discarded without a word: a user running
``aiperf profile -f base.yaml --random-seed 42`` got a different seed than
they asked for and no diagnostic. For a benchmarking tool that silently
corrupts published numbers.

``CLIConfig`` is the source of truth, and that is structural rather than a
convention: ``resolve_config`` takes a ``CLIConfig`` and a path, and both
``aiperf profile`` and ``aiperf service`` call it with nothing else. A flag
can therefore reach ``AIPerfConfig`` only by being a field on ``CLIConfig``,
which is what lets the classification test enumerate ``model_fields`` and
claim completeness. Give ``resolve_config`` another source of user intent
and that claim quietly stops holding -- extend the classification in the
same change. (``aiperf service``'s own ``--service-id`` / ``--health-*``
parameters live on the command, never reach this path, and are out of scope
by construction.)

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


# Fields that are not benchmark configuration at all, so "routed under
# --config" does not apply to them. Each needs a reason: an unexplained
# exemption is a silent drop with extra steps.
EXEMPT_FROM_CONFIG_ROUTING: frozenset[str] = frozenset(
    {
        # The --config flag itself; it selects the file being merged.
        "config_file",
        # Alias input consumed during CLIConfig validation rather than routed
        # onto AIPerfConfig; it populates its canonical sibling, which is
        # itself classified.
        "stream",
    }
)

# Fields outside every section frozenset that nonetheless reach AIPerfConfig,
# verified by resolving with the flag set and diffing against the no-flag
# baseline. The section sets are opt-in and incomplete; this records what is
# genuinely routed rather than letting absence imply anything.
_ROUTED_OUTSIDE_SECTIONS: frozenset[str] = frozenset(
    {
        # service runtime / logging, via build_logging_runtime
        "api_host",
        "api_port",
        "extra_verbose",
        "log_level",
        "record_processor_service_count",
        "stats_interval",
        "ui_type",
        "verbose",
        "workers_max",
        "zmq_dual_bind",
        "zmq_ipc_path",
        "zmq_tcp_host",
        # telemetry collectors already wired into build_cli_overrides
        "gpu_telemetry",
        "no_gpu_telemetry",
        "server_metrics",
        "no_server_metrics",
        "server_metrics_formats",
        # wandb, via build_wandb (the non-project flags are guarded loudly
        # rather than dropped when --wandb-project is absent)
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_tags",
        # agentic phase fields, via _apply_agentic_replay_fields
        "agentic_cache_warmup_duration",
        "agentic_warmup_grace_period",
        # publishing targets, via build_mlflow / build_otel /
        # build_network_latency -- the builders the CLI-only converter has
        # always used, now called by build_cli_overrides too
        "mlflow_artifact_globs",
        "mlflow_experiment",
        "mlflow_parent_run_id",
        "mlflow_run_name",
        "mlflow_tags",
        "mlflow_tracking_uri",
        "otel_resource_attributes",
        "otel_url",
        "gen_ai_provider",
        "network_latency_automatic",
        "network_latency_mean",
        "network_latency_ping_interval",
        # scenario lock, via _apply_scenario_overrides
        "scenario",
        "unsafe_override",
        # inter-turn delay cap, carried onto trace datasets by build_dataset
        "inter_turn_delay_cap_seconds",
        # verbatim system prompt, carried onto the dataset by
        # _VERBATIM_DATASET_FIELDS (added on main by #1268)
        "system_prompt",
        "system_prompt_file",
        # baseten_trace dataset knobs carried by _VERBATIM_DATASET_FIELDS;
        # against another format they raise rather than drop
        "force_min_tokens",
        "max_idle_gap_cap_seconds",
        "omit_kv_hints",
        "open_loop_replay",
        "open_loop_strict",
        "replay_speedup",
        "trace_session_sample_ratio",
    }
)


# SWEEPING members that resolve cleanly under --config and change nothing,
# each verified by resolving with the flag and diffing the result. Notably
# --ttft-sla-ms does NOT take effect even alongside --search-recipe plus
# --streaming, contrary to the example in resolver's module docstring.
SWEEP_FIELDS_NOT_ROUTED: frozenset[str] = frozenset(
    {
        "concurrency_max",
        "concurrency_min",
        "concurrency_steps",
        "convergence_mode",
        "convergence_stat",
        "degradation_metric_tag",
        "degradation_stat",
        "e2e_sla_ms",
        "isl_max",
        "isl_min",
        "isl_steps",
        "itl_sla_ms",
        "osl_max",
        "osl_min",
        "osl_steps",
        "parameter_sweep_cooldown_seconds",
        "parameter_sweep_mode",
        "parameter_sweep_same_seed",
        "search_style",
        "sweep_variants",
        "tpot_sla_ms",
        "ttft_sla_ms",
    }
)


# Flags that only route when a companion flag is also present, verified by
# resolving with and without the companion. Without it they resolve cleanly
# and change nothing, so they must be rejected rather than quietly ignored.
COMPANION_ROUTED: dict[str, frozenset[str]] = {
    # _apply_endpoint_overrides writes `models.strategy` only inside the
    # `model_names` branch.
    "model_selection_strategy": frozenset({"model_names"}),
    # _apply_phase_loadgen_overrides consults arrival_pattern only when
    # rewriting a phase for --request-rate-series.
    "arrival_pattern": frozenset({"request_rate_series"}),
}


# Flags that would replace the dataset the config file declared rather than
# shape it: its source file, its public-dataset identity, its format. The YAML
# owns those, and build_dataset drops the corresponding keys in override mode,
# so the flags must keep erroring instead of appearing to work.
DATASET_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        # Switch the dataset TYPE, not just a value inside it. The variants
        # barely overlap -- FileDataset carries a dozen fields PublicDataset
        # rejects -- so accepting these would silently invalidate whole groups
        # of keys the user wrote in the config file.
        "public_dataset",
        "hf_weka_dataset",
    }
)

# Source flags that change a value WITHIN the declared type: the trace file
# and the loader that reads it. Overriding these is the "same benchmark,
# different trace" loop, and is no more surprising than --model overriding
# models.items. build_dataset applies them only when the config file declares
# type: file; against any other type they raise, because that would be a type
# switch wearing a different hat.
DATASET_SOURCE_IN_TYPE_FIELDS: frozenset[str] = frozenset({})

# INPUT_FIELDS members that build_dataset does not carry: they land on the
# slos block or the profiling phase instead. They ARE routed (see
# _apply_phase_loadgen_overrides and the goodput handling in
# build_cli_overrides); this set exists to keep them out of the dataset
# reconciliation, which would otherwise report them as inert because
# build_dataset legitimately emits nothing for them.
_INPUT_NOT_ON_DATASET: frozenset[str] = frozenset(
    {
        # --goodput lands on the benchmark `slos` block.
        "goodput",
        # The fixed-schedule flags select and shape the profiling phase, so
        # they are routed by _apply_phase_loadgen_overrides rather than
        # carried on the dataset -- and must stay out of the dataset
        # reconciliation, which would otherwise report them as inert.
        "fixed_schedule",
        "fixed_schedule_auto_offset",
        "fixed_schedule_end_offset",
        "fixed_schedule_start_offset",
    }
)


# Dataset-carried fields that are not INPUT_FIELDS members. build_dataset
# reads them (via _VERBATIM_DATASET_FIELDS and the trace helpers), so the
# per-flag reconciliation in _apply_dataset_overrides must consider them too;
# keying that check on INPUT_FIELDS alone let them no-op silently.
DATASET_FIELDS_OUTSIDE_INPUT: frozenset[str] = frozenset(
    {
        "force_min_tokens",
        "inter_turn_delay_cap_seconds",
        "max_idle_gap_cap_seconds",
        "omit_kv_hints",
        "open_loop_replay",
        "open_loop_strict",
        "replay_speedup",
        "system_prompt",
        "system_prompt_file",
        "trace_idle_gap_cap_seconds",
        "trace_session_sample_ratio",
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
    from aiperf.config.flags._converter_profiling import (
        _AGENTIC_REPLAY_ROUTES,
        _GAMMA_ONLY_ROUTES,
        _RAMP_FIELDS,
    )
    from aiperf.config.flags.resolver import _LOADGEN_PHASE_FIELD_MAP

    # _apply_endpoint_overrides: the field map, plus --url and the
    # --model-names/--model-selection-strategy pair that lands on `models`.
    endpoint = (
        set(_ENDPOINT_FIELD_MAP)
        | {
            "urls",
            "model_names",
            "model_selection_strategy",
        }
        | {
            # probe sub-blocks, via _maybe_build_reset_kv_cache /
            # _maybe_build_server_profiler
            field
            for field in ENDPOINT_FIELDS
            if field.startswith(("reset_kv_cache", "server_profiler"))
        }
    )

    # _apply_input_overrides routes headers/extra onto the endpoint block;
    # every other dataset-shaping field goes through _apply_dataset_overrides,
    # which delegates to build_dataset in override mode.
    #
    # The exceptions are the fields that would replace the dataset the config
    # file declared rather than shape it -- source, type, and format. The YAML
    # owns those, so the flags stay rejected (see DATASET_SOURCE_FIELDS).
    inputs = (INPUT_FIELDS - DATASET_SOURCE_FIELDS) - _INPUT_NOT_ON_DATASET
    inputs |= DATASET_SOURCE_IN_TYPE_FIELDS

    # _apply_phase_loadgen_overrides: the phase field map, the rate-series
    # pair, and the AGENTIC_REPLAY fields shared with BasePhaseConfig.
    loadgen = (
        {attr for attr, _ in _LOADGEN_PHASE_FIELD_MAP}
        | {"request_rate_series", "arrival_pattern"}
        | (set(_AGENTIC_REPLAY_ROUTES) & LOADGEN_FIELDS)
        # Phase shaping, applied by _apply_phase_shaping_overrides through the
        # converter's own helpers. Derived from the same tables those helpers
        # read, so a new ramp or gamma-only route counts as routed here too.
        | {attr for attr, _ in _RAMP_FIELDS}
        | {attr for _, attr in _GAMMA_ONLY_ROUTES}
        | {"request_cancellation_rate", "request_cancellation_delay"}
        # Warmup phase shaping, applied by _apply_warmup_overrides.
        | {field for field in LOADGEN_FIELDS if field.startswith("warmup_")}
        # Fixed-schedule phase selection and its offsets, and --goodput, which
        # lands on the benchmark slos block.
        | {field for field in INPUT_FIELDS if field.startswith("fixed_schedule")}
        | {"goodput"}
    )

    # Whole-section builders consumed by build_cli_overrides: build_artifacts
    # + resolve_auto_plot, build_tokenizer, build_accuracy.
    whole_sections = OUTPUT_FIELDS | TOKENIZER_FIELDS | ACCURACY_FIELDS

    # SWEEPING minus the members verified to resolve cleanly while changing
    # nothing (see SWEEP_FIELDS_NOT_ROUTED). The rest -- --search-recipe,
    # --search-space and friends -- do take effect. build_sweep / expand_search_recipe
    # consume the section, but only conditionally -- e.g. --concurrency-min /
    # --concurrency-max / --concurrency-steps under --config verifiably
    # produce no sweep at all, while --ttft-sla-ms does take effect alongside
    # --search-recipe. Separating genuine drops from "needs a companion flag"
    # requires a combination audit that is out of scope here, and erroring on
    # the whole section would break valid recipe invocations. Tracked as
    # follow-up work; see AIP-1133.
    sweeping = set(SWEEPING_FIELDS) - SWEEP_FIELDS_NOT_ROUTED

    return frozenset(
        endpoint
        | inputs
        | loadgen
        | whole_sections
        | sweeping
        | _ROUTED_OUTSIDE_SECTIONS
        | DATASET_FIELDS_OUTSIDE_INPUT
    )


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
        # ----- input: dataset identity, owned by the config file -----
        # Every other INPUT field now routes through _apply_dataset_overrides;
        # these would swap the dataset the YAML declared rather than shape it.
        *DATASET_SOURCE_FIELDS,
        # ----- sweep flags that resolve cleanly and do nothing -----
        *SWEEP_FIELDS_NOT_ROUTED,
        # ----- outside every section frozenset -----
        # Still dropped: --sweep-type needs a sweep block to attach to, and
        # --disable-auto-fixed-schedule is consumed by phase construction
        # which this path does not rebuild. Both need their own routing.
        "sweep_type",
        "disable_auto_fixed_schedule",
        # ----- loadgen: ramps, pacing, cancellation -----
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
    from aiperf.config.flags import CLIConfig

    info = CLIConfig.model_fields.get(field)
    if info is None:
        return ()
    # Fields declare their flags via CLIParameter or via cyclopts' Parameter
    # (--parameter-sweep-same-seed uses the latter). Match structurally on a
    # `name` tuple of dash-prefixed spellings rather than on either class, so
    # a third declaration style still yields real flag names.
    for meta in info.metadata:
        names = getattr(meta, "name", None)
        if isinstance(names, (tuple, list)) and any(
            isinstance(n, str) and n.startswith("--") for n in names
        ):
            return tuple(n for n in names if isinstance(n, str))
    return ()


def _describe(field: str) -> str:
    """Render a field as every spelling it accepts, falling back to the name.

    Naming only the first declared spelling meant a user who typed
    ``--num-warmup-requests`` was told about ``--warmup-request-count``, and
    ``--num-conversations`` was reported as ``--conversation-num`` -- flags
    absent from their shell history. This layer cannot see which spelling was
    typed (the CLIConfig it receives has already been parsed), so it names
    them all rather than guessing.
    """
    names = flag_names_for(field)
    return "/".join(names) if names else field


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

    # Keyed on every CLIConfig field rather than on the section frozensets:
    # those are opt-in, and a field nobody sectioned was invisible here.
    unrouted = cli.model_fields_set - ROUTED_UNDER_CONFIG - EXEMPT_FROM_CONFIG_ROUTING
    # Companion-routed flags are inert on their own, so treat a missing
    # companion the same as no routing at all.
    unrouted |= {
        field
        for field, companions in COMPANION_ROUTED.items()
        if field in cli.model_fields_set and not (companions & cli.model_fields_set)
    }
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
