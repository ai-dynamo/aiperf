# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-CLI-flag round-trip tests for the v1 -> v2 converter.

For every CLI flag bearing v1 field, asserts:

1. Setting the flag in isolation lands the value at the expected v2 path.
2. Leaving every flag unset emits NO override for that field (model_fields_set
   gating works for that field).
3. Setting one flag does not accidentally emit overrides for unrelated v2
   sections (no cross-contamination via shared mutable state).

The table here is the source of truth for "what flag maps where". When a new
v1 flag lands, add a row. When a flag's v2 destination changes, update the
row. A regression in `_ENDPOINT_FIELD_MAP` or any of the
`build_*` mapping dicts shows up as a single failed parametrize case with a
clear "expected X, got Y" diff.

The flags themselves are documented in ``docs/cli-options.md``; this is the
*test* counterpart that locks in the mapping.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._resolver import build_v1_overrides


def _get_at_path(d: dict[str, Any], path: str) -> Any:
    """Walk ``d`` via dotted ``path`` (no list-of-named-dict resolution).

    Used by the table-driven tests; the v2 paths exercised here are all
    plain dict traversals (e.g. ``endpoint.streaming``,
    ``multi_run.num_runs``). For the few list-of-named-dict cases (models /
    phases / datasets) we use indexed paths like ``models.items.0.name``.
    """
    cur: Any = d
    for seg in path.split("."):
        if seg.isdigit() and isinstance(cur, list):
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            cur = cur[seg]
        else:
            raise KeyError(f"path {path!r} hit non-traversable {type(cur).__name__}")
    return cur


# =====================================================================
# Endpoint flags (v1 EndpointConfig + InputConfig.{headers, extra})
# =====================================================================
#
# Format: (v1_path, v1_value, v2_path, expected_v2_value)
#   v1_path: dotted path on UserConfig (e.g. "endpoint.streaming")
#   v2_path: dotted path on the override dict (use ".0" for list indices)
#   expected_v2_value: what _get_at_path should return (None means "passthrough")

_ENDPOINT_FLAG_TABLE: list[tuple[str, Any, str, Any]] = [
    # --streaming → endpoint.streaming
    ("endpoint.streaming", True, "endpoint.streaming", True),
    # --endpoint-type chat → endpoint.type
    ("endpoint.type", "chat", "endpoint.type", "chat"),
    # --endpoint-type embeddings (different value through enum normalization)
    ("endpoint.type", "embeddings", "endpoint.type", "embeddings"),
    # --custom-endpoint /v1/foo → endpoint.path (renamed in _ENDPOINT_FIELD_MAP)
    ("endpoint.custom_endpoint", "/v1/foo", "endpoint.path", "/v1/foo"),
    # --api-key → endpoint.api_key
    ("endpoint.api_key", "sk-test", "endpoint.api_key", "sk-test"),
    # --request-timeout-seconds → endpoint.timeout (renamed)
    ("endpoint.timeout_seconds", 600, "endpoint.timeout", 600),
    # --ready-check-timeout → endpoint.ready_check_timeout
    ("endpoint.ready_check_timeout", 30.0, "endpoint.ready_check_timeout", 30.0),
    # --ready-check-mode → endpoint.ready_check_mode
    ("endpoint.ready_check_mode", "models", "endpoint.ready_check_mode", "models"),
    # --transport http2 → endpoint.transport
    ("endpoint.transport", "http2", "endpoint.transport", "http2"),
    # --use-legacy-max-tokens → endpoint.use_legacy_max_tokens
    (
        "endpoint.use_legacy_max_tokens",
        True,
        "endpoint.use_legacy_max_tokens",
        True,
    ),
    # --use-server-token-count → endpoint.use_server_token_count
    (
        "endpoint.use_server_token_count",
        True,
        "endpoint.use_server_token_count",
        True,
    ),
    # --connection-reuse-strategy → endpoint.connection_reuse (renamed)
    (
        "endpoint.connection_reuse_strategy",
        "pooled",
        "endpoint.connection_reuse",
        "pooled",
    ),
    # --download-video-content → endpoint.download_video_content
    (
        "endpoint.download_video_content",
        True,
        "endpoint.download_video_content",
        True,
    ),
    # --request-content-type → endpoint.request_content_type
    (
        "endpoint.request_content_type",
        "application/json",
        "endpoint.request_content_type",
        "application/json",
    ),
    # --url-selection-strategy → endpoint.url_strategy (renamed)
    (
        "endpoint.url_selection_strategy",
        "round_robin",
        "endpoint.url_strategy",
        "round_robin",
    ),
    # --urls → endpoint.urls (passthrough)
    (
        "endpoint.urls",
        ["http://a", "http://b"],
        "endpoint.urls",
        ["http://a", "http://b"],
    ),
    # --headers (lives on v1 InputConfig.headers, lifts to v2 endpoint.headers)
    (
        "input.headers",
        ["Authorization: Bearer X"],
        "endpoint.headers",
        # InputConfig parses a colon-separated list into a dict.
        {"Authorization": "Bearer X"},
    ),
]


@pytest.mark.parametrize(
    "v1_path,v1_value,v2_path,expected",
    [param(*row, id=f"{row[0]}_to_{row[2]}") for row in _ENDPOINT_FLAG_TABLE],
)
def test_endpoint_flag_round_trip(
    v1_path: str, v1_value: Any, v2_path: str, expected: Any
) -> None:
    """Each endpoint-side CLI flag lands at its declared v2 path, unchanged
    in shape (modulo the renames documented in ``_ENDPOINT_FIELD_MAP``)."""
    section, field = v1_path.split(".", 1)
    user = UserConfig.model_validate({section: {field: v1_value}})
    out = build_v1_overrides(user)
    assert _get_at_path(out, v2_path) == expected


# =====================================================================
# Models flag (model_names, model_selection_strategy live on v1 EndpointConfig
# but lift to v2 `models` section, NOT `endpoint`)
# =====================================================================


def test_endpoint_model_names_lifts_to_models_block() -> None:
    """v1 ``endpoint.model_names`` is the only field that lives on
    EndpointConfig but maps to a different v2 section. The rename is
    invisible at the CLI level (`--model X` is just ergonomic) but the
    routing must hold or `models.items` ends up empty."""
    user = UserConfig.model_validate({"endpoint": {"model_names": ["a", "b"]}})
    out = build_v1_overrides(user)
    assert "endpoint" not in out  # No spurious endpoint block.
    assert out["models"]["items"] == [{"name": "a"}, {"name": "b"}]


def test_endpoint_model_selection_strategy_only_when_model_names_set() -> None:
    """Selection strategy is meaningless without model_names; the converter
    skips emitting it if model_names wasn't also set."""
    user = UserConfig.model_validate(
        {"endpoint": {"model_selection_strategy": "round_robin"}}
    )
    out = build_v1_overrides(user)
    assert "models" not in out


# =====================================================================
# LoadGenerator: multi-run / convergence / sweep mapping table
# =====================================================================
# These map under v2 multi_run.* (some renamed). Single source of truth is
# the `mapping` dict in ``_converter_optionals.build_multi_run``.

_MULTI_RUN_FLAG_TABLE: list[tuple[str, Any, str, Any]] = [
    # --num-profile-runs → multi_run.num_runs (renamed)
    ("loadgen.num_profile_runs", 3, "multi_run.num_runs", 3),
    # --profile-run-cooldown-seconds → multi_run.cooldown_seconds (renamed)
    (
        "loadgen.profile_run_cooldown_seconds",
        5.0,
        "multi_run.cooldown_seconds",
        5.0,
    ),
    # --confidence-level → multi_run.confidence_level
    ("loadgen.confidence_level", 0.95, "multi_run.confidence_level", 0.95),
    # --profile-run-disable-warmup-after-first → multi_run.disable_warmup_after_first
    (
        "loadgen.profile_run_disable_warmup_after_first",
        False,
        "multi_run.disable_warmup_after_first",
        False,
    ),
    # --set-consistent-seed → multi_run.set_consistent_seed
    ("loadgen.set_consistent_seed", True, "multi_run.set_consistent_seed", True),
    # --convergence-metric → multi_run.convergence_metric (requires num_profile_runs)
    # We need to set num_profile_runs along with it for the validator path.
]


@pytest.mark.parametrize(
    "v1_path,v1_value,v2_path,expected",
    [param(*row, id=row[0]) for row in _MULTI_RUN_FLAG_TABLE],
)
def test_multi_run_flag_round_trip(
    v1_path: str, v1_value: Any, v2_path: str, expected: Any
) -> None:
    """multi-run flags map to v2 multi_run.* with the renames documented in
    `_converter_optionals.build_multi_run`'s mapping dict."""
    section, field = v1_path.split(".", 1)
    user = UserConfig.model_validate({section: {field: v1_value}})
    out = build_v1_overrides(user)
    assert _get_at_path(out, v2_path) == expected


def test_convergence_metric_emits_on_multi_run_block() -> None:
    """--convergence-metric requires --num-profile-runs > 1 to be valid; it
    still emits to multi_run.convergence_metric in the override (downstream
    AIPerfConfig validation enforces the cross-field rule)."""
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "convergence_metric": "request_latency",
                "convergence_threshold": 0.05,
                "num_profile_runs": 3,
            }
        }
    )
    out = build_v1_overrides(user)
    assert out["multi_run"]["convergence_metric"] == "request_latency"
    assert out["multi_run"]["convergence_threshold"] == 0.05


def test_parameter_sweep_mode_renames_to_mode() -> None:
    """`parameter_sweep_mode` is one of the few CLI flags that gets a
    name-shortening rewrite (-> ``multi_run.mode``)."""
    user = UserConfig.model_validate(
        {"loadgen": {"parameter_sweep_mode": "independent"}}
    )
    out = build_v1_overrides(user)
    assert out["multi_run"]["mode"] == "independent"


# =====================================================================
# Tokenizer flags
# =====================================================================
#
# Tokenizer flags are 1:1 mapped (no renames) by `build_tokenizer`.

_TOKENIZER_FLAG_TABLE: list[tuple[str, Any, str, Any]] = [
    ("tokenizer.name", "Qwen/Qwen3-0.6B", "tokenizer.name", "Qwen/Qwen3-0.6B"),
    ("tokenizer.revision", "main", "tokenizer.revision", "main"),
    (
        "tokenizer.trust_remote_code",
        True,
        "tokenizer.trust_remote_code",
        True,
    ),
]


@pytest.mark.parametrize(
    "v1_path,v1_value,v2_path,expected",
    [param(*row, id=row[0]) for row in _TOKENIZER_FLAG_TABLE],
)
def test_tokenizer_flag_round_trip(
    v1_path: str, v1_value: Any, v2_path: str, expected: Any
) -> None:
    section, field = v1_path.split(".", 1)
    user = UserConfig.model_validate({section: {field: v1_value}})
    out = build_v1_overrides(user)
    assert _get_at_path(out, v2_path) == expected


# =====================================================================
# Accuracy flags
# =====================================================================

_ACCURACY_FLAG_TABLE: list[tuple[str, Any, str, Any]] = [
    ("accuracy.benchmark", "mmlu", "accuracy.benchmark", "mmlu"),
    ("accuracy.tasks", ["t1", "t2"], "accuracy.tasks", ["t1", "t2"]),
    ("accuracy.n_shots", 5, "accuracy.n_shots", 5),
    ("accuracy.enable_cot", True, "accuracy.enable_cot", True),
    ("accuracy.grader", "exact_match", "accuracy.grader", "exact_match"),
    ("accuracy.system_prompt", "be precise", "accuracy.system_prompt", "be precise"),
    ("accuracy.verbose", True, "accuracy.verbose", True),
]


@pytest.mark.parametrize(
    "v1_path,v1_value,v2_path,expected",
    [param(*row, id=row[0]) for row in _ACCURACY_FLAG_TABLE],
)
def test_accuracy_flag_round_trip(
    v1_path: str, v1_value: Any, v2_path: str, expected: Any
) -> None:
    section, field = v1_path.split(".", 1)
    user = UserConfig.model_validate({section: {field: v1_value}})
    out = build_v1_overrides(user)
    assert _get_at_path(out, v2_path) == expected


# =====================================================================
# Output / artifacts flags
# =====================================================================
#
# v1 OutputConfig fields fold into v2 `artifacts`; some get renamed.


def test_output_artifact_directory_lands_on_artifacts_dir() -> None:
    user = UserConfig.model_validate(
        {"output": {"artifact_directory": "/tmp/artifacts-test"}}
    )
    out = build_v1_overrides(user)
    # build_artifacts writes the value as a Path; cast for compare.
    assert str(out["artifacts"]["dir"]).endswith("artifacts-test")


def test_output_export_level_emits_records_format_for_records_level() -> None:
    """`--export-level records` upgrades the records list to include csv
    (not just the default jsonl). Locks in the conditional emission in
    build_artifacts."""
    user = UserConfig.model_validate({"output": {"export_level": "records"}})
    out = build_v1_overrides(user)
    assert "records" in out["artifacts"]
    assert "csv" in out["artifacts"]["records"]
    assert "jsonl" in out["artifacts"]["records"]


def test_output_slice_duration_passes_through(self_=None) -> None:
    user = UserConfig.model_validate({"output": {"slice_duration": 5.0}})
    out = build_v1_overrides(user)
    assert out["artifacts"]["slice_duration"] == 5.0


def test_output_export_http_trace_passes_through() -> None:
    user = UserConfig.model_validate({"output": {"export_http_trace": True}})
    out = build_v1_overrides(user)
    assert out["artifacts"]["trace"] is True


def test_output_export_per_chunk_data_passes_through() -> None:
    user = UserConfig.model_validate({"output": {"export_per_chunk_data": True}})
    out = build_v1_overrides(user)
    assert out["artifacts"]["per_chunk_data"] is True


def test_output_show_trace_timing_passes_through() -> None:
    user = UserConfig.model_validate({"output": {"show_trace_timing": True}})
    out = build_v1_overrides(user)
    assert out["artifacts"]["show_trace_timing"] is True


# =====================================================================
# ServiceConfig flags (--ui, --log-level, --verbose, ...)
# =====================================================================
#
# These flow through build_logging_runtime; the resolver passes service
# alongside user.


def test_service_log_level_lands_on_logging_block() -> None:
    user = UserConfig()
    service = ServiceConfig.model_validate({"log_level": "DEBUG"})
    out = build_v1_overrides(user, service)
    assert out["logging"]["level"] == "DEBUG"


def test_service_ui_type_lands_on_runtime_ui() -> None:
    user = UserConfig()
    service = ServiceConfig.model_validate({"ui_type": "simple"})
    out = build_v1_overrides(user, service)
    assert out["runtime"]["ui"] == "simple"


def test_service_api_port_lands_on_runtime_api_port() -> None:
    user = UserConfig()
    service = ServiceConfig.model_validate({"api_port": 19090})
    out = build_v1_overrides(user, service)
    assert out["runtime"]["api_port"] == 19090


def test_service_verbose_promotes_log_level_to_debug() -> None:
    """`--verbose` is a derived effect: the converter promotes the log
    level even when the user didn't set --log-level. Locks in the
    `_apply_verbosity_and_ui` policy."""
    user = UserConfig()
    service = ServiceConfig.model_validate({"verbose": True})
    out = build_v1_overrides(user, service)
    assert out["logging"]["level"] == "DEBUG"


# =====================================================================
# No-spurious-overrides invariant
# =====================================================================
#
# Setting one flag in a single section MUST NOT cause sibling sections to
# appear in the override dict. Otherwise a YAML+CLI deep_merge would
# clobber every untouched section.

_SINGLE_FLAG_NON_LEAKAGE: list[tuple[str, Any, set[str]]] = [
    # (v1_path, v1_value, expected_top_level_keys_set)
    ("endpoint.streaming", True, {"endpoint"}),
    ("endpoint.model_names", ["m"], {"models"}),
    ("loadgen.num_profile_runs", 2, {"multi_run"}),
    ("tokenizer.name", "Qwen/Qwen3-0.6B", {"tokenizer"}),
    ("accuracy.benchmark", "mmlu", {"accuracy"}),
    ("output.artifact_directory", "/tmp/out-x", {"artifacts"}),
]


@pytest.mark.parametrize(
    "v1_path,v1_value,expected_top_level",
    [
        param(*row, id=f"only_{row[0]}_emits_{','.join(sorted(row[2]))}")
        for row in _SINGLE_FLAG_NON_LEAKAGE
    ],
)
def test_single_flag_does_not_leak_into_other_sections(
    v1_path: str, v1_value: Any, expected_top_level: set[str]
) -> None:
    """Each row says: setting JUST this v1 flag must produce overrides
    whose top-level keys are exactly ``expected_top_level``. Any extra key
    indicates a section-builder is no longer gating on
    ``model_fields_set`` and is leaking spurious defaults into the
    deep_merge -- which would silently clobber the YAML."""
    section, field = v1_path.split(".", 1)
    user = UserConfig.model_validate({section: {field: v1_value}})
    out = build_v1_overrides(user)
    assert set(out.keys()) == expected_top_level, (
        f"Unexpected leakage: {set(out.keys()) - expected_top_level}"
    )


def test_completely_default_user_config_emits_no_overrides() -> None:
    """The bedrock invariant: a pristine UserConfig with no CLI flags must
    emit an empty override dict so that ``aiperf profile -f base.yaml``
    leaves the YAML 100% intact."""
    out = build_v1_overrides(UserConfig())
    assert out == {}


def test_default_service_alone_emits_no_logging() -> None:
    """ServiceConfig() in a non-TTY emits ui_type via the TTY-based default
    (UIType.NONE), so a `runtime` block with `ui` will appear. Logging is
    NOT auto-derived without --verbose / --log-level, so the logging block
    must stay absent on a pristine ServiceConfig."""
    out = build_v1_overrides(UserConfig(), ServiceConfig())
    assert "logging" not in out


# =====================================================================
# Recipe-input flag round-trips (sla / threshold / isl-min/max / etc.)
# =====================================================================
#
# Recipe inputs feed `expand_search_recipe(user)` and surface inside
# adaptive_search.sla_filters or sweep.variables -- not as raw multi_run
# fields. Verify each input lands at its expected indirect path.


def test_ttft_sla_ms_drives_recipe_threshold() -> None:
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {
                "search_recipe": "max-throughput-ttft-sla",
                "ttft_sla_ms": 175.0,
            },
        }
    )
    out = build_v1_overrides(user)
    sla = out["multi_run"]["adaptive_search"]["sla_filters"][0]
    assert sla["threshold"] == 175.0
    assert sla["metric_tag"] == "time_to_first_token"


def test_itl_sla_ms_drives_recipe_threshold() -> None:
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {
                "search_recipe": "max-throughput-itl-sla",
                "itl_sla_ms": 42.0,
            },
        }
    )
    out = build_v1_overrides(user)
    sla = out["multi_run"]["adaptive_search"]["sla_filters"][0]
    assert sla["threshold"] == 42.0
    assert sla["metric_tag"] == "inter_token_latency"


def test_isl_min_max_drive_prefill_recipe_grid() -> None:
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {
                "search_recipe": "prefill-ttft-curve",
                "isl_min": 256,
                "isl_max": 1024,
            },
        }
    )
    out = build_v1_overrides(user)
    isl_values = out["sweep"]["variables"]["datasets.main.prompts.isl"]
    assert min(isl_values) == 256
    assert max(isl_values) == 1024


def test_degradation_threshold_propagates_to_post_process_params() -> None:
    """The grid concurrency-ramp recipe's --degradation-threshold drives
    its post_process handler params (consumed by
    aggregate_sweep_and_export, in-process only). Verify it reaches the
    converter output even though K8s strips it later."""
    user = UserConfig.model_validate(
        {
            "loadgen": {
                "search_recipe": "concurrency-ramp",
                "degradation_threshold": 0.35,
            }
        }
    )
    out = build_v1_overrides(user)
    pp = out["multi_run"]["post_process"]
    assert pp["params"]["threshold_pct"] == 0.35
