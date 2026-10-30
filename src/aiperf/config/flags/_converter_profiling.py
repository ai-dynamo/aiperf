# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig -> profiling phase dict."""

from __future__ import annotations

import gzip
import zlib
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config.flags import CLIConfig


_PROF_FIELD_ROUTES: tuple[tuple[str, str], ...] = (
    ("duration", "benchmark_duration"),
    ("grace_period", "benchmark_grace_period"),
    ("concurrency", "concurrency"),
    ("prefill_concurrency", "prefill_concurrency"),
    ("requests", "request_count"),
    ("sessions", "conversation_num"),
    ("users", "num_users"),
    ("rate", "request_rate"),
    ("rate", "user_centric_rate"),
)


_GAMMA_ONLY_ROUTES: tuple[tuple[str, str], ...] = (
    ("smoothness", "arrival_smoothness"),
)


_FIXED_SCHEDULE_ONLY_ROUTES: tuple[tuple[str, str], ...] = (
    ("auto_offset", "fixed_schedule_auto_offset"),
    ("start_offset", "fixed_schedule_start_offset"),
    ("end_offset", "fixed_schedule_end_offset"),
)


_RAMP_FIELDS: tuple[tuple[str, str], ...] = (
    ("concurrency_ramp_duration", "concurrency_ramp"),
    ("prefill_concurrency_ramp_duration", "prefill_ramp"),
    ("request_rate_ramp_duration", "rate_ramp"),
)


# AGENTIC_REPLAY phase fields that pass through verbatim onto BasePhaseConfig.
# (output_key == attr_name on CLIConfig.)
_AGENTIC_REPLAY_ROUTES: tuple[str, ...] = (
    "failed_request_threshold",
    "trajectory_start_min_ratio",
    "trajectory_start_max_ratio",
    "burst_phase_starts",
    "system_idle_gap_cap_seconds",
    "agentic_cache_warmup_duration",
    "warmup_requests_per_lane",
    "agentic_warmup_grace_period",
)


def _apply_agentic_replay_fields(phase: dict[str, Any], cli: CLIConfig) -> None:
    """Copy explicitly-set AGENTIC_REPLAY phase fields onto a phase dict.

    These four fields live on ``BasePhaseConfig`` (shared by profiling and
    warmup), so the same helper feeds both converters.
    """
    fields_set = cli.model_fields_set
    for attr in _AGENTIC_REPLAY_ROUTES:
        if attr in fields_set:
            phase[attr] = getattr(cli, attr)
    # v1 parity: under a --scenario, --warmup-grace-period fed the agentic
    # warmup barrier grace (there was no dedicated flag). Route it onto
    # agentic_warmup_grace_period when the dedicated flag is unset; an
    # explicit --agentic-warmup-grace-period wins.
    if (
        cli.scenario is not None
        and "agentic_warmup_grace_period" not in fields_set
        and cli.warmup_grace_period is not None
    ):
        phase["agentic_warmup_grace_period"] = cli.warmup_grace_period


_RATE_SHAPE_SEARCH_FIELDS: frozenset[str] = frozenset(
    {"rate", "rate_ramp", "smoothness"}
)


def _search_space_dimensions(cli: CLIConfig) -> dict[str, float]:
    from aiperf.orchestrator.search_planner.parsing import parse_search_space

    dims: dict[str, float] = {}
    for dimension in parse_search_space(list(cli.search_space or [])):
        segments = dimension.path.split(".")
        if len(segments) == 3 and segments[:2] == ["phases", "profiling"]:
            dims[segments[-1]] = dimension.lo
    return dims


def _profiling_phase_type(cli: CLIConfig, search_dims: dict[str, float]) -> Any:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE

    user_centric_needed = "users" in search_dims
    user_centric_selected = cli.user_centric_rate is not None or user_centric_needed
    rate_shape_needed = bool(set(search_dims) & _RATE_SHAPE_SEARCH_FIELDS)

    # UserCentricPhase has no 'smoothness' field and GammaPhase has no
    # 'users' field -- no phase type can satisfy both, unlike users+rate
    # (UserCentricPhase legitimately has rate/rate_ramp/rate_series too,
    # inherited from RatePhaseConfig). Gated on user_centric_selected (not
    # just user_centric_needed) so an explicit --user-centric-rate without
    # a 'users' dimension is caught too: it still resolves to USER_CENTRIC,
    # which still has no 'smoothness' field for a searched 'smoothness'
    # dimension to land on.
    if user_centric_selected and "smoothness" in search_dims:
        raise ValueError(
            "--search-space targets 'smoothness' (a gamma-shaped benchmark), "
            "but --user-centric-rate (or a 'users' dimension) selects a "
            "user-centric-shaped benchmark. A benchmark can only have one "
            "shape at a time -- search these in separate runs."
        )

    if user_centric_selected:
        return PhaseType.USER_CENTRIC
    if (
        cli.request_rate is not None
        or cli.request_rate_series is not None
        or rate_shape_needed
    ):
        # v1 parity (user_config.py auto-promote): --arrival-smoothness /
        # --vllm-burstiness without an explicit --arrival-pattern resolves to
        # gamma, since smoothness is a gamma-distribution knob. Without this the
        # flag fell through to POISSON and then _apply_phase_specific_routes
        # hard-rejected it ("only supported with gamma") -- a cutover regression
        # that made --vllm-burstiness unusable on its own. A 'smoothness'
        # search-space dimension is the same knob, so it auto-promotes too.
        if "arrival_pattern" not in cli.model_fields_set and (
            cli.arrival_smoothness is not None or "smoothness" in search_dims
        ):
            return PhaseType.GAMMA
        match cli.arrival_pattern:
            case ArrivalPattern.GAMMA:
                return PhaseType.GAMMA
            case ArrivalPattern.CONSTANT:
                return PhaseType.CONSTANT
            case _:
                return PhaseType.POISSON
    return PhaseType.CONCURRENCY


def _apply_search_space_shape_seeds(
    prof: dict[str, Any], search_dims: dict[str, float]
) -> None:
    """Seed the required scalar(s) a search-space-inferred shape still needs.

    ``_profiling_phase_type`` can auto-switch the phase *type* from
    ``--search-space`` alone, but rate-controlled/user-centric phases also
    require a *value* for their defining scalar (``rate``, and for
    user-centric also ``users``) before Pydantic will accept the base
    config -- the search planner only supplies that value once trials
    start. When the searched field is the scalar itself, its own lower
    bound is a natural seed (the planner immediately overrides it on trial
    0 anyway). When it isn't -- e.g. searching 'smoothness' or 'rate_ramp'
    without ever searching or explicitly setting 'rate' -- there's no value
    to seed from; raise a clear error instead of letting Pydantic's
    "rate-controlled phases require rate or rate_series" surface as an
    unexplained crash on the very first config build.

    Dimension bounds and kinds have already been validated by the shared
    search-space model, including dimensions outside the profiling phase.
    """
    from aiperf.config.phases import PhaseType

    phase_type = prof["type"]

    # 'smoothness' only auto-promotes the phase to GAMMA when --arrival-pattern
    # is unset (see the v1-parity comment in _profiling_phase_type); an
    # EXPLICIT non-gamma --arrival-pattern silently wins instead, leaving a
    # searched 'smoothness' dimension with nowhere to land (GammaPhase is the
    # only phase with that field). Mirrors the equivalent gate for the
    # --arrival-smoothness *flag* in _apply_phase_specific_routes below.
    if "smoothness" in search_dims and phase_type != PhaseType.GAMMA:
        raise ValueError(
            "--search-space 'smoothness' is only supported with --arrival-pattern "
            "gamma. Pass --arrival-pattern gamma (or omit --arrival-pattern so "
            "'smoothness' can auto-select it), or drop the 'smoothness' dimension."
        )

    if phase_type == PhaseType.USER_CENTRIC and "users" in search_dims:
        _seed_users_dimension(prof, search_dims)

    if phase_type in (
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
        PhaseType.USER_CENTRIC,
    ):
        _seed_rate_dimension(prof, search_dims, phase_type)


def _seed_users_dimension(prof: dict[str, Any], search_dims: dict[str, float]) -> None:
    if "users" not in prof:
        prof["users"] = int(search_dims["users"])


def _seed_rate_dimension(
    prof: dict[str, Any], search_dims: dict[str, float], phase_type: Any
) -> None:
    """Validate and (if absent) seed prof["rate"] from a 'rate' dimension.

    Raises when the phase needs a base rate that neither an explicit CLI
    flag nor a 'rate' search-space dimension can supply.
    """
    has_base_rate = "rate" in prof or "rate_series" in prof
    if "rate" not in search_dims:
        if not has_base_rate:
            raise ValueError(
                f"--search-space selects a rate-shaped benchmark (phase type "
                f"{phase_type}), which also requires a base rate. Pass "
                "--request-rate <value> (or --user-centric-rate for a "
                "user-shaped benchmark), or add a 'rate' dimension to "
                "--search-space."
            )
        return

    if "rate_series" in prof:
        raise ValueError(
            "--search-space targets 'rate', but --request-rate-series "
            "already supplies a fixed rate schedule for this phase -- "
            "'rate' and 'rate_series' are mutually exclusive on a "
            "rate-controlled phase. Drop --request-rate-series to "
            "search 'rate', or drop the 'rate' dimension to keep the "
            "fixed schedule."
        )
    if not has_base_rate:
        prof["rate"] = search_dims["rate"]


def apply_cancellation(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Route --request-cancellation-rate/-delay onto a phase dict.

    Shared with the YAML+CLI resolver so both paths carry the same shape and
    the same dependency guard.
    """
    delay_set = "request_cancellation_delay" in cli.model_fields_set
    if cli.request_cancellation_rate:
        cancel: dict[str, Any] = {"rate": cli.request_cancellation_rate}
        if delay_set:
            cancel["delay"] = cli.request_cancellation_delay
        prof["cancellation"] = cancel
    elif delay_set:
        # Mirror --arrival-smoothness gating: refuse to silently drop a
        # user-supplied flag whose dependency wasn't met.
        raise ValueError(
            "--request-cancellation-delay requires --request-cancellation-rate "
            "to be set (cancellation is disabled when rate is unset). "
            "Pass --request-cancellation-rate > 0 to enable cancellation, or "
            "drop --request-cancellation-delay."
        )


def _apply_profiling_ramps(prof: dict[str, Any], cli: CLIConfig) -> None:
    fields_set = cli.model_fields_set
    for field, key in _RAMP_FIELDS:
        if field in fields_set:
            prof[key] = {"duration": getattr(cli, field)}


def _apply_profiling_rate_series(prof: dict[str, Any], cli: CLIConfig) -> None:
    if "request_rate_series" not in cli.model_fields_set:
        return
    if "request_rate" in cli.model_fields_set:
        raise ValueError(
            "--request-rate and --request-rate-series are mutually exclusive."
        )
    if cli.user_centric_rate is not None:
        raise ValueError(
            "--request-rate-series is not supported with --user-centric-rate."
        )
    from aiperf.config.rate_series import RateSeriesConfig

    series = RateSeriesConfig(path=str(cli.request_rate_series))
    prof["rate_series"] = series.model_dump(exclude_none=True, exclude={"path"})


def _reject_orphan_load_generator_flags(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Reject CLI flags whose load-generator partner wasn't supplied.

    Mirrors v1's ``validate_unused_options`` for the load-generator group:
    catches mismatches with a targeted message before they surface as
    generic Pydantic ``extra_forbidden`` errors against the resolved
    phase subclass.
    """
    from aiperf.config.phases import PhaseType

    fields_set = cli.model_fields_set
    phase_type = prof["type"]

    if "num_users" in fields_set and phase_type != PhaseType.USER_CENTRIC:
        raise ValueError(
            "--num-users requires --user-centric-rate. Pass --user-centric-rate "
            "to enable user-centric mode, or drop --num-users to use the default "
            "concurrency/rate timing mode."
        )

    # --request-rate-ramp-duration only ramps rate-controlled phases.
    if "rate_ramp" in prof and phase_type not in (
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
        PhaseType.USER_CENTRIC,
    ):
        raise ValueError(
            "--request-rate-ramp-duration can only be used with rate-controlled "
            "scheduling (--request-rate or --user-centric-rate). Pass one of "
            "those to enable rate ramping, or drop --request-rate-ramp-duration."
        )

    if "rate_series" in prof and phase_type not in (
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
    ):
        raise ValueError(
            "--request-rate-series can only be used with rate-controlled scheduling."
        )


def _apply_phase_specific_routes(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Apply routes whose output keys only exist on a specific phase subclass.

    Errors out with a clear message when the user supplied a phase-specific
    flag that doesn't match the resolved phase type, instead of letting the
    flag silently no-op (fixed-schedule offsets) or crash PhaseConfig with
    ``extra_forbidden`` (gamma smoothness).
    """
    from aiperf.config.phases import PhaseType

    phase_type = prof["type"]
    fields_set = cli.model_fields_set

    for output_key, attr_name in _GAMMA_ONLY_ROUTES:
        if attr_name not in fields_set:
            continue
        if phase_type != PhaseType.GAMMA:
            raise ValueError(
                "--arrival-smoothness is only supported with --arrival-pattern gamma. "
                "Pass --arrival-pattern gamma to enable smoothness, or drop "
                "--arrival-smoothness to use the default arrival pattern."
            )
        prof[output_key] = getattr(cli, attr_name)

    for output_key, attr_name in _FIXED_SCHEDULE_ONLY_ROUTES:
        if attr_name not in fields_set:
            continue
        if phase_type != PhaseType.FIXED_SCHEDULE:
            raise ValueError(
                "--fixed-schedule-{auto,start,end}-offset requires --fixed-schedule. "
                "Pass --fixed-schedule with a trace file to enable offsets, or drop "
                "these flags."
            )
        prof[output_key] = getattr(cli, attr_name)


def _detect_cli_magic_sweep(cli: CLIConfig) -> tuple[str, list] | None:
    """Return the first CLI-set magic-list field, or None.

    Mirrors v1's ``loadgen.get_sweep_parameter()`` against
    ``CLIConfig.model_fields_set`` so the converter can refuse sweep-
    incompatible mode combinations (fixed_schedule, trace auto-promote)
    before they propagate into the YAML expansion stage.
    """
    from aiperf.config.sweep.expand import MAGIC_LIST_FIELDS

    for name in cli.model_fields_set:
        if name not in MAGIC_LIST_FIELDS:
            continue
        value = getattr(cli, name, None)
        if isinstance(value, list) and len(value) > 1:
            return (name.replace("_", "-"), value)
    return None


def _validate_profiling(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Reject invalid mode combinations, then apply dataset-aware and stop-condition defaults."""
    from aiperf.config.phases import PhaseType

    # `--conversation-turn-mean` may be a list when used as a magic-list
    # sweep. User-centric mode requires every variation to satisfy
    # turn_mean >= 2, so check the floor of the swept range.
    raw_turn_mean = cli.conversation_turn_mean or 1
    if isinstance(raw_turn_mean, list):
        turn_mean = min(raw_turn_mean) if raw_turn_mean else 1
    else:
        turn_mean = raw_turn_mean
    if prof["type"] == PhaseType.USER_CENTRIC and turn_mean < 2:
        raise ValueError(
            "User-centric rate mode requires --session-turns-mean >= 2. "
            "For single-turn workloads, use --request-rate instead."
        )

    _apply_dataset_aware_autodefaults(prof, cli)

    # After autodefaults so the trace auto-promotion has had its chance to
    # flip phase.type to FIXED_SCHEDULE; refuse the swept-trace combo with
    # a single, targeted error.
    sweep = _detect_cli_magic_sweep(cli)
    if sweep is not None and prof["type"] == PhaseType.FIXED_SCHEDULE:
        param_name, param_values = sweep
        joined = ",".join(map(str, param_values))
        raise ValueError(
            f"Parameter sweeps (e.g., --{param_name} {joined}) cannot be "
            "used with --fixed-schedule mode (including the auto-promotion "
            "of trace datasets with per-record timestamps). Fixed schedule "
            "replays exact timing patterns from the trace, which is "
            "incompatible with varying parameter values. Use a single "
            "parameter value, or pass --no-fixed-schedule to keep your "
            "rate/concurrency mode and ignore the trace timestamps."
        )

    if (
        not any(k in prof for k in ("requests", "duration", "sessions"))
        and prof["type"] != PhaseType.FIXED_SCHEDULE
    ):
        _apply_default_stop_condition(prof, cli)
    apply_cancellation(prof, cli)


_MIN_DEFAULT_REQUESTS = 10
_DEFAULT_REQUESTS_PER_CONCURRENCY_SLOT = 2


def _apply_default_stop_condition(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Bound a CLI run that set no requests, sessions, or duration.

    PhaseConfig itself requires an explicit stop condition (YAML users must
    choose one); this is the CLI-only default.
    """
    from aiperf.config.phases import FULL_PASS_SESSIONS, PhaseType

    # Scenarios lock their own timing mode and user-centric sizing is
    # governed by users, so both keep the flat floor.
    keeps_flat_floor = (
        cli.scenario is not None or prof["type"] == PhaseType.USER_CENTRIC
    )
    if not keeps_flat_floor and _is_full_pass_dataset(cli):
        prof["sessions"] = FULL_PASS_SESSIONS
        _logger.warning(
            "No stop condition given; running one full pass over every "
            f"conversation in {cli.public_dataset or cli.custom_dataset_type}. "
            "Pass --request-count, --num-conversations, or --benchmark-duration "
            "to bound the run instead."
        )
        return

    requests = _MIN_DEFAULT_REQUESTS
    reason = ""
    if prof["type"] == PhaseType.CONCURRENCY and not keeps_flat_floor:
        # Sweeps and searches size every variation for the largest value so
        # the busiest variation still saturates its slots.
        concurrency = prof.get("concurrency") or 1
        label = "concurrency"
        if isinstance(concurrency, list):
            concurrency = max(concurrency)
            label = "largest swept concurrency"
        searched = _search_space_upper_bound(cli, "concurrency")
        if searched is not None:
            concurrency = math.ceil(searched)
            label = "largest searched concurrency"
        requests = max(
            _MIN_DEFAULT_REQUESTS,
            concurrency * _DEFAULT_REQUESTS_PER_CONCURRENCY_SLOT,
        )
        reason = (
            f" (max({_MIN_DEFAULT_REQUESTS}, "
            f"{_DEFAULT_REQUESTS_PER_CONCURRENCY_SLOT} x {label} {concurrency}))"
        )
    prof["requests"] = requests
    _logger.warning(
        f"No stop condition given; defaulting to {requests} requests{reason}. "
        "Pass --request-count, --num-conversations, or --benchmark-duration "
        "to change it."
    )


def _search_space_upper_bound(cli: CLIConfig, field: str) -> float | None:
    """Upper bound of the ``--search-space`` dimension on ``phases.profiling.<field>``."""
    for raw in cli.search_space or []:
        parsed = _parse_shape_dimension(raw)
        if parsed is None or parsed[0] != field:
            continue
        # Grammar is path:lo,hi[:kind]; the real parser reports malformed bounds.
        try:
            return float(raw.split(":")[1].split(",", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _is_full_pass_dataset(cli: CLIConfig) -> bool:
    """Whether the selected dataset is a suite that defaults to one full pass."""
    from aiperf.plugin import plugins

    if cli.public_dataset is not None:
        return plugins.get_public_dataset_loader_metadata(
            str(cli.public_dataset)
        ).full_pass_by_default
    if cli.custom_dataset_type is not None:
        return plugins.get_dataset_loader_metadata(
            str(cli.custom_dataset_type)
        ).full_pass_by_default
    return False


def _maybe_auto_promote_trace(
    prof: dict[str, Any], cli: CLIConfig, file_path: Path | None
) -> None:
    """Flip phase.type to FIXED_SCHEDULE if a trace dataset has timestamps."""
    from aiperf.config.phases import PhaseType
    from aiperf.plugin import plugins

    dataset_type = cli.custom_dataset_type
    if (
        dataset_type is None
        or file_path is None
        or cli.disable_auto_fixed_schedule
        # A --scenario locks its own timing_mode (e.g. agentic_replay), which
        # would immediately conflict with an auto-promoted FIXED_SCHEDULE
        # phase in the scenario validator. v1 parity: only an EXPLICIT
        # --fixed-schedule conflicts with a scenario; the auto-derived
        # promotion is simply skipped so the phase keeps its default shape.
        or cli.scenario is not None
        or _timeline_opt_out_suppresses_promotion(cli, dataset_type)
        or prof["type"] == PhaseType.FIXED_SCHEDULE
        or not plugins.is_trace_dataset(str(dataset_type))
        or not _trace_carries_timing(dataset_type, file_path)
    ):
        return

    # FixedSchedulePhase doesn't accept rate/users/smoothness. If the user
    # explicitly opted into a rate-controlled mode against a timestamped
    # trace, refuse the combo loudly rather than silently dropping their
    # flag — they almost certainly want one or the other, not both. `prof`
    # may hold these because of an explicit CLI flag OR because
    # _apply_search_space_shape_seeds seeded them from a --search-space
    # dimension -- "flags" alone would be unactionable in the latter case
    # (there's no flag to drop), so name both sources.
    conflicts = [k for k in ("rate", "users", "smoothness") if k in prof]
    if conflicts:
        raise ValueError(
            "Trace dataset has per-record timestamps and would be "
            "auto-promoted to fixed_schedule, but the following flags or "
            f"--search-space dimensions are incompatible with "
            f"fixed_schedule mode: {conflicts}. Either drop the "
            "conflicting flags/dimensions to enable auto-fixed-schedule, "
            "or pass --no-fixed-schedule to keep your user-selected timing "
            "mode and ignore trace timestamps."
        )
    prof["type"] = PhaseType.FIXED_SCHEDULE


def _maybe_set_dag_root_sessions(
    prof: dict[str, Any], cli: CLIConfig, file_path: Path | None
) -> None:
    """For dag_jsonl with no stop condition, set ``sessions`` from root count."""
    from aiperf.plugin.enums import CustomDatasetType

    dataset_type = cli.custom_dataset_type
    is_dag = dataset_type is not None and str(dataset_type) == str(
        CustomDatasetType.DAG_JSONL
    )
    if not is_dag or file_path is None:
        return
    if any(k in prof for k in ("requests", "duration", "sessions")):
        return

    from aiperf.config.dataset.resolver import _collect_dag_session_and_fork_ids

    try:
        all_ids, referenced = _collect_dag_session_and_fork_ids(str(file_path))
    except (OSError, FileNotFoundError):
        return
    roots = len(all_ids - referenced)
    if roots > 0:
        prof["sessions"] = roots


def _apply_dataset_aware_autodefaults(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Apply dataset-sensitive CLI defaults for trace/fixed/dag datasets."""

    from aiperf.config.phases import PhaseType

    file_path: Path | None = cli.input_file if cli.input_file is not None else None

    _maybe_auto_promote_trace(prof, cli, file_path)

    # fixed_schedule autodefault: dataset entry count -> requests.
    #
    # Skipped for formats whose record count is not a line count.
    # ``_count_dataset_records`` counts non-blank lines for a non-JSONL file, so
    # a weka trace -- one JSON object per ``.json`` file, holding N requests --
    # yields the file's line count: the same 3-request trace resolves to 1
    # minified, 59 pretty-printed, and None as a directory. That number is not
    # merely cosmetic; the resolved config is exported into
    # ``profile_export_aiperf.json``, so the artifact would claim a bound the
    # run never had. ``PhaseRunner`` overrides the real total from
    # ``metadata.total_turn_count`` for fixed schedule anyway, so leaving it
    # unset is both honest and sufficient.
    if (
        prof["type"] == PhaseType.FIXED_SCHEDULE
        and "requests" not in prof
        and file_path is not None
        and not _line_count_misreports_records(cli.custom_dataset_type)
    ):
        records = _count_dataset_records(file_path)
        if records > 0:
            prof["requests"] = records

    _maybe_set_dag_root_sessions(prof, cli, file_path)


def _line_count_misreports_records(dataset_type: object) -> bool:
    """Whether counting lines would misreport this format's record count.

    A weka trace nests its requests inside a single JSON object per file, so a
    line count measures formatting, not workload.
    """
    from aiperf.plugin.enums import CustomDatasetType

    return str(dataset_type) == str(CustomDatasetType.WEKA_TRACE)


def _columnar_file_has_timestamp(path: Path) -> bool | None:
    """Probe known columnar formats, or return None for another format."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return False

        try:
            return "timestamp_start_unix_ms" in set(pq.read_schema(path).names)
        except (OSError, pa.ArrowException):
            return False
    if suffix in {".arrow", ".ipc"}:
        from aiperf.dataset.loader.baseten_trace import BasetenTraceDatasetLoader

        return BasetenTraceDatasetLoader.can_load(filename=path)
    return None


def _has_timing_events_timestamp(data: dict) -> bool:
    events = data.get("timing_events")
    return bool(
        events
        and isinstance(events, list)
        and isinstance(events[0], dict)
        and events[0].get("timestamp") is not None
    )


def _timeline_opt_out_suppresses_promotion(
    cli: CLIConfig, dataset_type: object
) -> bool:
    """Whether a timing opt-out should block auto-promotion for this format.

    Only for weka_trace. Its loader lets fixed schedule win over
    ``--ignore-trace-delays``, so promoting a run that passed the flag would
    accept it and then quietly ignore it -- the silent no-op CLAUDE.md's
    flag-routing rule exists to prevent.

    Applying it to every format instead is a regression: a 50-row timestamped
    mooncake trace with either flag drops from fixed_schedule/50 to
    concurrency/10, silently replacing most of the workload. Those loaders do
    not suppress the flags under fixed schedule, and ``--use-think-time-only``
    documents no effect on non-Weka loaders, so there is no no-op to prevent
    there -- only workload to lose.
    """
    from aiperf.plugin.enums import CustomDatasetType

    if str(dataset_type) != str(CustomDatasetType.WEKA_TRACE):
        return False
    return bool(cli.ignore_trace_delays or cli.use_think_time_only)


def _trace_carries_timing(dataset_type: object, file_path: object) -> bool:
    """Whether a trace dataset has timing worth auto-promoting on.

    ``_first_record_has_timestamp`` looks for a top-level ``timestamp`` key and
    refuses directories outright. A weka_trace keeps its timing in
    ``requests[].t`` and is documented as a *directory* of files, so neither is
    visible to that probe, and ``docs/tutorials/weka-trace.md`` documents
    auto-promotion as the default for trace datasets.

    Scoped to weka_trace on purpose. The resolver's ``_implicit_timing_types``
    is the natural shared source of truth, and routing through it reads better,
    but it also holds burst_gpt_trace, sagemaker_data_capture, baseten_trace and
    tracelab -- none of which auto-promote today. Promoting them here would
    silently change their default timing mode, and would turn
    ``--custom-dataset-type burst_gpt_trace --request-rate 10`` from a working
    command into a hard error, because the rate/fixed-schedule conflict check
    below only fires once promotion is decided. Widening the set is a
    deliberate compatibility decision per format, not a side effect of this fix.
    """
    from aiperf.plugin.enums import CustomDatasetType

    if str(dataset_type) == str(CustomDatasetType.WEKA_TRACE):
        return True
    return _first_record_has_timestamp(file_path)


def _first_record_has_timestamp(file_path: object) -> bool:
    """Return True when a trace file carries timestamp data."""
    from pathlib import Path

    from aiperf.common.utils import load_json_str, open_text_maybe_gzip

    path = Path(file_path)
    if not path.is_file():
        return False
    if (columnar_result := _columnar_file_has_timestamp(path)) is not None:
        return columnar_result
    try:
        with open_text_maybe_gzip(path) as f:
            for line in f:
                if not (stripped := line.strip()):
                    continue
                try:
                    data = load_json_str(stripped)
                except (ValueError, TypeError):
                    return False
                if not isinstance(data, dict):
                    return False
                return data.get(
                    "timestamp"
                ) is not None or _has_timing_events_timestamp(data)
    except (EOFError, gzip.BadGzipFile, zlib.error) as e:
        _logger.warning(f"Truncated or corrupt gzip in '{file_path}': {e}")
        return False
    except (OSError, UnicodeDecodeError):
        return False
    return False


def _count_dataset_records(file_path: object) -> int:
    """Count records across JSONL, Parquet, or Arrow IPC input."""
    from pathlib import Path

    from aiperf.common.utils import open_text_maybe_gzip

    path = Path(file_path)
    try:
        if path.is_dir():
            total = 0
            for jsonl in path.rglob("*.jsonl"):
                with open(jsonl, encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            return total
        if path.suffix.lower() == ".parquet" and path.is_file():
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError:
                return 0

            try:
                return pq.ParquetFile(path).metadata.num_rows
            except (OSError, pa.ArrowException):
                return 0
        if path.suffix.lower() in {".arrow", ".ipc"} and path.is_file():
            from aiperf.dataset.loader.baseten_trace import (
                count_baseten_records,
            )

            return count_baseten_records(str(path))
        if path.is_file():
            with open_text_maybe_gzip(path) as f:
                return sum(1 for line in f if line.strip())
    except (EOFError, gzip.BadGzipFile, zlib.error) as e:
        _logger.warning(f"Truncated or corrupt gzip in '{file_path}': {e}")
        return 0
    except (OSError, UnicodeDecodeError):
        return 0
    return 0


def build_profiling(cli: CLIConfig) -> dict[str, Any]:
    """Produce the canonical profiling-phase dict from ``cli``."""
    from aiperf.config.phases import PhaseType

    fields_set = cli.model_fields_set
    prof: dict[str, Any] = {}
    for output_key, attr_name in _PROF_FIELD_ROUTES:
        if attr_name in fields_set:
            prof[output_key] = getattr(cli, attr_name)
    if (
        cli.benchmark_duration is not None
        and "benchmark_grace_period" not in fields_set
    ):
        prof["grace_period"] = cli.benchmark_grace_period

    _apply_profiling_ramps(prof, cli)
    _apply_agentic_replay_fields(prof, cli)
    _apply_profiling_rate_series(prof, cli)

    search_dims = _search_space_dimensions(cli)
    prof["type"] = _profiling_phase_type(cli, search_dims)
    _apply_search_space_shape_seeds(prof, search_dims)
    _reject_orphan_load_generator_flags(prof, cli)
    _apply_phase_specific_routes(prof, cli)

    if prof["type"] == PhaseType.FIXED_SCHEDULE and "start_offset" in prof:
        prof.setdefault("auto_offset", False)

    # grace_period is a duration-phase concept (a tail on top of ``duration``);
    # PhaseConfig rejects it without ``duration`` set. Refuse the combination
    # loudly instead of silently dropping, so users discover the mismatch at
    # config time rather than wondering why their cooldown didn't apply.
    if "grace_period" in prof and prof.get("duration") is None:
        # A scenario's auto-filled duration is applied after conversion, so it
        # cannot satisfy this check; tell scenario users how to tune grace.
        scenario_hint = (
            f" Under --scenario {cli.scenario}, pass --benchmark-duration "
            "explicitly (at least the scenario's minimum) to tune the grace period."
            if cli.scenario is not None
            else ""
        )
        raise ValueError(
            "--benchmark-grace-period requires --benchmark-duration to be set. "
            "Grace period only applies after a duration-bounded run; drop "
            "--benchmark-grace-period or pass --benchmark-duration as well."
            + scenario_hint
        )

    _validate_profiling(prof, cli)
    return prof
