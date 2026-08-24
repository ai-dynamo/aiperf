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
    {"rate", "rate_ramp", "rate_series", "smoothness"}
)


def _search_space_dimensions(cli: CLIConfig) -> dict[str, float]:
    """Field name -> lower bound for every dimension in ``--search-space``.

    A lightweight companion to ``parse_search_space()`` (which runs later,
    in ``_build_adaptive_search``): only extracts each dimension's final
    path segment (resolving bare-name aliases the same way the real parser
    does) and its lower bound, so ``_profiling_phase_type``/``build_profiling``
    can pick a compatible phase shape -- and seed a self-supplying field's
    initial value from its own search range -- before search-space
    bounds/kind are even validated. Malformed entries are skipped here; the
    real parser reports the actual grammar error. Only dimensions that
    target ``phases.profiling.*`` are considered (bare aliases always
    resolve there); a fully-qualified path targeting a different phase
    (e.g. ``phases.warmup.*``) is ignored by this shape-inference helper --
    the real search-space parser still processes it normally later, just
    not for the purpose of phase-shape inference.
    """
    if not cli.search_space:
        return {}

    from aiperf.config.loader.dotted_path import _resolve_path_alias

    dims: dict[str, float] = {}
    for raw in cli.search_space:
        path_part, _, bounds_part = raw.partition(":")
        path = path_part.strip()
        if not path:
            continue
        if "." not in path:
            path = _resolve_path_alias(path)
        if not path.startswith("phases.profiling."):
            continue
        field = path.rsplit(".", 1)[-1]
        lo_str = bounds_part.split(",", 1)[0].strip()
        try:
            dims[field] = float(lo_str)
        except ValueError:
            continue
    return dims


def _profiling_phase_type(cli: CLIConfig, search_dims: dict[str, float]) -> Any:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE

    user_centric_needed = "users" in search_dims
    rate_shape_needed = bool(set(search_dims) & _RATE_SHAPE_SEARCH_FIELDS)

    # UserCentricPhase has no 'smoothness' field and GammaPhase has no
    # 'users' field -- no phase type can satisfy both, unlike users+rate
    # (UserCentricPhase legitimately has rate/rate_ramp/rate_series too,
    # inherited from RatePhaseConfig).
    if user_centric_needed and "smoothness" in search_dims:
        raise ValueError(
            "--search-space targets both 'users' (a user-centric-shaped "
            "benchmark) and 'smoothness' (a gamma-shaped benchmark). A "
            "benchmark can only have one shape at a time -- search these in "
            "separate runs."
        )

    if cli.user_centric_rate is not None or user_centric_needed:
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

    By construction this is a no-op whenever the phase type came from an
    explicit CLI flag rather than search-space inference: --request-rate,
    --request-rate-series, and --user-centric-rate all populate
    prof["rate"]/prof["rate_series"] via _PROF_FIELD_ROUTES /
    _apply_profiling_rate_series before this runs.
    """
    from aiperf.config.phases import PhaseType

    phase_type = prof["type"]

    if (
        phase_type == PhaseType.USER_CENTRIC
        and "users" not in prof
        and "users" in search_dims
    ):
        if search_dims["users"] < 1:
            raise ValueError(
                f"--search-space 'users' lower bound must be >= 1 (got "
                f"{search_dims['users']!r}); the number of simulated users "
                "can't be less than one."
            )
        prof["users"] = int(search_dims["users"])

    if (
        phase_type
        in (
            PhaseType.POISSON,
            PhaseType.GAMMA,
            PhaseType.CONSTANT,
            PhaseType.USER_CENTRIC,
        )
        and "rate" not in prof
        and "rate_series" not in prof
    ):
        if "rate" in search_dims:
            if search_dims["rate"] <= 0:
                raise ValueError(
                    f"--search-space 'rate' lower bound must be > 0 (got "
                    f"{search_dims['rate']!r}); rate must be positive."
                )
            prof["rate"] = search_dims["rate"]
        else:
            raise ValueError(
                f"--search-space selects a rate-shaped benchmark (phase type "
                f"{phase_type}), which also requires a base rate. Pass "
                "--request-rate <value> (or --user-centric-rate for a "
                "user-shaped benchmark), or add a 'rate' dimension to "
                "--search-space."
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
        # Why: when no bound is given for an unbounded run, default to
        # 10 requests so the run terminates in a reasonable time.
        # Deliberate override of the PhaseConfig default (which would
        # leave it unbounded).
        prof.setdefault("requests", 10)
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
        or prof["type"] == PhaseType.FIXED_SCHEDULE
        or not plugins.is_trace_dataset(str(dataset_type))
        or not _first_record_has_timestamp(file_path)
    ):
        return

    # FixedSchedulePhase doesn't accept rate/users/smoothness. If the user
    # explicitly opted into a rate-controlled mode against a timestamped
    # trace, refuse the combo loudly rather than silently dropping their
    # flag — they almost certainly want one or the other, not both.
    conflicts = [k for k in ("rate", "users", "smoothness") if k in prof]
    if conflicts:
        raise ValueError(
            "Trace dataset has per-record timestamps and would be "
            "auto-promoted to fixed_schedule, but the following flags "
            f"are incompatible with fixed_schedule mode: {conflicts}. "
            "Either drop the conflicting flags to enable auto-fixed-"
            "schedule, or pass --no-fixed-schedule to keep your "
            "user-selected timing mode and ignore trace timestamps."
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
    if (
        prof["type"] == PhaseType.FIXED_SCHEDULE
        and "requests" not in prof
        and file_path is not None
    ):
        records = _count_dataset_records(file_path)
        if records > 0:
            prof["requests"] = records

    _maybe_set_dag_root_sessions(prof, cli, file_path)


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
        raise ValueError(
            "--benchmark-grace-period requires --benchmark-duration to be set. "
            "Grace period only applies after a duration-bounded run; drop "
            "--benchmark-grace-period or pass --benchmark-duration as well."
        )

    _validate_profiling(prof, cli)
    return prof
