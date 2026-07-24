# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig -> profiling phase dict."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _profiling_phase_type(cli: CLIConfig) -> Any:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE
    if cli.user_centric_rate is not None:
        return PhaseType.USER_CENTRIC
    if cli.request_rate is not None or cli.request_rate_series is not None:
        match cli.arrival_pattern:
            case ArrivalPattern.GAMMA:
                return PhaseType.GAMMA
            case ArrivalPattern.CONSTANT:
                return PhaseType.CONSTANT
            case _:
                return PhaseType.POISSON
    return PhaseType.CONCURRENCY


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
        and cli.scenario is None
    ):
        # Why: when no bound is given for an unbounded run, default to
        # 10 requests so the run terminates in a reasonable time.
        # Deliberate override of the PhaseConfig default (which would
        # leave it unbounded).
        #
        # Skipped when a ``--scenario`` is active: the scenario owns the
        # benchmark invariants and auto-fills the REAL stop condition (e.g.
        # the profiling ``duration``) at resolution time. Applying a
        # 10-*request* default here would defeat that -- and for a
        # recorded-graph/agentic scenario it is actively wrong: a "request"
        # is a single turn, one weka trace carries hundreds, and the native
        # ``CyclingGraphTraceSource`` reads ``requests`` as a whole-trace
        # static-node budget, so a value below the first trace's node count
        # admits no trace at all and dispatches nothing.
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


def _uses_runner_owned_graph_input(cli: CLIConfig) -> bool:
    """Return whether Rust directly parses the complete authored graph input."""
    return str(cli.custom_dataset_type) in {
        "dag_jsonl",
        "dynamo_trace",
        "weka_trace",
    }


def _apply_dataset_aware_autodefaults(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Apply dataset-sensitive defaults only to Python-owned linear inputs."""

    from aiperf.config.phases import PhaseType

    if _uses_runner_owned_graph_input(cli):
        # Direct graph adapters receive the authored file unchanged. Python
        # must not probe timing, count rows/roots, or derive a stop condition
        # from formats whose complete semantics are owned by Rust.
        return

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


def _first_record_has_timestamp(file_path: object) -> bool:
    """Return True when a trace file carries timestamp data."""
    from pathlib import Path

    from aiperf.common.utils import load_json_str

    path = Path(file_path)
    if not path.is_file():
        return False
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return False

        try:
            return "timestamp_start_unix_ms" in set(pq.read_schema(path).names)
        except (OSError, pa.ArrowException):
            return False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not (stripped := line.strip()):
                    continue
                try:
                    data = load_json_str(stripped)
                except (ValueError, TypeError):
                    return False
                if not isinstance(data, dict):
                    return False
                return data.get("timestamp") is not None
    except OSError:
        return False
    return False


def _count_dataset_records(file_path: object) -> int:
    """Count records across a JSONL file/directory or Parquet trace file."""
    from pathlib import Path

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
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
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

    _apply_profiling_ramps(prof, cli)
    _apply_profiling_rate_series(prof, cli)

    prof["type"] = _profiling_phase_type(cli)
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
