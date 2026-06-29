# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig -> profiling phase dict.

Reads load-generator fields, dataset/schedule fields, and session-turn count
directly off the flat ``CLIConfig``.

Each entry in ``_PROF_FIELD_ROUTES`` declares ``(output_key, attr_name)``
where ``attr_name`` is a top-level field on ``CLIConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.orchestrator.search_planner.parsing import parse_sla_filter

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config.flags import CLIConfig


# (output_key, attr_name) — attr_name is the top-level field on CLIConfig
# whose user-set status we test via ``cli.model_fields_set``.
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
    ("adaptive_scale", "adaptive_scale"),
    ("adaptive_sustain_duration", "adaptive_sustain_duration"),
    ("adaptive_assessment_period", "adaptive_assessment_period"),
)


# Routes whose output keys only exist on GammaPhase. Routed here only when the
# resolved phase type is GAMMA; otherwise the user-supplied value is rejected
# with a clear error rather than crashing PhaseConfig with extra_forbidden.
_GAMMA_ONLY_ROUTES: tuple[tuple[str, str], ...] = (
    ("smoothness", "arrival_smoothness"),
)


# Routes whose output keys only exist on FixedSchedulePhase. Routed here only
# when the resolved phase type is FIXED_SCHEDULE; otherwise we fail loud
# instead of silently dropping the offsets the user passed.
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

    if cli.adaptive_scale:
        return PhaseType.CONCURRENCY
    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE
    if cli.user_centric_rate is not None:
        return PhaseType.USER_CENTRIC
    if cli.request_rate is not None:
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


def _apply_adaptive_scale_sla(prof: dict[str, Any], cli: CLIConfig) -> None:
    if not prof.get("adaptive_scale"):
        return
    if "adaptive_scale_sla" not in cli.model_fields_set or not cli.adaptive_scale_sla:
        return

    prof["sla"] = [
        parse_sla_filter(value).model_dump(mode="json")
        for value in cli.adaptive_scale_sla
    ]


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

    # --num-users only makes sense with --user-centric-rate. Without
    # user-centric mode the resolved phase has no ``users`` field, so
    # routing it through would crash PhaseConfig with extra_forbidden.
    if prof.get("adaptive_scale") and phase_type != PhaseType.CONCURRENCY:
        raise ValueError("--adaptive-scale requires concurrency timing mode")
    if prof.get("adaptive_scale") and "concurrency" not in prof:
        raise ValueError("--adaptive-scale requires --concurrency")
    if (
        prof.get("adaptive_scale")
        and "search_sla" in fields_set
        and "adaptive_scale_sla" not in fields_set
    ):
        raise ValueError(
            "--adaptive-scale uses --adaptive-scale-sla; --search-sla is reserved "
            "for adaptive-search/grid runs"
        )

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
    """CLI-only port of the v1 dataset-aware autodefaults.

    Three behaviors, all conditional on the user's CLI invocation supplying
    a dataset file (no behavior change for YAML-only configs, which are
    expected to be complete):

    1. Trace auto-promotion: a trace ``--custom-dataset-type`` whose first
       record carries a ``timestamp`` field flips the phase to
       fixed_schedule unless the user passed ``--no-fixed-schedule``.
    2. fixed_schedule autodefault: when a fixed_schedule phase has no
       stop condition, fill ``requests`` from the dataset record count
       (single-pass).
    3. Forking-dataset autodefault: when the dataset is ``dag_jsonl`` and
       no stop condition is set, fill ``sessions`` from the DAG root
       count so the run executes each root once instead of truncating
       mid-tree.

    Bare-string ``--custom-dataset-type`` (no ``--input-file``) is a no-op
    for I/O-dependent steps.
    """

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


def _first_record_has_timestamp(file_path: object) -> bool:
    """Return True when the first non-empty JSONL record carries a timestamp."""
    from pathlib import Path

    from aiperf.common.utils import load_json_str

    path = Path(file_path)
    if not path.is_file():
        return False
    try:
        with open(path) as f:
            for line in f:
                if not (stripped := line.strip()):
                    continue
                try:
                    data = load_json_str(stripped)
                except (ValueError, TypeError):
                    return False
                return data.get("timestamp") is not None
    except OSError:
        return False
    return False


def _count_dataset_records(file_path: object) -> int:
    """Count non-empty lines across a JSONL file or directory of JSONLs."""
    from pathlib import Path

    path = Path(file_path)
    try:
        if path.is_dir():
            total = 0
            for jsonl in path.rglob("*.jsonl"):
                with open(jsonl) as f:
                    total += sum(1 for line in f if line.strip())
            return total
        if path.is_file():
            with open(path) as f:
                return sum(1 for line in f if line.strip())
    except OSError:
        return 0
    return 0


def build_profiling(cli: CLIConfig) -> dict[str, Any]:
    """Produce the canonical profiling-phase dict from ``cli``.

    Reads load-generator settings (concurrency, rate, ramps, cancellation),
    schedule/replay flags, and session-turn count directly from ``cli``
    (all fields are top-level post-Task-13). Returns a dict whose ``type``
    is one of ``PhaseType.{CONCURRENCY, POISSON, GAMMA, CONSTANT,
    USER_CENTRIC, FIXED_SCHEDULE}`` plus the keys mapped by
    ``_PROF_FIELD_ROUTES`` and any ramp/cancellation sub-dicts.

    Raises:
        ValueError: when USER_CENTRIC mode is selected but
            ``conversation_turn_mean`` is < 2.
    """
    from aiperf.config.phases import PhaseType

    fields_set = cli.model_fields_set
    prof: dict[str, Any] = {}
    for output_key, attr_name in _PROF_FIELD_ROUTES:
        if attr_name in fields_set:
            prof[output_key] = getattr(cli, attr_name)

    _apply_profiling_ramps(prof, cli)

    prof["type"] = _profiling_phase_type(cli)
    _apply_adaptive_scale_sla(prof, cli)

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
