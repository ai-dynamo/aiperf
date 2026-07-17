# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.config.dataset.defaults import InputDefaults
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import (
    ArrivalPattern,
    DatasetSamplingStrategy,
    PhaseType,
    TimingMode,
    URLSelectionStrategy,
)
from aiperf.timing.adaptive_config import (
    ADAPTIVE_TIMING_FIELDS,
    AdaptiveControlVariable,
    AdaptiveTimingConfig,
)
from aiperf.timing.request_cancellation import RequestCancellationConfig

if TYPE_CHECKING:
    from aiperf.config.phases import PhaseConfig
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = AIPerfLogger(__name__)


# Map ``PhaseType`` values onto the ``ArrivalPattern`` values consumed by the
# timing strategies. Concurrency / fixed_schedule phases don't use an arrival
# pattern; we still set a sensible default so downstream code paths remain
# uniform when they consult this field.
_PHASE_TYPE_TO_ARRIVAL_PATTERN: dict[PhaseType, ArrivalPattern] = {
    PhaseType.POISSON: ArrivalPattern.POISSON,
    PhaseType.GAMMA: ArrivalPattern.GAMMA,
    PhaseType.CONSTANT: ArrivalPattern.CONSTANT,
    PhaseType.USER_CENTRIC: ArrivalPattern.POISSON,
    PhaseType.CONCURRENCY: ArrivalPattern.CONCURRENCY_BURST,
    PhaseType.FIXED_SCHEDULE: ArrivalPattern.CONCURRENCY_BURST,
}


def _phase_timing_mode(phase: PhaseConfig) -> TimingMode:
    """Map a phase to the timing strategy used for credit issuance."""
    if getattr(phase, "adaptive_scale", False):
        return TimingMode.ADAPTIVE_SCALE
    if phase.type == PhaseType.FIXED_SCHEDULE:
        return TimingMode.FIXED_SCHEDULE
    if phase.type == PhaseType.USER_CENTRIC:
        return TimingMode.USER_CENTRIC_RATE
    return TimingMode.REQUEST_RATE


class TimingConfig(AIPerfBaseModel):
    """Configuration for TimingManager and timing strategies.

    Controls timing mode (REQUEST_RATE, FIXED_SCHEDULE, USER_CENTRIC_RATE,
    ADAPTIVE_SCALE, or GRAPH_IR), rate/concurrency settings, warmup/profiling
    phase stop conditions, and request cancellation behavior.
    """

    model_config = ConfigDict(frozen=True)

    phase_configs: list[CreditPhaseConfig] = Field(
        ...,
        description="List of phase configs to execute in order. These specify the exact behavior of each phase.",
    )
    request_cancellation: RequestCancellationConfig = Field(
        default_factory=RequestCancellationConfig,
        description="Configuration for request cancellation policy.",
    )
    urls: list[str] = Field(
        default_factory=list,
        description="List of endpoint URLs for load balancing. If multiple URLs provided, "
        "requests are distributed according to url_selection_strategy.",
    )
    url_selection_strategy: URLSelectionStrategy = Field(
        default=URLSelectionStrategy.ROUND_ROBIN,
        description="Strategy for selecting URLs when multiple URLs are provided.",
    )

    @classmethod
    def from_run(cls, run: BenchmarkRun) -> TimingConfig:
        """Build ordered list of credit-phase configs from a ``BenchmarkRun``.

        Iterates ``run.cfg.get_warmup_phases()`` first (each becomes a WARMUP
        CreditPhaseConfig) followed by ``run.cfg.get_profiling_phases()``
        (each becomes a PROFILING CreditPhaseConfig). The cancellation policy
        is sourced from the first profiling phase that declares one; URLs and
        url-selection strategy come from the endpoint section.
        """
        cfg = run.cfg

        # Resolved dataset-selection policy (populated by the post-scenario
        # graph-dispatch resolver for graph runs; None otherwise). Carried onto
        # each CreditPhaseConfig so ``GraphIRReplayStrategy`` can consume them;
        # non-graph phases just pass the Nones through.
        dataset_sampling_strategy = run.resolved.dataset_sampling_strategy
        allow_dataset_wrap = run.resolved.allow_dataset_wrap

        # Weka graph IR workloads replay a conversation DAG via
        # ``GraphIRReplayStrategy`` (TimingMode.GRAPH_IR), not the linear
        # per-turn timing modes. Read the memoized single-source resolution so
        # BOTH the warmup and profiling phases select the graph strategy;
        # non-graph workloads are unchanged. An explicit, graph-incompatible
        # ``--custom-dataset-type`` takes PRECEDENCE over the detection -- a
        # user who pinned a different loader is never silently rerouted to the
        # graph pipeline. The veto composes OVER the accessor (behavior this
        # module alone owns), it is not absorbed into it.
        from aiperf.dataset.graph.workload_detect import (
            _resolve_graph_max_context,
            _resolve_graph_num_entries,
            resolve_graph_workload,
        )

        is_graph = resolve_graph_workload(
            run
        ) is not None and not _explicit_non_graph_format(run)

        # Resolved per-trace-instance cache-bust target (scenario-auto-filled on
        # ``endpoint.cache_bust`` before this runs). Threaded onto each graph
        # CreditPhaseConfig so ``GraphIRReplayStrategy``'s dispatch duplication
        # report can decide whether recycle duplication is safe. None for
        # non-graph runs (the strategy is never built there).
        cache_bust = cfg.endpoint.cache_bust if is_graph else None

        # Resolved graph-plane corpus-selection caps, threaded onto each graph
        # CreditPhaseConfig so ``GraphIRReplayStrategy``'s wrap-guard can phrase
        # an over-subscription shortfall as CAPPED (a knob shrank the corpus)
        # rather than EXHAUSTED. REUSE the build plane's OWN resolvers so the
        # dispatch-side caps match the build-side exactly (single source of
        # truth). None for non-graph runs.
        num_dataset_entries = _resolve_graph_num_entries(run) if is_graph else None
        max_context_length = _resolve_graph_max_context(run) if is_graph else None

        # Resolved t* snapshot window + phase-start burst mode, threaded from
        # the run config (--trajectory-start-min/max-ratio, scenario-auto-
        # applied when unset; --burst-phase-starts) onto each graph
        # CreditPhaseConfig so the strategy samples the same window the
        # auto-warmup decision used. None for non-graph runs.
        trajectory_start_min_ratio = (
            (cfg.trajectory_start_min_ratio or 0.0) if is_graph else None
        )
        trajectory_start_max_ratio = (
            (cfg.trajectory_start_max_ratio or 0.0) if is_graph else None
        )
        burst_phase_starts = cfg.burst_phase_starts if is_graph else None

        # Run seed for deterministic per-trace t* sampling (salted per trace/
        # lane inside the strategy) -- the SAME --random-seed that seeds
        # synthesized content, so one seed reproduces the whole run.
        random_seed = run.random_seed

        # Extended (cache-pressure) warmup duration, per-run config from
        # --agentic-cache-warmup-duration. None = no pressure stage.
        cache_pressure_duration = (
            cfg.agentic_cache_warmup_duration if is_graph else None
        )

        artifact_dir = cfg.artifacts.dir

        configs: list[CreditPhaseConfig] = []
        warmup_phases = list(cfg.get_warmup_phases())
        profiling_phases = list(cfg.get_profiling_phases())
        # Explicit graph-incompatible phase choices take precedence over the
        # detection (same rule as --custom-dataset-type above): reject loudly
        # instead of silently rerouting. Checked BEFORE the pressure supersede
        # so a rate-typed user warmup is rejected, not absorbed.
        if is_graph:
            _reject_graph_incompatible_phases(warmup_phases, profiling_phases)
        # Pressure-mode graph warmups are MODE-OWNED (agentx parity: the
        # agentic warmup pins expected_duration_sec=None and its own grace
        # regardless of user warmup settings). A user warmup phase here would
        # bound priming+pressure combined and its count caps would starve the
        # pressure budget, so supersede it loudly with the auto shape -- but
        # carry an EXPLICIT user grace through (agentx honors it verbatim).
        graph_pressure_user_grace: float | None = None
        if is_graph and cache_pressure_duration is not None and warmup_phases:
            graph_pressure_user_grace = next(
                (
                    p.grace_period
                    for p in warmup_phases
                    if getattr(p, "grace_period", None) is not None
                ),
                None,
            )
            _logger.notice(
                "graph cache-pressure warmup supersedes the user-configured "
                f"warmup phase(s) ({len(warmup_phases)}): boundary priming + "
                "the pressure stage own the warmup shape (duration=None, "
                "grace=user grace if set, else min(pressure duration, cap)); "
                "profiling phases are unchanged"
            )
            warmup_phases = []
        for phase in warmup_phases:
            configs.append(
                _build_warmup_config(
                    phase,
                    is_graph=is_graph,
                    artifact_dir=artifact_dir,
                    dataset_sampling_strategy=dataset_sampling_strategy,
                    allow_dataset_wrap=allow_dataset_wrap,
                    cache_bust=cache_bust,
                    num_dataset_entries=num_dataset_entries,
                    max_context_length=max_context_length,
                    trajectory_start_min_ratio=trajectory_start_min_ratio,
                    trajectory_start_max_ratio=trajectory_start_max_ratio,
                    burst_phase_starts=burst_phase_starts,
                    random_seed=random_seed,
                    cache_pressure_duration=cache_pressure_duration,
                )
            )
        # AgentX parity (timing/config.py::_build_warmup_config graph-replay
        # branch): a t*-snapshot weka graph run ALWAYS runs a WARMUP phase to
        # prime each live chain's pre-t* boundary turn into the server KV cache,
        # so the profiled (at/after-t*) turns measure a warm cache -- NOT a cold
        # start. AgentX auto-creates that warmup unconditionally for the agentic
        # replay mode; our graph path only built one when the user passed an
        # explicit --warmup-* trigger, so a default t*>0 run started PROFILING
        # cold (no warmup phase). Inject the auto-warmup when the t* window is
        # active (trajectory_start_max_ratio > 0) and the user supplied no explicit warmup
        # phase. With t*=0 (full native replay) there is no pre-t* prefix to
        # prime, so rewrite_for_warmup returns an empty graph and the phase
        # finalizes immediately -- harmless -- but we still skip it to keep the
        # full-replay phase list byte-identical (one PROFILING phase).
        # The pressure stage (extended warmup) also requires the phase even at t*=0.
        if (
            is_graph
            and not warmup_phases
            and (
                _graph_tstar_active(trajectory_start_max_ratio)
                or cache_pressure_duration is not None
            )
        ):
            auto = _build_graph_auto_warmup_config(
                profiling_phases,
                user_grace=graph_pressure_user_grace,
                dataset_sampling_strategy=dataset_sampling_strategy,
                allow_dataset_wrap=allow_dataset_wrap,
                cache_bust=cache_bust,
                num_dataset_entries=num_dataset_entries,
                max_context_length=max_context_length,
                trajectory_start_min_ratio=trajectory_start_min_ratio,
                trajectory_start_max_ratio=trajectory_start_max_ratio,
                burst_phase_starts=burst_phase_starts,
                random_seed=random_seed,
                cache_pressure_duration=cache_pressure_duration,
            )
            if auto is not None:
                configs.append(auto)
        for phase in profiling_phases:
            configs.append(
                _build_profiling_config(
                    phase,
                    is_graph=is_graph,
                    artifact_dir=artifact_dir,
                    dataset_sampling_strategy=dataset_sampling_strategy,
                    allow_dataset_wrap=allow_dataset_wrap,
                    cache_bust=cache_bust,
                    num_dataset_entries=num_dataset_entries,
                    max_context_length=max_context_length,
                    trajectory_start_min_ratio=trajectory_start_min_ratio,
                    trajectory_start_max_ratio=trajectory_start_max_ratio,
                    burst_phase_starts=burst_phase_starts,
                    random_seed=random_seed,
                    cache_pressure_duration=cache_pressure_duration,
                )
            )

        cancellation_config: RequestCancellationConfig = RequestCancellationConfig()
        for phase in cfg.get_profiling_phases():
            if getattr(phase, "cancellation", None) is not None:
                cancellation_config = RequestCancellationConfig(
                    rate=phase.cancellation.rate,
                    delay=phase.cancellation.delay,
                )
                break

        return cls(
            phase_configs=configs,
            request_cancellation=cancellation_config,
            urls=list(cfg.endpoint.urls),
            url_selection_strategy=cfg.endpoint.url_strategy,
        )


class CreditPhaseConfig(AIPerfBaseModel):
    """Model for credit phase config. This is used to configure a credit phase.

    Stop conditions (first one reached wins):
    - total_expected_requests: Stop after sending this many total requests
    - expected_num_sessions: Stop starting NEW user sessions after this many (complete ongoing ones)
    - expected_duration_sec: Stop after this time
    """

    model_config = ConfigDict(frozen=True)

    phase: CreditPhase = Field(..., description="The phase of the credit phase.")
    timing_mode: TimingMode = Field(
        ...,
        description="The timing mode of the credit phase. Used to determine "
        "how to send requests to the workers.",
    )
    total_expected_requests: int | None = Field(
        default=None, gt=0, description="The total number of expected requests to send."
    )
    expected_num_sessions: int | None = Field(
        default=None, gt=0, description="The total number of expected sessions to send."
    )
    expected_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="The expected duration of the credit phase in seconds.",
    )
    seamless: bool = Field(
        default=False,
        description="Whether the credit phase should be seamless. "
        "Seamless phases start immediately after the previous phase sends all credits, "
        "without waiting for all credits to return. This can be used to maintain concurrency "
        "during phase transitions.",
    )
    concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max concurrency of the credit phase. "
        "This is the max number of requests that can be in flight at once. "
        "If None, the concurrency is unlimited.",
    )
    prefill_concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max concurrency of the prefill phase. "
        "This is the max number of requests that can be waiting for the first token at once. "
        "If None, the prefill concurrency is unlimited.",
    )
    request_rate: float | None = Field(
        default=None, gt=0, description="The request rate of the credit phase."
    )
    arrival_pattern: ArrivalPattern = Field(
        default=ArrivalPattern.POISSON,
        description="The arrival pattern of the credit phase.",
    )
    arrival_smoothness: float | None = Field(
        default=None,
        gt=0,
        description="The smoothness parameter for gamma distribution arrivals. "
        "Only used when arrival_pattern is GAMMA. Controls the shape of the distribution: "
        "1.0 = Poisson-like (exponential), <1.0 = bursty, >1.0 = smooth/regular. "
        "If None, defaults to 1.0 when using GAMMA arrival pattern.",
    )
    grace_period_sec: float | None = Field(
        default=None,
        ge=0,
        description="The grace period of the credit phase in seconds. "
        "This is the time to wait after the expected duration of the phase has elapsed "
        "before the phase is considered complete. This can be used to ensure that all requests "
        "have returned before the phase is considered complete. "
        "If None, the grace period is disabled.",
    )
    num_users: int | None = Field(
        default=None,
        ge=1,
        description="The number of concurrent users to use for the credit phase. "
        "This is only applicable when using user-centric rate limiting mode. ",
    )
    concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp session concurrency from 1 to target. "
        "If None, concurrency starts at target immediately.",
    )
    prefill_concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp prefill concurrency from 1 to target. "
        "If None, prefill concurrency starts at target immediately.",
    )
    request_rate_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp request rate from 1 QPS to target. "
        "If None, request rate starts at target immediately.",
    )
    auto_offset_timestamps: bool = Field(
        default=InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET,
        description="The auto offset timestamps of the timing manager.",
    )
    fixed_schedule_start_offset: int | None = Field(
        default=None,
        ge=0,
        description="The fixed schedule start offset of the timing manager.",
    )
    fixed_schedule_end_offset: int | None = Field(
        default=None,
        ge=0,
        description="The fixed schedule end offset of the timing manager.",
    )

    artifact_dir: Path | None = Field(
        default=None,
        description="Directory for phase-owned timing artifacts.",
    )
    adaptive: AdaptiveTimingConfig = Field(
        default_factory=AdaptiveTimingConfig,
        description="Adaptive scale timing settings.",
    )
    dataset_sampling_strategy: DatasetSamplingStrategy | None = Field(
        default=None,
        description="Resolved run-level dataset sampling strategy for graph "
        "workloads, carried from `run.resolved.dataset_sampling_strategy` so the "
        "GraphIRReplayStrategy can consume it. None for non-graph phases and "
        "until resolution derives it.",
    )
    allow_dataset_wrap: bool | None = Field(
        default=None,
        description="Resolved graph-plane dataset-wrap policy, carried from "
        "`run.resolved.allow_dataset_wrap` so the GraphIRReplayStrategy can "
        "consume it. None for non-graph phases and until resolution derives it.",
    )
    cache_bust: CacheBustTarget | None = Field(
        default=None,
        description="Resolved per-trace-instance cache-bust target, carried from "
        "`endpoint.cache_bust` so the GraphIRReplayStrategy's dispatch "
        "duplication report can decide whether recycle duplication is safe "
        "(cache-bust ON) or warns (OFF). None for non-graph phases.",
    )
    num_dataset_entries: int | None = Field(
        default=None,
        ge=1,
        description="Resolved explicit --num-dataset-entries corpus cap for graph "
        "workloads (the run's default-dataset `entries`), carried so the "
        "GraphIRReplayStrategy wrap-guard phrases an over-subscription shortfall "
        "as CAPPED rather than EXHAUSTED. None for non-graph phases and when unset.",
    )
    max_context_length: int | None = Field(
        default=None,
        ge=1,
        description="Resolved --max-context-length per-trace context cap for graph "
        "workloads (`synthesis.max_context_length`), carried so the "
        "GraphIRReplayStrategy wrap-guard phrases an over-subscription shortfall "
        "as CAPPED rather than EXHAUSTED. None for non-graph phases and when unset.",
    )
    trajectory_start_min_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Resolved t* snapshot-window lower bound for graph "
        "workloads, carried from `cfg.trajectory_start_min_ratio` "
        "(`--trajectory-start-min-ratio`, scenario-auto-applied when unset) so "
        "the GraphIRReplayStrategy samples the same window the auto-warmup "
        "decision used. None for non-graph phases; unset resolves to 0.0.",
    )
    trajectory_start_max_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Resolved t* snapshot-window upper bound for graph "
        "workloads, carried from `cfg.trajectory_start_max_ratio` "
        "(`--trajectory-start-max-ratio`, scenario-auto-applied when unset). "
        "None for non-graph phases; unset resolves to 0.0 (window OFF).",
    )
    burst_phase_starts: bool | None = Field(
        default=None,
        description="Resolved --burst-phase-starts phase-start dispatch mode "
        "for graph workloads, carried from `cfg.burst_phase_starts`. "
        "None for non-graph phases.",
    )
    random_seed: int | None = Field(
        default=None,
        ge=0,
        description="The run's resolved --random-seed, threaded so the graph "
        "strategy's t* sampling derives from the SAME seed as synthesized "
        "content (one seed reproduces the whole run; sweep cells decorrelate "
        "via the orchestrator's per-variation seed derivation).",
    )
    cache_pressure_duration: float | None = Field(
        default=None,
        gt=0,
        description="Extended (cache-pressure) warmup duration in seconds, "
        "carried from `cfg.agentic_cache_warmup_duration` "
        "(--agentic-cache-warmup-duration). None = no pressure stage; also "
        "None for non-graph phases.",
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_adaptive_timing_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        folded = dict(data)
        adaptive = dict(folded.get("adaptive") or {})
        for field in ADAPTIVE_TIMING_FIELDS:
            if field in folded:
                adaptive[field] = folded.pop(field)
        if adaptive:
            folded["adaptive"] = adaptive
        return folded

    def model_copy(
        self, *, update: dict[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update:
            update = self._fold_adaptive_update(update)
        return super().model_copy(update=update, deep=deep)

    def _fold_adaptive_update(self, update: dict[str, Any]) -> dict[str, Any]:
        folded = dict(update)
        adaptive_update = {
            field: folded.pop(field)
            for field in list(folded)
            if field in ADAPTIVE_TIMING_FIELDS
        }
        if adaptive_update:
            adaptive_payload = self.adaptive.model_dump(mode="python")
            adaptive_payload.update(adaptive_update)
            folded["adaptive"] = AdaptiveTimingConfig.model_validate(adaptive_payload)
        return folded

    @property
    def adaptive_sustain_duration_sec(self) -> float | None:
        return self.adaptive.adaptive_sustain_duration_sec

    @property
    def adaptive_assessment_period_sec(self) -> float:
        return self.adaptive.adaptive_assessment_period_sec

    @property
    def adaptive_control_variable(self) -> AdaptiveControlVariable:
        return self.adaptive.adaptive_control_variable

    @property
    def adaptive_control_min(self) -> float:
        return self.adaptive.adaptive_control_min

    @property
    def adaptive_control_max(self) -> float | None:
        return self.adaptive.adaptive_control_max

    @property
    def adaptive_scale_strategy_type(self) -> Literal["ramp_until_fail"]:
        return self.adaptive.adaptive_scale_strategy_type

    @property
    def adaptive_scale_step_policy(self) -> Literal["sla_margin", "fixed_percent_step"]:
        return self.adaptive.adaptive_scale_step_policy

    @property
    def adaptive_scale_base_step(self) -> int:
        return self.adaptive.adaptive_scale_base_step

    @property
    def adaptive_scale_max_step_multiplier(self) -> int:
        return self.adaptive.adaptive_scale_max_step_multiplier

    @property
    def adaptive_scale_step_percent(self) -> float:
        return self.adaptive.adaptive_scale_step_percent

    @property
    def adaptive_min_completed_requests(self) -> int:
        return self.adaptive.adaptive_min_completed_requests

    @property
    def adaptive_sla_filters(self) -> tuple[SLAFilter, ...]:
        return self.adaptive.adaptive_sla_filters


def _ramp_duration(ramp: object | None) -> float | None:
    """Extract the ramp duration in seconds from a ``RamperConfig`` (or None)."""
    if ramp is None:
        return None
    return getattr(ramp, "duration", None)


def _explicit_non_graph_format(run: BenchmarkRun) -> bool:
    """True when the run pins an explicit, graph-incompatible dataset format.

    The weka graph workload is auto-detected by a file-content sniff
    (``workload_detect.resolve_graph_workload``); there is no ``weka`` value in
    ``DatasetFormat``. A bare ``--input-file weka.json`` run resolves to the
    default ``DatasetFormat.SINGLE_TURN``, so we treat SINGLE_TURN as "unset --
    let the sniff decide". Any OTHER resolved format means the user explicitly
    pinned a different loader (``--custom-dataset-type multi_turn`` /
    ``mooncake_trace`` / ...), which must take PRECEDENCE over the sniff -- the
    run is NOT silently rerouted to the graph IR pipeline. Returns False for any
    non-file dataset (no ``format`` attribute) so synthetic / public runs are
    unaffected.
    """
    from aiperf.common.enums import DatasetFormat

    try:
        dataset = run.cfg.get_default_dataset()
    except (IndexError, AttributeError):
        return False
    fmt = getattr(dataset, "format", None)
    if fmt is None:
        return False
    return fmt != DatasetFormat.SINGLE_TURN


def _reject_graph_incompatible_phases(
    warmup_phases: list[PhaseConfig], profiling_phases: list[PhaseConfig]
) -> None:
    """Reject phase configs whose explicit pacing a graph replay would discard.

    The recorded graph replay (TimingMode.GRAPH_IR) owns pacing and
    concurrency, so rerouting these phases to GRAPH_IR would
    silently discard the user's explicit choice:

    * ``adaptive_scale`` profiling phases drive their own concurrency ladder.
    * Rate-controlled and fixed-schedule phase TYPES encode explicit user
      pacing (``--request-rate`` / ``--user-centric-rate`` /
      ``--fixed-schedule``).
    """
    if any(getattr(phase, "adaptive_scale", False) for phase in profiling_phases):
        raise ValueError(
            "adaptive_scale is not supported for graph workloads: the "
            "recorded graph replay (TimingMode.GRAPH_IR) owns pacing and "
            "concurrency. Remove adaptive_scale from the phase config, or "
            "pin a non-graph loader with --custom-dataset-type to run "
            "this input through the linear pipeline."
        )
    for phase in (*warmup_phases, *profiling_phases):
        if phase.type != PhaseType.CONCURRENCY:
            raise ValueError(
                f"phase '{phase.name}' (type={phase.type}) is not "
                "supported for graph workloads: the recorded graph "
                "replay (TimingMode.GRAPH_IR) owns pacing, so "
                "rate-controlled arrivals (--request-rate / "
                "--user-centric-rate) and --fixed-schedule "
                "timestamps would be silently discarded. Remove the "
                "rate/schedule options (--concurrency bounds the "
                "replay lanes), or pin a non-graph loader with "
                "--custom-dataset-type to run this input through "
                "the linear pipeline."
            )


def resolve_graph_content_seed(run: BenchmarkRun) -> int | None:
    """Return the run's seed for Weka content synthesis -- the AIPerf seed.

    Just ``run.random_seed`` (``--random-seed``), the same seed schedule /
    topology / t* derive from. The DatasetManager is the only parser; the
    TimingManager ingests the graph_meta sidecar from the graph-typed dataset
    broadcast. Resolving from the run config alone keeps every parse of the same
    run (in-process or spawn-started pool worker) byte-identical. ``None`` (no
    ``--random-seed``) keeps the ambient global RNG -- there is no weka-specific
    seed fallback.
    """
    return run.random_seed


def resolve_graph_content_tokenizer(run: BenchmarkRun) -> str:
    """Resolve the tokenizer the Weka content synthesizer must use.

    The synthesized message content (block + partial-tail token IDs decoded to
    wire text) is only valid if it is decoded with the SAME tokenizer the run
    dispatches and token-counts against. So this delegates to the ONE existing
    resolver -- ``TokenizerConfig.get_tokenizer_name_for_model`` -- the exact
    call :meth:`DatasetManager._configure_tokenizer` uses to load the dispatch
    tokenizer (CLI-resolved name, else explicit ``--tokenizer``, else the model
    name). No separate resolution logic lives here.

    The DatasetManager is the only parser; the TimingManager ingests the
    graph_meta sidecar from the graph-typed dataset broadcast. Resolving from
    the run config alone keeps every parse of the same run (in-process or
    spawn-started pool worker) byte-identical (the same content contract as
    :func:`resolve_graph_content_seed`).
    """
    cfg = run.cfg
    model_names = cfg.get_model_names()
    model = model_names[0] if model_names else ""
    return cfg.tokenizer.get_tokenizer_name_for_model(model)


def _phase_request_rate(phase: PhaseConfig) -> float | None:
    """Return the configured request rate for a phase, if any."""
    # Lazy import: aiperf.config.phases is only a TYPE_CHECKING import here.
    from aiperf.config.phases import get_phase_rate

    return get_phase_rate(phase)


def _phase_arrival_pattern(phase: PhaseConfig) -> ArrivalPattern:
    """Map a phase type to its arrival pattern."""
    return _PHASE_TYPE_TO_ARRIVAL_PATTERN.get(phase.type, ArrivalPattern.POISSON)


def _build_warmup_config(
    phase: PhaseConfig,
    *,
    is_graph: bool = False,
    artifact_dir: Path | None = None,
    dataset_sampling_strategy: DatasetSamplingStrategy | None = None,
    allow_dataset_wrap: bool | None = None,
    cache_bust: CacheBustTarget | None = None,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    trajectory_start_min_ratio: float | None = None,
    trajectory_start_max_ratio: float | None = None,
    burst_phase_starts: bool | None = None,
    random_seed: int | None = None,
    cache_pressure_duration: float | None = None,
) -> CreditPhaseConfig:
    """Build a warmup CreditPhaseConfig from a warmup PhaseConfig.

    Warmup triggers JIT compilation, memory allocation, and connection pool
    initialization so profiling measurements aren't polluted by cold-start effects.

    When the phase doesn't set ``grace_period``, default to infinity (wait
    forever for in-flight requests). This differs from the CreditPhaseConfig
    field default of None (disabled) because warmup should always complete all
    in-flight requests before transitioning to profiling.

    ``is_graph`` forces ``TimingMode.GRAPH_IR`` so a weka graph workload's WARMUP
    phase runs the graph strategy (re-seeding the boundary prefix via warmup
    materialization over the graph store's profiling bytes) rather than the linear
    ``RequestRateStrategy`` over zero-turn ``Conversation`` stubs -- which would
    send nothing useful (the per-node payloads live in the graph store mmap, not
    the stub conversations). Mirrors ``_build_profiling_config``'s ``is_graph``.
    """
    grace_period = phase.grace_period
    if grace_period is None:
        grace_period = float("inf")

    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        timing_mode=TimingMode.GRAPH_IR if is_graph else TimingMode.REQUEST_RATE,
        total_expected_requests=phase.requests,
        expected_duration_sec=phase.duration,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        prefill_concurrency=phase.prefill_concurrency,
        request_rate=_phase_request_rate(phase),
        arrival_pattern=_phase_arrival_pattern(phase),
        arrival_smoothness=getattr(phase, "smoothness", None),
        seamless=False,
        grace_period_sec=grace_period,
        concurrency_ramp_duration_sec=_ramp_duration(phase.concurrency_ramp),
        prefill_concurrency_ramp_duration_sec=_ramp_duration(phase.prefill_ramp),
        request_rate_ramp_duration_sec=_ramp_duration(
            getattr(phase, "rate_ramp", None)
        ),
        artifact_dir=artifact_dir,
        dataset_sampling_strategy=dataset_sampling_strategy,
        allow_dataset_wrap=allow_dataset_wrap,
        cache_bust=cache_bust,
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
        trajectory_start_min_ratio=trajectory_start_min_ratio,
        trajectory_start_max_ratio=trajectory_start_max_ratio,
        burst_phase_starts=burst_phase_starts,
        random_seed=random_seed,
        cache_pressure_duration=cache_pressure_duration,
    )


def _graph_tstar_active(trajectory_start_max_ratio: float | None) -> bool:
    """True iff the graph t* snapshot window is engaged (max ratio > 0).

    Takes the SAME resolved config value ``from_run`` threads onto each graph
    ``CreditPhaseConfig``, so the auto-warmup decision and the strategy's t*
    sampling agree: a positive upper ratio means at least some traces sample
    ``t* > 0`` and therefore have a pre-``t*`` prefix worth priming. ``[0, 0]``
    (full native replay) leaves it inactive -- no warmup needed.
    """
    return (trajectory_start_max_ratio or 0.0) > 0.0


def _graph_pressure_grace_sec(user_grace: float | None, duration: float) -> float:
    """Drain grace for a pressure-mode warmup.

    AgentX parity (`_agentic_warmup_grace_period`, its timing/config.py:226-240):
    an EXPLICIT user warmup grace is honored verbatim (the operator's escape
    hatch when a healthy drain outlives the pressure duration -- e.g. a
    45s prefill in flight at a 30s deadline); otherwise min(duration, cap)
    bounds the drain so a wedged or lost return cannot hang the run. We do
    not port agentx's benchmark-grace floor branch. The strategy's stash
    completeness gate converts a drain that force-completes with unreturned
    credits into a safe handoff skip. Callers gate on a set
    pressure duration first.
    """
    from aiperf.common.environment import Environment

    if user_grace is not None:
        return user_grace

    cap = Environment.GRAPH.PRESSURE_DRAIN_GRACE_CAP
    return min(duration, cap)


def _build_graph_auto_warmup_config(
    profiling_phases: list[PhaseConfig],
    *,
    user_grace: float | None = None,
    dataset_sampling_strategy: DatasetSamplingStrategy | None = None,
    allow_dataset_wrap: bool | None = None,
    cache_bust: CacheBustTarget | None = None,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    trajectory_start_min_ratio: float | None = None,
    trajectory_start_max_ratio: float | None = None,
    burst_phase_starts: bool | None = None,
    random_seed: int | None = None,
    cache_pressure_duration: float | None = None,
) -> CreditPhaseConfig | None:
    """Auto-build the t*-snapshot WARMUP phase for a weka graph-IR run.

    AgentX parity: the graph-replay scenario path always prepends a WARMUP phase priming the
    boundary (``k_i-1``) turn of every chain live at ``t*`` (one priming credit
    per live chain), dispatched as a ``CONCURRENCY_BURST`` with an infinite grace
    period so the warmup barrier holds until every priming credit returns. The
    GraphIRReplayStrategy owns warmup completion (its warmup phase variant runs
    ``aiperf.timing.strategies.graph_ir_replay.rewrite_for_warmup``, a flat
    START-rooted graph firing exactly the live chains' boundary turns, then
    drains), so this carries no stop-condition counts -- only the concurrency
    (inherited from the profiling phase so the warmup primes at the same width
    the profiling phase will run).

    Grace: the boundary-priming default is an infinite grace (the barrier holds
    until every priming credit returns). When the cache-pressure stage is active
    (a set pressure duration), the drain is instead bounded by
    :func:`_graph_pressure_grace_sec` so a wedged or lost pressure return cannot
    hang the run; ``user_grace`` (a graph run's explicit ``--warmup-grace-period``
    carried through the mode-owned supersede) is honored verbatim there.

    Returns ``None`` when there is no profiling phase to inherit concurrency from
    (degenerate config); the caller then simply skips the auto-warmup.
    """
    if not profiling_phases:
        return None
    base = profiling_phases[0]
    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        timing_mode=TimingMode.GRAPH_IR,
        concurrency=base.concurrency,
        prefill_concurrency=base.prefill_concurrency,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        seamless=False,
        grace_period_sec=(
            _graph_pressure_grace_sec(user_grace, cache_pressure_duration)
            if cache_pressure_duration is not None
            else float("inf")
        ),
        dataset_sampling_strategy=dataset_sampling_strategy,
        allow_dataset_wrap=allow_dataset_wrap,
        cache_bust=cache_bust,
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
        trajectory_start_min_ratio=trajectory_start_min_ratio,
        trajectory_start_max_ratio=trajectory_start_max_ratio,
        burst_phase_starts=burst_phase_starts,
        random_seed=random_seed,
        cache_pressure_duration=cache_pressure_duration,
    )


def _build_profiling_config(
    phase: PhaseConfig,
    *,
    is_graph: bool = False,
    artifact_dir: Path | None = None,
    dataset_sampling_strategy: DatasetSamplingStrategy | None = None,
    allow_dataset_wrap: bool | None = None,
    cache_bust: CacheBustTarget | None = None,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    trajectory_start_min_ratio: float | None = None,
    trajectory_start_max_ratio: float | None = None,
    burst_phase_starts: bool | None = None,
    random_seed: int | None = None,
    cache_pressure_duration: float | None = None,
) -> CreditPhaseConfig:
    """Build a profiling CreditPhaseConfig from a profiling PhaseConfig.

    Main benchmark phase where all performance metrics are collected.
    Grace period allows in-flight requests to complete after the stop condition
    is met, ensuring metrics include requests that were sent before the deadline.

    ``is_graph`` forces ``TimingMode.GRAPH_IR`` so a weka graph workload selects
    ``GraphIRReplayStrategy`` regardless of the phase's ``type``.
    """
    timing_mode = TimingMode.GRAPH_IR if is_graph else _phase_timing_mode(phase)
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=timing_mode,
        expected_duration_sec=phase.duration,
        total_expected_requests=phase.requests,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        prefill_concurrency=phase.prefill_concurrency,
        request_rate=_phase_request_rate(phase),
        arrival_pattern=_phase_arrival_pattern(phase),
        arrival_smoothness=getattr(phase, "smoothness", None),
        seamless=phase.seamless,
        grace_period_sec=phase.grace_period,
        num_users=getattr(phase, "users", None),
        concurrency_ramp_duration_sec=_ramp_duration(phase.concurrency_ramp),
        prefill_concurrency_ramp_duration_sec=_ramp_duration(phase.prefill_ramp),
        request_rate_ramp_duration_sec=_ramp_duration(
            getattr(phase, "rate_ramp", None)
        ),
        # Fixed schedule config
        auto_offset_timestamps=getattr(
            phase, "auto_offset", InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET
        ),
        fixed_schedule_start_offset=getattr(phase, "start_offset", None),
        fixed_schedule_end_offset=getattr(phase, "end_offset", None),
        artifact_dir=artifact_dir,
        adaptive_sustain_duration_sec=getattr(phase, "adaptive_sustain_duration", None),
        adaptive_assessment_period_sec=getattr(
            phase, "adaptive_assessment_period", None
        )
        or 30.0,
        adaptive_control_variable=getattr(
            phase, "adaptive_control_variable", "concurrency"
        ),
        adaptive_control_min=getattr(phase, "adaptive_control_min", 1),
        adaptive_control_max=getattr(phase, "adaptive_control_max", None),
        adaptive_scale_strategy_type=getattr(
            phase, "adaptive_scale_strategy_type", "ramp_until_fail"
        ),
        adaptive_scale_step_policy=getattr(
            phase, "adaptive_scale_step_policy", "sla_margin"
        ),
        adaptive_scale_base_step=getattr(phase, "adaptive_scale_base_step", 10),
        adaptive_scale_max_step_multiplier=getattr(
            phase, "adaptive_scale_max_step_multiplier", 4
        ),
        adaptive_scale_step_percent=getattr(phase, "adaptive_scale_step_percent", 25.0),
        adaptive_min_completed_requests=getattr(
            phase, "adaptive_min_completed_requests", 1
        ),
        adaptive_sla_filters=tuple(getattr(phase, "sla", ()) or ()),
        dataset_sampling_strategy=dataset_sampling_strategy,
        allow_dataset_wrap=allow_dataset_wrap,
        cache_bust=cache_bust,
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
        trajectory_start_min_ratio=trajectory_start_min_ratio,
        trajectory_start_max_ratio=trajectory_start_max_ratio,
        burst_phase_starts=burst_phase_starts,
        random_seed=random_seed,
        cache_pressure_duration=cache_pressure_duration,
    )
