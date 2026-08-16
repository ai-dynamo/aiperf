# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import PhaseKind
from aiperf.config.dataset.defaults import InputDefaults
from aiperf.config.rate_series import RateSeriesConfig
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
    from aiperf.dataset.graph.models import ParsedGraph


_logger = AIPerfLogger(__name__)

_AGENTIC_CACHE_WARMUP_DEFAULT_GRACE_PERIOD_SEC = 300.0


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


def _is_agentic_replay(profiling_phases: list[PhaseConfig]) -> bool:
    """True when the profiling phase resolves to the AGENTIC_REPLAY timing mode.

    AGENTIC_REPLAY is selected by the agentic scenario lock (ScenarioResolver /
    apply_scenario), which stamps ``timing_mode = AGENTIC_REPLAY`` on the
    profiling phase. Detection reads that phase ``timing_mode`` when present,
    falling back to the phase type mapping. Normal / dag_jsonl runs never
    resolve to AGENTIC_REPLAY, so this stays False for them.
    """
    if not profiling_phases:
        return False
    phase = profiling_phases[0]
    explicit = getattr(phase, "timing_mode", None)
    if explicit is not None:
        return explicit == TimingMode.AGENTIC_REPLAY
    return _phase_timing_mode(phase) == TimingMode.AGENTIC_REPLAY


class TimingConfig(AIPerfBaseModel):
    """Configuration for TimingManager and timing strategies.

    Controls timing mode (REQUEST_RATE, FIXED_SCHEDULE, or USER_CENTRIC_RATE),
    rate/concurrency settings, warmup/profiling phase stop conditions, and
    request cancellation behavior.
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
    concurrency: int | None = Field(
        default=None,
        gt=0,
        description="User-configured target concurrency. Required by AGENTIC_REPLAY "
        "to size the trajectory list built once at PhaseOrchestrator construction.",
    )
    random_seed: int | None = Field(
        default=None,
        ge=0,
        description="User-configured random seed. Used by AGENTIC_REPLAY to derive "
        "deterministic per-trace start-turn indices for trajectories.",
    )
    trajectory_start_min_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="AGENTIC_REPLAY: lower bound (inclusive) on the random "
        "per-trajectory start position, as a fraction of the trace's total "
        "turn count.",
    )
    trajectory_start_max_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="AGENTIC_REPLAY: upper bound (inclusive) on the random "
        "per-trajectory start position, as a fraction of the trace's total "
        "turn count. Effective per-trace ceiling is min(int(max_ratio * n), n - 2).",
    )
    allow_dataset_wrap: bool = Field(
        default=False,
        description="Allow AGENTIC_REPLAY to reuse distinct eligible traces "
        "across concurrency lanes when concurrency exceeds the loaded pool. "
        "Defaults to False so over-subscription requires explicit opt-in.",
    )
    cache_bust_enabled: bool = Field(
        default=False,
        description="Whether the active dataset has a non-NONE cache-bust "
        "target. An active cache-bust marker keeps repeated-trace traffic "
        "distinct, so it satisfies the dataset-wrap opt-in on its own.",
    )

    @classmethod
    def from_run(
        cls,
        run: BenchmarkRun,
        *,
        parsed_graph: ParsedGraph | None = None,
    ) -> TimingConfig:
        """Build ordered list of credit-phase configs from a ``BenchmarkRun``.

        Preserves the ordered ``cfg.phases`` list. Each executable phase gets
        stable identity metadata. AGENTIC_REPLAY replaces declared warmup phases
        with its synthesized trajectory warmup.

        ``parsed_graph``, when supplied, is the built corpus the run will
        actually replay. It is the ONLY source of truth for corpus-supplied
        warmup: a call without it (the pre-configure construction in
        ``TimingManager.__init__``) carries no corpus warmup phase and is
        re-resolved once the sidecar lands.
        """
        cfg = run.cfg

        profiling_phases = cfg.get_profiling_phases()
        agentic = _is_agentic_replay(profiling_phases)
        artifact_dir = cfg.artifacts.dir

        # Agent graph workloads replay a conversation DAG via
        # ``AgentGraphReplayStrategy`` (TimingMode.AGENT_GRAPH), not the linear
        # per-turn timing modes. Every plane uses the memoized structural
        # resolution, so an auto-detected graph input cannot have its build and
        # schedule paths disagree. Non-graph workloads are unchanged.
        from aiperf.config.phases import resolve_graph_tstar_window
        from aiperf.dataset.graph.workload_detect import (
            _resolve_graph_max_context,
            _resolve_graph_num_entries,
            resolve_graph_workload,
        )

        is_graph = resolve_graph_workload(run) is not None

        first_profiling = profiling_phases[0] if profiling_phases else None
        graph_fields: dict[str, Any] = {}
        graph_warmup_phases = list(cfg.get_warmup_phases())
        graph_tstar_min, graph_tstar_max = resolve_graph_tstar_window(first_profiling)
        if is_graph:
            from aiperf.config.dataset import as_file_dataset

            graph_file_ds = as_file_dataset(cfg.get_default_dataset())
            # Resolved per-trace-instance cache-bust target (scenario-auto-filled
            # before this runs). Threaded onto each graph CreditPhaseConfig so
            # ``AgentGraphReplayStrategy``'s dispatch duplication report can decide
            # whether recycle duplication is safe.
            #
            # Resolved graph-plane corpus-selection caps let the wrap-guard
            # phrase an over-subscription shortfall as CAPPED (a knob shrank the
            # corpus) rather than EXHAUSTED. REUSE the build plane's OWN
            # resolvers so the dispatch-side caps match the build-side exactly.
            #
            # The t* snapshot window + phase-start burst mode come from the
            # profiling phase so the strategy samples the same window the
            # auto-warmup decision used; the run seed drives deterministic
            # per-trace t* sampling (the SAME --random-seed that seeds
            # synthesized content, so one seed reproduces the whole run).
            graph_fields = {
                "dataset_sampling_strategy": run.resolved.dataset_sampling_strategy,
                "allow_dataset_wrap": run.resolved.allow_dataset_wrap,
                "cache_bust": cfg.get_cache_bust_target(),
                "num_dataset_entries": _resolve_graph_num_entries(run),
                "max_context_length": _resolve_graph_max_context(run),
                # An UNSET t* window means 0.0 on the agent-graph path (window
                # OFF, full recorded replay); only an explicit
                # --trajectory-start-*-ratio activates the snapshot window here.
                # AGENTIC_REPLAY resolves the same unset state to the full trace.
                "trajectory_start_min_ratio": graph_tstar_min,
                "trajectory_start_max_ratio": graph_tstar_max,
                "burst_phase_starts": (
                    first_profiling.burst_phase_starts
                    if first_profiling is not None
                    else None
                ),
                "random_seed": run.random_seed,
                # Declared ONLY on FileDataset; a synthetic/public default
                # dataset falls back to the declared default of the
                # CreditPhaseConfig field this value populates, so the fallback
                # never drifts from the field it feeds.
                "replay_speedup": (
                    graph_file_ds.replay_speedup
                    if graph_file_ds
                    else _phase_field_default("replay_speedup")
                ),
                "open_loop_replay": (
                    graph_file_ds.open_loop_replay
                    if graph_file_ds
                    else _phase_field_default("open_loop_replay")
                ),
                "open_loop_strict": (
                    graph_file_ds.open_loop_strict
                    if graph_file_ds
                    else _phase_field_default("open_loop_strict")
                ),
                "graph_tool_image": (
                    graph_file_ds.graph_tool_image if graph_file_ds else None
                ),
                "graph_tool_persistent_session": (
                    graph_file_ds.graph_tool_persistent_session
                    if graph_file_ds
                    else False
                ),
            }
            # Explicit graph-incompatible phase choices take precedence over the
            # detection (same rule as --custom-dataset-type above): reject loudly
            # instead of silently rerouting.
            _reject_graph_incompatible_phases(graph_warmup_phases, profiling_phases)

        profiling_default_cancellation = _default_cancellation_config(cfg.phases)
        warmup_default_cancellation = RequestCancellationConfig()

        configs: list[CreditPhaseConfig] = []
        if agentic:
            agentic_warmup = _build_agentic_warmup_config(profiling_phases[0])
            if agentic_warmup is not None:
                configs.append(agentic_warmup)

        # Two independent reasons a graph run needs a WARMUP phase, both
        # suppressed when the user supplied an explicit warmup phase:
        #
        # * t*-snapshot priming (BOUNDARY_SNAPSHOT): synthesized from the
        #   profiled traces by rewrite_for_warmup; needs the t* window open.
        # * corpus-supplied warmup (RECORDED): the adapter lowered real warmup
        #   graphs into parsed_graph.warmup_traces (e.g. Agent Trace Replay). The corpus
        #   -- not the flag -- is authoritative: a flag set on an adapter that
        #   emits nothing would inject an empty phase, and a corpus that emits
        #   warmup without the flag would silently profile it.
        if (
            is_graph
            and not graph_warmup_phases
            and (
                _graph_tstar_active(graph_fields.get("trajectory_start_max_ratio"))
                or bool(parsed_graph is not None and parsed_graph.warmup_traces)
            )
        ):
            auto = _build_graph_auto_warmup_config(
                profiling_phases,
                graph_fields=graph_fields,
            )
            if auto is not None:
                configs.append(auto)

        profiling_index = 0
        for phase_index, phase in enumerate(cfg.phases):
            if agentic and phase.kind == "warmup":
                continue
            current_profiling_index = None
            if phase.kind == "profiling":
                current_profiling_index = profiling_index
                profiling_index += 1
            default_cancellation = (
                profiling_default_cancellation
                if phase.kind == "profiling"
                else warmup_default_cancellation
            )
            configs.append(
                _build_phase_config(
                    phase,
                    artifact_dir=artifact_dir,
                    default_cancellation=default_cancellation,
                    phase_index=phase_index,
                    profiling_index=current_profiling_index,
                    is_graph=is_graph,
                    graph_fields=graph_fields or None,
                )
            )

        # Agentic sizing fields: concurrency from the profiling phase;
        # random_seed from the run; trajectory_start_* from the profiling
        # phase (BasePhaseConfig fields added in P1). Defaults preserve normal
        # / dag_jsonl behavior (these are only consumed on the agentic path).
        concurrency = first_profiling.concurrency if first_profiling else None
        # None IS the unset state on the phase; AGENTIC_REPLAY resolves it to
        # the full trace. A conditional, not `or` -- a deliberate 0.0 must survive.
        phase_min = (
            first_profiling.trajectory_start_min_ratio if first_profiling else None
        )
        phase_max = (
            first_profiling.trajectory_start_max_ratio if first_profiling else None
        )
        trajectory_min = 0.0 if phase_min is None else phase_min
        trajectory_max = 1.0 if phase_max is None else phase_max
        synthesis = getattr(cfg.get_default_dataset(), "synthesis", None)
        allow_dataset_wrap = bool(
            getattr(synthesis, "allow_dataset_wrap", False) if synthesis else False
        )
        cache_bust_enabled = cfg.get_cache_bust_target() != CacheBustTarget.NONE

        return cls(
            phase_configs=configs,
            request_cancellation=profiling_default_cancellation,
            urls=list(cfg.endpoint.urls),
            url_selection_strategy=cfg.endpoint.url_strategy,
            concurrency=concurrency,
            random_seed=run.random_seed,
            trajectory_start_min_ratio=trajectory_min,
            trajectory_start_max_ratio=trajectory_max,
            allow_dataset_wrap=allow_dataset_wrap,
            cache_bust_enabled=cache_bust_enabled,
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
    phase_index: int | None = Field(
        default=None, ge=0, description="Absolute index in the ordered phases list."
    )
    profiling_index: int | None = Field(
        default=None,
        ge=0,
        description="Index among profiling-kind phases; None for warmup.",
    )
    phase_name: str | None = Field(
        default=None, description="User-provided unique phase name."
    )
    phase_kind: PhaseKind | None = Field(
        default=None, description="Phase semantic kind: warmup or profiling."
    )
    request_cancellation: RequestCancellationConfig = Field(
        default_factory=RequestCancellationConfig,
        description="Phase-local request cancellation policy.",
    )
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
    concurrency_explicitly_set: bool = Field(
        default=False,
        description="True when the operator explicitly chose ``concurrency`` rather "
        "than inheriting the phase default. Carried from the source phase config's "
        "persisted provenance flag: ``concurrency`` defaults to a positive 1 on the "
        "default concurrency phase, so its VALUE cannot distinguish an inherited "
        "ceiling from a requested one. The graph open-loop replay path gates trace "
        "admission on this, never on the value.",
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
    request_rate_series: RateSeriesConfig | None = Field(
        default=None,
        description="Piecewise-linear request-rate schedule, if enabled.",
    )
    replay_speedup: float | None = Field(
        default=None,
        gt=0,
        description="Graph replay speedup factor; recorded pacing is divided by this value.",
    )
    open_loop_replay: bool = Field(
        default=True,
        description="Schedule graph traces from recorded timestamps instead of admission order.",
    )
    open_loop_strict: bool = Field(
        default=False,
        description="Schedule graph nodes independently from recorded timestamps, ignoring graph dependencies.",
    )
    graph_tool_image: str | None = Field(
        default=None,
        description="Resolved `--graph-tool-image` for graph workloads, carried "
        "from the file dataset so AgentGraphReplayStrategy can pick the tool "
        "sandbox backend. None (and for non-graph phases) selects the LOCAL "
        "backend; a non-empty image selects the Docker backend with that image. "
        "Read only when the graph actually carries tool nodes.",
    )
    graph_tool_persistent_session: bool = Field(
        default=False,
        description="Resolved `--graph-tool-persistent-session`. False: fresh "
        "docker exec per command (OL-matching). True: persistent bash session "
        "inside the container. Carried from the file dataset.",
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
    agentic_cache_warmup_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration of the accelerated cache-pressure substage for "
        "agentic replay warmup.",
    )

    dataset_sampling_strategy: DatasetSamplingStrategy | None = Field(
        default=None,
        description="Resolved run-level dataset sampling strategy for graph "
        "workloads, carried from `run.resolved.dataset_sampling_strategy` so the "
        "AgentGraphReplayStrategy can consume it. None for non-graph phases and "
        "until resolution derives it.",
    )
    allow_dataset_wrap: bool | None = Field(
        default=None,
        description="Resolved graph-plane dataset-wrap policy, carried from "
        "`run.resolved.allow_dataset_wrap` so the AgentGraphReplayStrategy can "
        "consume it. None for non-graph phases and until resolution derives it.",
    )
    cache_bust: CacheBustTarget | None = Field(
        default=None,
        description="Resolved per-trace-instance cache-bust target, carried from "
        "the run's resolved cache-bust target so the AgentGraphReplayStrategy's "
        "dispatch duplication report can decide whether recycle duplication is "
        "safe (cache-bust ON) or warns (OFF). None for non-graph phases.",
    )
    num_dataset_entries: int | None = Field(
        default=None,
        ge=1,
        description="Resolved explicit --num-dataset-entries corpus cap for graph "
        "workloads (the run's default-dataset `entries`), carried so the "
        "AgentGraphReplayStrategy wrap-guard phrases an over-subscription shortfall "
        "as CAPPED rather than EXHAUSTED. None for non-graph phases and when unset.",
    )
    max_context_length: int | None = Field(
        default=None,
        ge=1,
        description="Resolved --max-context-length per-trace context cap for graph "
        "workloads (`synthesis.max_context_length`), carried so the "
        "AgentGraphReplayStrategy wrap-guard phrases an over-subscription shortfall "
        "as CAPPED rather than EXHAUSTED. None for non-graph phases and when unset.",
    )
    trajectory_start_min_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Resolved t* snapshot-window lower bound for graph "
        "workloads, carried from the profiling phase's "
        "`trajectory_start_min_ratio` (`--trajectory-start-min-ratio`, "
        "scenario-auto-applied when unset) so the AgentGraphReplayStrategy samples "
        "the same window the auto-warmup decision used. None for non-graph "
        "phases; unset resolves to 0.0.",
    )
    trajectory_start_max_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Resolved t* snapshot-window upper bound for graph "
        "workloads, carried from the profiling phase's "
        "`trajectory_start_max_ratio` (`--trajectory-start-max-ratio`, "
        "scenario-auto-applied when unset). None for non-graph phases; unset "
        "resolves to 0.0 (window OFF).",
    )
    burst_phase_starts: bool | None = Field(
        default=None,
        description="Resolved --burst-phase-starts phase-start dispatch mode "
        "for graph workloads, carried from the profiling phase's "
        "`burst_phase_starts`. None for non-graph phases.",
    )
    random_seed: int | None = Field(
        default=None,
        ge=0,
        description="The run's resolved --random-seed, threaded so the graph "
        "strategy's t* sampling derives from the SAME seed as synthesized "
        "content (one seed reproduces the whole run; sweep cells decorrelate "
        "via the orchestrator's per-variation seed derivation).",
    )
    artifact_dir: Path | None = Field(
        default=None,
        description="Directory for phase-owned timing artifacts.",
    )
    adaptive: AdaptiveTimingConfig = Field(
        default_factory=AdaptiveTimingConfig,
        description="Adaptive scale timing settings.",
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


def _phase_field_default(name: str) -> Any:
    """Return the declared default of ``CreditPhaseConfig.<name>``.

    Used by the graph-field builder so a missing ``FileDataset`` falls back to
    the single declared default of the field the value populates, rather than a
    restated literal that can drift from it. ``FileDataset`` declares the same
    defaults on its own side; the two are asserted to agree by
    ``tests/unit/timing/test_graph_field_defaults.py``.
    """
    return CreditPhaseConfig.model_fields[name].get_default(call_default_factory=True)


def _ramp_duration(ramp: object | None) -> float | None:
    """Extract the ramp duration in seconds from a ``RamperConfig`` (or None)."""
    if ramp is None:
        return None
    return getattr(ramp, "duration", None)


def _phase_request_rate(phase: PhaseConfig) -> float | None:
    """Return the configured request rate for a phase, if any."""
    # Lazy import: aiperf.config.phases is only a TYPE_CHECKING import here.
    from aiperf.config.phases import get_phase_rate

    return get_phase_rate(phase)


def _phase_arrival_pattern(phase: PhaseConfig) -> ArrivalPattern:
    """Map a phase type to its arrival pattern."""
    return _PHASE_TYPE_TO_ARRIVAL_PATTERN.get(phase.type, ArrivalPattern.POISSON)


def _reject_graph_incompatible_phases(
    warmup_phases: list[PhaseConfig], profiling_phases: list[PhaseConfig]
) -> None:
    """Reject phase configs whose explicit pacing a graph replay would discard.

    The recorded graph replay (TimingMode.AGENT_GRAPH) owns pacing and
    concurrency, so rerouting these phases to AGENT_GRAPH would
    silently discard the user's explicit choice:

    * ``adaptive_scale`` profiling phases drive their own concurrency ladder.
    * Rate-controlled and fixed-schedule phase TYPES encode explicit user
      pacing (``--request-rate`` / ``--user-centric-rate`` /
      ``--fixed-schedule``).
    """
    if any(phase.adaptive_scale for phase in profiling_phases):
        raise ValueError(
            "adaptive_scale is not supported for graph workloads: the "
            "recorded graph replay (TimingMode.AGENT_GRAPH) owns pacing and "
            "concurrency. Remove adaptive_scale from the phase config, or "
            "pin a non-graph loader with --custom-dataset-type to run "
            "this input through the linear pipeline."
        )
    for phase in (*warmup_phases, *profiling_phases):
        if phase.type != PhaseType.CONCURRENCY:
            raise ValueError(
                f"phase '{phase.name}' (type={phase.type}) is not "
                "supported for graph workloads: the recorded graph "
                "replay (TimingMode.AGENT_GRAPH) owns pacing, so "
                "rate-controlled arrivals (--request-rate / "
                "--user-centric-rate) and --fixed-schedule "
                "timestamps would be silently discarded. Remove the "
                "rate/schedule options (--concurrency bounds the "
                "replay lanes), or pin a non-graph loader with "
                "--custom-dataset-type to run this input through "
                "the linear pipeline."
            )


def resolve_graph_content_seed(run: BenchmarkRun) -> int | None:
    """Return the run's seed for graph content synthesis -- the AIPerf seed.

    Just ``run.random_seed`` (``--random-seed``), the same seed schedule /
    topology / t* derive from. The DatasetManager is the only parser; the
    TimingManager ingests the graph_meta sidecar from the graph-typed dataset
    broadcast. Resolving from the run config alone keeps every parse of the same
    run (in-process or spawn-started pool worker) byte-identical. ``None`` (no
    ``--random-seed``) keeps the ambient global RNG -- there is no graph-specific
    seed fallback.
    """
    return run.random_seed


def resolve_graph_content_tokenizer(run: BenchmarkRun) -> str:
    """Resolve the tokenizer the graph content synthesizer must use.

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


def _graph_tstar_active(trajectory_start_max_ratio: float | None) -> bool:
    """True iff the graph t* snapshot window is engaged (max ratio > 0).

    Takes the SAME resolved config value ``from_run`` threads onto each graph
    ``CreditPhaseConfig``, so the auto-warmup decision and the strategy's t*
    sampling agree: a positive upper ratio means at least some traces sample
    ``t* > 0`` and therefore have a pre-``t*`` prefix worth priming. ``[0, 0]``
    (full recorded replay) leaves it inactive -- no warmup needed.
    """
    return (trajectory_start_max_ratio or 0.0) > 0.0


def _build_graph_auto_warmup_config(
    profiling_phases: list[PhaseConfig],
    *,
    graph_fields: dict[str, Any] | None = None,
) -> CreditPhaseConfig | None:
    """Auto-build the t*-snapshot WARMUP phase for an agent-graph run.

    Agentic parity: the graph-replay path always prepends a WARMUP phase priming
    the boundary (``k_i-1``) turn of every chain live at ``t*`` (one priming credit
    per live chain), dispatched as a ``CONCURRENCY_BURST`` with an infinite grace
    period so the warmup barrier holds until every priming credit returns. The
    AgentGraphReplayStrategy owns warmup completion (its warmup phase variant runs
    ``aiperf.timing.strategies.agent_graph_replay.rewrite_for_warmup``, a flat
    START-rooted graph firing exactly the live chains' boundary turns, then
    drains), so this carries no stop-condition counts -- only the concurrency
    (inherited from the profiling phase so the warmup primes at the same width
    the profiling phase will run).

    Grace: an infinite grace period -- the barrier holds until every priming
    credit returns.

    Returns ``None`` when there is no profiling phase to inherit concurrency from
    (degenerate config); the caller then simply skips the auto-warmup.
    """
    if not profiling_phases:
        return None
    fields = graph_fields or {}
    base = profiling_phases[0]
    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        phase_kind="warmup",
        timing_mode=TimingMode.AGENT_GRAPH,
        concurrency=base.concurrency,
        concurrency_explicitly_set=base.concurrency_explicitly_set,
        prefill_concurrency=base.prefill_concurrency,
        # Inherited alongside concurrency: the strategy's over-subscription guard
        # stands down when the session budget fits the loaded corpus, and the
        # warmup primes at the profiling phase's width, so it must see the same
        # budget or a narrow corpus raises on warmup but not on profiling.
        expected_num_sessions=base.sessions,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        seamless=False,
        grace_period_sec=float("inf"),
        **fields,
    )


def _default_cancellation_config(
    phases: list[PhaseConfig],
) -> RequestCancellationConfig:
    for phase in phases:
        if getattr(phase, "kind", None) != "profiling":
            continue
        cancellation = getattr(phase, "cancellation", None)
        if cancellation is not None:
            return RequestCancellationConfig(
                rate=cancellation.rate, delay=cancellation.delay
            )
    return RequestCancellationConfig()


def _phase_cancellation_config(
    phase: PhaseConfig, default_cancellation: RequestCancellationConfig
) -> RequestCancellationConfig:
    cancellation = getattr(phase, "cancellation", None)
    if cancellation is None:
        return default_cancellation
    return RequestCancellationConfig(rate=cancellation.rate, delay=cancellation.delay)


def _build_phase_config(
    phase: PhaseConfig,
    *,
    artifact_dir: Path | None = None,
    default_cancellation: RequestCancellationConfig,
    phase_index: int,
    profiling_index: int | None,
    is_graph: bool = False,
    graph_fields: dict[str, Any] | None = None,
) -> CreditPhaseConfig:
    if phase.kind == "warmup":
        return _build_warmup_config(
            phase,
            artifact_dir=artifact_dir,
            default_cancellation=default_cancellation,
            phase_index=phase_index,
            profiling_index=profiling_index,
            is_graph=is_graph,
            graph_fields=graph_fields,
        )
    return _build_profiling_config(
        phase,
        artifact_dir=artifact_dir,
        default_cancellation=default_cancellation,
        phase_index=phase_index,
        profiling_index=profiling_index,
        is_graph=is_graph,
        graph_fields=graph_fields,
    )


def _build_warmup_config(
    phase: PhaseConfig,
    *,
    artifact_dir: Path | None = None,
    default_cancellation: RequestCancellationConfig,
    phase_index: int,
    profiling_index: int | None,
    is_graph: bool = False,
    graph_fields: dict[str, Any] | None = None,
) -> CreditPhaseConfig:
    """Build a warmup CreditPhaseConfig from a warmup PhaseConfig.

    Warmup triggers JIT compilation, memory allocation, and connection pool
    initialization so profiling measurements aren't polluted by cold-start effects.

    When the phase doesn't set ``grace_period``, default to infinity (wait
    forever for in-flight requests). This differs from the CreditPhaseConfig
    field default of None (disabled) because warmup should always complete all
    in-flight requests before transitioning to profiling.

    ``is_graph`` forces ``TimingMode.AGENT_GRAPH`` so a graph workload's WARMUP
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
        phase_index=phase_index,
        profiling_index=profiling_index,
        phase_name=phase.name,
        phase_kind=phase.kind,
        request_cancellation=_phase_cancellation_config(phase, default_cancellation),
        # Non-graph warmup is always request-rate paced; a graph workload must run
        # the agent graph strategy here so warmup replays the recorded topology.
        timing_mode=TimingMode.AGENT_GRAPH if is_graph else TimingMode.REQUEST_RATE,
        total_expected_requests=phase.requests,
        expected_duration_sec=phase.duration,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        concurrency_explicitly_set=getattr(phase, "concurrency_explicitly_set", False),
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
        request_rate_series=getattr(phase, "rate_series", None),
        **(graph_fields or {}),
    )


def _agentic_warmup_grace_period(phase: PhaseConfig) -> float | None:
    """Resolve the agentic auto-warmup barrier grace from the profiling phase.

    Explicit ``--agentic-warmup-grace-period`` wins. Otherwise, an accelerated
    cache-pressure warmup drain is bounded by the larger of the benchmark grace
    period (resolved onto the profiling phase's ``grace_period``) and a relaxed
    default of ``min(cache_warmup_duration, 300s)`` — long accelerated warmups
    hold many in-flight one-token requests, so a 30s benchmark grace drains too
    aggressively. A plain snapshot warmup keeps the infinite barrier.
    """
    grace_period = getattr(phase, "agentic_warmup_grace_period", None)
    if grace_period is not None:
        return grace_period
    cache_warmup_duration = getattr(phase, "agentic_cache_warmup_duration", None)
    if cache_warmup_duration is not None:
        default_grace_period = min(
            cache_warmup_duration,
            _AGENTIC_CACHE_WARMUP_DEFAULT_GRACE_PERIOD_SEC,
        )
        benchmark_grace_period = getattr(phase, "grace_period", None)
        if benchmark_grace_period is None:
            return default_grace_period
        return max(benchmark_grace_period, default_grace_period)
    return float("inf")


def _build_agentic_warmup_config(phase: PhaseConfig) -> CreditPhaseConfig | None:
    """Build the AGENTIC_REPLAY auto-warmup phase from the profiling PhaseConfig.

    AGENTIC_REPLAY auto-creates a warmup phase sized to the trajectory list
    (one credit per concurrency lane), dispatched as a single
    CONCURRENCY_BURST. ``total_expected_requests=concurrency`` lets the
    sending-complete stop condition fire after the warmup burst; if the pool
    is smaller than concurrency the strategy emits ``mark_sending_complete``
    itself.

    The warmup barrier grace comes from ``agentic_warmup_grace_period`` (the
    ``--agentic-warmup-grace-period`` knob, routed onto the profiling phase),
    NOT from the profiling phase's own ``grace_period``. The agentic warmup is
    synthesized rather than a user-declared warmup phase, so it cannot inherit
    ``--warmup-grace-period`` (which requires ``--warmup-duration``); reusing the
    profiling grace would leak the profiling tail into the warmup barrier. When
    unset, grace is infinite so the barrier holds until every primed trajectory
    returns (origin/agentx semantics) — except under accelerated cache-pressure
    warmup, where the strategy-terminated drain must be bounded: there the
    benchmark grace period (resolved onto the profiling phase's
    ``grace_period``) caps the drain instead.
    """
    concurrency = getattr(phase, "concurrency", None)
    grace_period = _agentic_warmup_grace_period(phase)
    cache_warmup_duration = getattr(phase, "agentic_cache_warmup_duration", None)
    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        # An accelerated cache-pressure warmup is strategy-terminated (the
        # strategy emits ``mark_sending_complete`` when the duration elapses),
        # so leave the request cap open instead of sizing it to concurrency.
        total_expected_requests=(
            None if cache_warmup_duration is not None else concurrency
        ),
        expected_duration_sec=None,
        expected_num_sessions=None,
        concurrency=concurrency,
        prefill_concurrency=getattr(phase, "prefill_concurrency", None),
        request_rate=None,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        arrival_smoothness=getattr(phase, "smoothness", None),
        seamless=False,
        grace_period_sec=grace_period if grace_period is not None else float("inf"),
        agentic_cache_warmup_duration_sec=cache_warmup_duration,
    )


def _build_profiling_config(
    phase: PhaseConfig,
    *,
    artifact_dir: Path | None = None,
    default_cancellation: RequestCancellationConfig,
    phase_index: int,
    profiling_index: int | None,
    is_graph: bool = False,
    graph_fields: dict[str, Any] | None = None,
) -> CreditPhaseConfig:
    """Build a profiling CreditPhaseConfig from a profiling PhaseConfig.

    Main benchmark phase where all performance metrics are collected.
    Grace period allows in-flight requests to complete after the stop condition
    is met, ensuring metrics include requests that were sent before the deadline.

    ``is_graph`` forces ``TimingMode.AGENT_GRAPH`` so a graph workload selects
    ``AgentGraphReplayStrategy`` regardless of the phase's ``type``.
    """
    # An explicit ``timing_mode`` on the phase (set by the agentic scenario
    # lock in P2) wins; otherwise derive it from the phase type. This is how
    # AGENTIC_REPLAY reaches the profiling CreditPhaseConfig.
    explicit_mode = getattr(phase, "timing_mode", None)
    timing_mode = (
        TimingMode.AGENT_GRAPH
        if is_graph
        else (explicit_mode or _phase_timing_mode(phase))
    )
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_index=phase_index,
        profiling_index=profiling_index,
        phase_name=phase.name,
        phase_kind=phase.kind,
        request_cancellation=_phase_cancellation_config(phase, default_cancellation),
        timing_mode=timing_mode,
        expected_duration_sec=phase.duration,
        total_expected_requests=phase.requests,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        concurrency_explicitly_set=getattr(phase, "concurrency_explicitly_set", False),
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
        request_rate_series=getattr(phase, "rate_series", None),
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
        **(graph_fields or {}),
    )
