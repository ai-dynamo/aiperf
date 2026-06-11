# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WekaTraceLoader: native AIPerf loader for kv-cache-tester agentic traces.

Accepts a single JSON file or a directory of per-conversation JSON files.
Each trace emits one root Conversation plus one child Conversation per
``type: "subagent"`` entry, linked via SPAWN + SPAWN_JOIN prerequisites.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.hash_ids_synthesis import HashIdsPromptSynthesisMixin
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaSubagentEntry,
    WekaTrace,
)
from aiperf.plugin.enums import DatasetSamplingStrategy

_logger = AIPerfLogger(__name__)

_NormalRequestT = WekaNormalRequest | WekaStreamingRequest
_JOIN_EPSILON_SECONDS = 1e-6


def _subagent_request_absolute_t(
    entry: WekaSubagentEntry, req: WekaNormalRequest
) -> float:
    """Return a subagent inner request timestamp in root-trace coordinates.

    Current Weka captures store inner request ``t`` as an absolute timestamp,
    while older synthetic/unit fixtures used subagent-relative values. Treat a
    child timestamp before the spawn marker as relative so both shapes land on
    the same root-trace timeline.
    """
    if req.t + _JOIN_EPSILON_SECONDS < entry.t:
        return entry.t + req.t
    return req.t


def _request_end_seconds(start_seconds: float, api_time: float | None) -> float:
    """Request interval end in seconds; missing/negative/non-finite durations become zero."""
    duration = api_time if api_time is not None and math.isfinite(api_time) else 0.0
    return start_seconds + max(duration, 0.0)


def _sa_end_seconds(entry: WekaSubagentEntry) -> float:
    """Recorded end time of a subagent, in seconds.

    Uses ``duration_ms`` when present. Falls back to ``max(inner.t + inner.api_time)``
    when ``duration_ms`` is None (recorded for ``status='async_launched'`` subagents).
    Falls back further to ``entry.t`` when both are unavailable.
    """
    if entry.duration_ms is not None:
        duration_s = entry.duration_ms / 1000.0
        # A subagent cannot finish before it starts: clamp negative / non-finite
        # recorded durations to 0 so the end never precedes the spawn timestamp
        # (a NaN/negative end otherwise poisons the parent.t >= sa_end join scan).
        if not math.isfinite(duration_s) or duration_s < 0.0:
            duration_s = 0.0
        return entry.t + duration_s
    if entry.requests:
        return max(
            _request_end_seconds(_subagent_request_absolute_t(entry, ir), ir.api_time)
            for ir in entry.requests
        )
    return entry.t


def _trace_peak_context_length(trace: WekaTrace, max_osl: int | None = None) -> int:
    """Peak requested context length across parent and subagent requests.

    vLLM validates prompt tokens plus requested output tokens against the
    model context window. Filtering only ``input_length`` leaves deterministic
    4xxs for traces whose prompt fits but ``prompt + max_tokens`` exceeds the
    server's max model length.
    """

    def capped_output(req: _NormalRequestT) -> int:
        if max_osl is not None and req.output_length > max_osl:
            return max_osl
        return req.output_length

    peak = 0
    for req in trace.requests:
        if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
            peak = max(peak, req.input_length + capped_output(req))
        elif isinstance(req, WekaSubagentEntry):
            for child_req in req.requests:
                peak = max(peak, child_req.input_length + capped_output(child_req))
    return peak


def _pack_into_streams(
    requests: list[WekaNormalRequest],
) -> list[list[WekaNormalRequest]]:
    """Partition inner requests into the minimum number of non-overlapping
    sequential streams (interval-graph chromatic decomposition, greedy
    earliest-fit).

    Two requests ``A``, ``B`` overlap when ``[A.t, A.t + A.api_time)`` intersects
    ``[B.t, B.t + B.api_time)``. Each returned stream is a chain of
    non-overlapping requests in ``t``-order. The number of streams equals the
    maximum number of concurrent inner requests at any instant.

    A request with ``api_time = None`` is treated as zero-duration (the
    interval becomes the instant ``[t, t)``) - it never overlaps anything by
    itself, so it lands in the first stream by ``t``. This matches the
    behaviour of subagents whose telemetry was not captured.
    """
    sorted_reqs = sorted(requests, key=lambda r: r.t)
    streams: list[list[WekaNormalRequest]] = []
    stream_ends: list[float] = []
    for r in sorted_reqs:
        r_end = r.t + (r.api_time or 0.0)
        placed = False
        for i, end in enumerate(stream_ends):
            if end <= r.t:
                streams[i].append(r)
                stream_ends[i] = r_end
                placed = True
                break
        if not placed:
            streams.append([r])
            stream_ends.append(r_end)
    return streams


def _clamp_delay_ms(delay_ms: float, cap_seconds: float | None) -> float:
    """Clamp a delay to at most cap_seconds * 1000 ms.

    Only enforces the upper bound; negative or NaN values pass through unchanged.
    """
    if cap_seconds is None:
        return delay_ms
    cap_ms = cap_seconds * 1000.0
    if delay_ms > cap_ms:
        return cap_ms
    return delay_ms


@dataclass(frozen=True)
class _RequestTiming:
    timestamp_seconds: float
    delay_ms: float | None


@dataclass(frozen=True)
class _IdleGap:
    raw_start: float
    raw_end: float
    shift_before: float
    cap_seconds: float
    excess_seconds: float


@dataclass
class _TraceIdleTiming:
    parent_by_outer_idx: dict[int, _RequestTiming]
    # Keyed by (child_plan.session_id, request_idx_within_stream). Not id(req):
    # the parallel reconstruction path pickles request objects to worker
    # processes, where they materialize at fresh memory addresses and any
    # id()-based dict misses with KeyError. (session_id, idx) is stable
    # across the pickle round-trip.
    child_by_session_request: dict[tuple[str, int], _RequestTiming]
    subagent_end_by_outer_idx: dict[int, float]
    # Mapped subagent spawn time (warp.map(entry.t)), keyed by outer idx. The
    # branch start_timestamp_ms must live on the same compressed timeline as
    # the turns it links; the raw entry.t would land far past the parent's
    # last turn whenever an idle gap is compressed.
    subagent_start_by_outer_idx: dict[int, float]


class _IdleGapTimeWarp:
    """Compress request-start gaps in one trace and map raw seconds to adjusted seconds."""

    def __init__(self, request_starts: list[float], cap_seconds: float):
        self._gaps: list[_IdleGap] = []
        sorted_starts = sorted(request_starts)
        if not sorted_starts:
            return

        prev_start = sorted_starts[0]
        cumulative_shift = 0.0
        for start in sorted_starts[1:]:
            gap_seconds = start - prev_start
            if gap_seconds > cap_seconds:
                excess = gap_seconds - cap_seconds
                self._gaps.append(
                    _IdleGap(
                        raw_start=prev_start,
                        raw_end=start,
                        shift_before=cumulative_shift,
                        cap_seconds=cap_seconds,
                        excess_seconds=excess,
                    )
                )
                cumulative_shift += excess
            prev_start = start

    def map(self, t_seconds: float) -> float:
        """Map a raw timestamp to the per-trace idle-gap-capped timeline.

        Each long request-start gap ``[a, b]`` is compressed by keeping the first
        ``cap_seconds`` after request ``a`` intact and collapsing the remainder
        to the cap boundary. Requests at or after ``b`` shift left by the
        collapsed excess. Non-request events inside the collapsed tail, such as
        subagent end markers, map to the same boundary so joins cannot wait past
        the next shifted request.
        """
        shift = 0.0
        for gap in self._gaps:
            if t_seconds < gap.raw_start:
                return t_seconds - gap.shift_before
            if t_seconds < gap.raw_end:
                local = t_seconds - gap.raw_start
                if local <= gap.cap_seconds:
                    return t_seconds - gap.shift_before
                return gap.raw_start - gap.shift_before + gap.cap_seconds
            shift = gap.shift_before + gap.excess_seconds
        return t_seconds - shift


@dataclass
class _ParentPlan:
    trace_id: str
    normals: list[tuple[int, _NormalRequestT]]
    subagents: list[tuple[int, WekaSubagentEntry]]
    block_size: int


@dataclass
class _ChildPlan:
    session_id: str
    parent_trace_id: str
    subagent_index: int
    entry: WekaSubagentEntry
    stream_index: int
    stream_requests: list[WekaNormalRequest]
    block_size: int


def _expand_subagent_to_child_plans(
    trace_id: str,
    sa_index: int,
    entry: WekaSubagentEntry,
    block_size: int,
) -> list[_ChildPlan]:
    """Pack a subagent's inner requests into per-stream child plans.

    Single-stream subagents keep the legacy ``::sa:{agent_id}`` session-id
    shape; multi-stream subagents append ``:s{stream_index}``. Subagents with
    zero recorded inner requests still emit one (empty) child to preserve
    the parent SPAWN branch's child-conversation target.
    """
    streams = _pack_into_streams(list(entry.requests))
    if not streams:
        streams = [[]]
    plans: list[_ChildPlan] = []
    multi = len(streams) > 1
    for stream_idx, stream_reqs in enumerate(streams):
        if multi:
            child_sid = f"{trace_id}::sa:{entry.agent_id}:s{stream_idx}"
        else:
            child_sid = f"{trace_id}::sa:{entry.agent_id}"
        plans.append(
            _ChildPlan(
                session_id=child_sid,
                parent_trace_id=trace_id,
                subagent_index=sa_index,
                entry=entry,
                stream_index=stream_idx,
                stream_requests=stream_reqs,
                block_size=block_size,
            )
        )
    return plans


def _dropped_subagent_indices(plan: _ParentPlan) -> set[int]:
    normal_outer_indices = [outer_idx for outer_idx, _ in plan.normals]
    dropped: set[int] = set()
    for subagent_index, (sa_outer_idx, _) in enumerate(plan.subagents):
        if not any(outer_idx < sa_outer_idx for outer_idx in normal_outer_indices):
            dropped.add(subagent_index)
    return dropped


def _child_plans_for_active_subagents(
    plan: _ParentPlan, child_plans: list[_ChildPlan]
) -> list[_ChildPlan]:
    dropped = _dropped_subagent_indices(plan)
    return [
        cp
        for cp in child_plans
        if cp.parent_trace_id == plan.trace_id and cp.subagent_index not in dropped
    ]


def _count_seen_prefix_blocks(hash_ids: list[int], seen: set[int]) -> int:
    """Return leading blocks already present in ``seen`` for prefix-cache math."""
    hits = 0
    for hash_id in hash_ids:
        if hash_id not in seen:
            break
        hits += 1
    return hits


def _build_trace_idle_timing(
    *,
    plan: _ParentPlan,
    child_plans: list[_ChildPlan],
    cap_seconds: float,
) -> _TraceIdleTiming:
    """Build per-turn timing after capping request-start gaps in one root trace.

    The cap is per root trace, not global across the dataset. We collect every
    request submission timestamp from the parent and all subagents, compress
    any gap between consecutive starts above ``cap_seconds``, then derive parent
    and child conversation delays from that adjusted timeline.

    Example with ``cap_seconds=60``:
      - main request starts at t=0
      - subagent request starts at t=20 and originally takes 80s
      - next main request starts at t=220
    The capped gap is based on request starts only: 20 -> 220 is 200s, so the
    next main request shifts left by 140s to t=80. The original subagent
    latency still matters for join placement, but it does not prevent this idle
    gap from being compressed.
    """
    request_starts: list[float] = []
    for _, req in plan.normals:
        request_starts.append(req.t)

    child_plans_for_trace = _child_plans_for_active_subagents(plan, child_plans)
    for cp in child_plans_for_trace:
        for req in cp.stream_requests:
            request_starts.append(_subagent_request_absolute_t(cp.entry, req))

    warp = _IdleGapTimeWarp(request_starts, cap_seconds)
    parent_by_outer_idx: dict[int, _RequestTiming] = {}
    prev_t: float | None = None
    for outer_idx, req in plan.normals:
        t = warp.map(req.t)
        delay_ms = None if prev_t is None else (t - prev_t) * 1000.0
        parent_by_outer_idx[outer_idx] = _RequestTiming(t, delay_ms)
        prev_t = t

    child_by_session_request: dict[tuple[str, int], _RequestTiming] = {}
    for cp in child_plans_for_trace:
        prev_child_t: float | None = None
        for k, req in enumerate(cp.stream_requests):
            t = warp.map(_subagent_request_absolute_t(cp.entry, req))
            delay_ms = None if prev_child_t is None else (t - prev_child_t) * 1000.0
            child_by_session_request[(cp.session_id, k)] = _RequestTiming(t, delay_ms)
            prev_child_t = t

    subagent_end_by_outer_idx = {
        outer_idx: warp.map(_sa_end_seconds(entry))
        for outer_idx, entry in plan.subagents
    }
    subagent_start_by_outer_idx = {
        outer_idx: warp.map(entry.t) for outer_idx, entry in plan.subagents
    }
    return _TraceIdleTiming(
        parent_by_outer_idx=parent_by_outer_idx,
        child_by_session_request=child_by_session_request,
        subagent_end_by_outer_idx=subagent_end_by_outer_idx,
        subagent_start_by_outer_idx=subagent_start_by_outer_idx,
    )


class WekaTraceLoader(HashIdsPromptSynthesisMixin, BaseFileLoader):
    """Dataset loader for Weka KV-cache-tester agentic coding trace files.

    Note: despite the "trace" in the name, this loader is NOT part of the
    ``BaseTraceDatasetLoader`` family (sibling examples:
    ``MooncakeTraceDatasetLoader``, ``BurstGPTTraceDatasetLoader``). Weka
    traces require KV-cache-aware prompt synthesis with multi-segment
    ``raw_messages``, which doesn't fit the single-prompt-per-turn shape
    that ``BaseTraceDatasetLoader`` assumes. We extend ``BaseFileLoader``
    plus ``HashIdsPromptSynthesisMixin`` instead.

    Accepts a single JSON file or a directory of per-conversation JSON files
    (auto-detected via :meth:`can_load`). Each trace produces:

    - one root :class:`Conversation` from the trace's normal/streaming requests
    - one child :class:`Conversation` per ``type: "subagent"`` entry, linked
      via SPAWN + SPAWN_JOIN prerequisites on the parent's turns

    Reconstruction is byte-deterministic across the in-process serial path
    and the multiprocessing pool path (gated by ``WEKA_PARALLEL_THRESHOLD``
    and ``WEKA_PARALLEL_WORKERS`` env vars); both paths share the LCP-driven
    :class:`~aiperf.dataset.loader.weka_synth_buf.ConversationReconstructor`.

    Usage::

        loader = WekaTraceLoader(
            filename="/path/to/traces/",  # file or directory of *.json
            user_config=user_config,
            prompt_generator=prompt_generator,  # required for token replay
        )
        data = loader.load_dataset()              # {trace_id: [WekaTrace]}
        conversations = loader.convert_to_conversations(data)

    Side effects in :meth:`convert_to_conversations`:

    - clears ``prompt_generator._cache`` per trace (scope-local hash IDs)
    - resets ``prompt_generator._hash_id_corpus_rng`` per trace

    Raises:
        ValueError: malformed JSON, schema violation, or duplicate trace ID.
    """

    def __init__(
        self,
        *,
        filename: str | None = None,
        user_config: UserConfig,
        prompt_generator: PromptGenerator | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self._path = Path(filename) if filename is not None else None
        self.prompt_generator = prompt_generator
        if prompt_generator is not None:
            self._tokenizer_name = (
                prompt_generator.tokenizer.resolved_name
                or user_config.tokenizer.name
                or user_config.endpoint.model_names[0]
            )
        else:
            self._tokenizer_name = user_config.tokenizer.name
        self._trust_remote_code = user_config.tokenizer.trust_remote_code
        self._tokenizer_revision = user_config.tokenizer.revision
        user_block_size = user_config.input.prompt.input_tokens.block_size
        if user_block_size is not None:
            self._user_block_size_override: int | None = user_block_size
        elif default_block_size is not None:
            self._user_block_size_override = default_block_size
        else:
            self._user_block_size_override = None
        # ``self._block_size`` is preserved for callbacks (``_decode_block_tokens``
        # closes over it) and for tests that set it directly. It is overwritten
        # per-trace in the reconstruction loop with the result of
        # ``_block_size_for_trace`` so the user-override > trace-declared > 64
        # precedence is honored without changing the callback signature.
        self._block_size = self._user_block_size_override or 64
        self._delay_cap_tracker = DelayCapTracker(
            cap_seconds=user_config.loadgen.inter_turn_delay_cap_seconds
        )
        self._use_live_assistant = Environment.DATASET.WEKA_LIVE_ASSISTANT_RESPONSES

    def _block_size_for_trace(self, trace: WekaTrace) -> int:
        """Resolve block_size with precedence: user-override > trace-declared > 64.

        Real Weka captures declare their own ``block_size`` per file (see
        :class:`WekaTrace.block_size`). When the user hasn't passed
        ``--block-size`` (or whatever flag maps to
        ``user_config.input.prompt.input_tokens.block_size``) we honor that
        per-file value instead of silently using the historical default of 64.
        """
        if self._user_block_size_override is not None:
            return self._user_block_size_override
        return trace.block_size

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode:
        """Weka emits delta-encoded turns; the endpoint accumulates at request time.

        Overrides ``BaseFileLoader.get_default_context_mode`` (None) so the
        composer / dataset_manager picks the right delta mode for weka,
        which (a) matches the per-turn ``raw_messages`` shape this loader now
        emits and (b) correctly bypasses the preformat fast path in
        ``DatasetManager`` (deltas need at-request-time accumulation).

        When ``AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES`` is set, the
        loader emits user-only deltas and the worker threads live server
        responses into the session's ``turn_list`` via the
        ``DELTAS_WITHOUT_RESPONSES`` ``store_response`` path.
        """
        if Environment.DATASET.WEKA_LIVE_ASSISTANT_RESPONSES:
            return ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        return ConversationContextMode.DELTAS_WITH_RESPONSES

    def _resolved_context_mode(self) -> ConversationContextMode:
        """Per-instance counterpart to ``get_default_context_mode``.

        Read once at ``__init__`` time so all four ``Conversation`` construction
        sites in this loader pick the same mode, regardless of whether the env
        var is mutated mid-run.
        """
        if self._use_live_assistant:
            return ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        return ConversationContextMode.DELTAS_WITH_RESPONSES

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Return True when ``filename`` is a Weka JSON file or a directory of them.

        Directory detection is single-probe (matches ``RandomPoolDatasetLoader``)
        so plugin auto-detection stays O(1) on 739-file corpora.
        """
        if filename is None:
            return False
        path = Path(filename) if isinstance(filename, str) else filename
        try:
            if path.is_dir():
                # Sort for deterministic single-probe behavior; raw ``glob``
                # iteration order is filesystem-dependent (ext4 returns hash
                # order, not alphabetical).
                first = next(iter(sorted(path.glob("*.json"))), None)
                return first is not None and cls._probe_file(first)
            return cls._probe_file(path)
        except Exception as e:
            _logger.debug(f"WekaTraceLoader.can_load error on {path}: {e!r}")
            return False

    @classmethod
    def _probe_file(cls, path: Path) -> bool:
        if not path.is_file() or path.suffix != ".json":
            return False
        try:
            blob = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            return False
        if not isinstance(blob, dict):
            return False
        try:
            WekaTrace.model_validate(blob)
            return True
        except ValidationError:
            return False

    def load_dataset(self) -> dict[str, list[WekaTrace]]:
        """Parse every Weka trace file and return ``{trace_id: [WekaTrace]}``.

        The list is always length 1 — each file is its own conversation; the
        shape matches the ``dict[str, list[T]]`` contract used by Mooncake /
        Bailian loaders.
        """
        import time

        files = self._enumerate_files()
        n = len(files)
        _logger.info(f"WekaTraceLoader: parsing {n} trace file(s) from {self._path}")
        t0 = time.monotonic()
        log_every = max(1, n // 10)
        data: dict[str, list[WekaTrace]] = {}
        for i, path in enumerate(files, 1):
            trace = self._load_single_file(path)
            if trace.id in data:
                raise ValueError(
                    f"Duplicate trace id '{trace.id}' in directory: "
                    f"'{path}' conflicts with a prior file"
                )
            data[trace.id] = [trace]
            if i % log_every == 0 and i != n:
                _logger.info(
                    f"WekaTraceLoader: parsed {i}/{n} trace files "
                    f"({time.monotonic() - t0:.1f}s elapsed)"
                )
        _logger.info(
            f"WekaTraceLoader: parsed {n} trace file(s) in {time.monotonic() - t0:.1f}s"
        )
        return data

    def _enumerate_files(self) -> list[Path]:
        if self._path is None:
            raise ValueError(
                "WekaTraceLoader: load_dataset() requires a filename. "
                "This loader instance was constructed without one (e.g. for "
                "delegated reconstruction from a public HF source)."
            )
        if self._path.is_dir():
            return sorted(self._path.glob("*.json"))
        return [self._path]

    def _load_single_file(self, path: Path) -> WekaTrace:
        try:
            blob = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON: {e}") from e
        try:
            return WekaTrace.model_validate(blob)
        except ValidationError as e:
            raise ValueError(
                f"{path}: file is JSON but does not match the Weka trace schema: {e}"
            ) from e

    def _request_passes_filters(self, req: _NormalRequestT) -> bool:
        # fixed_schedule_*_offset are in milliseconds (per input_config.py);
        # weka traces record req.t in seconds. Compare in ms.
        start = self.user_config.input.fixed_schedule_start_offset
        end = self.user_config.input.fixed_schedule_end_offset
        t_ms = req.t * 1000.0
        if start is not None and t_ms < start:
            return False
        if end is not None and t_ms > end:
            return False
        max_isl = self.user_config.input.synthesis.max_isl
        return not (max_isl is not None and req.input_length > max_isl)

    def _filter_traces_by_max_context(
        self, data: dict[str, list[WekaTrace]], max_ctx: int
    ) -> dict[str, list[WekaTrace]]:
        """Drop traces whose peak requested context length exceeds ``max_ctx``.

        Uses the per-request ``input_length`` and ``output_length`` recorded
        in the WEKA trace so no client-side re-tokenization is required. The
        peak across parent and subagent requests is the trace's worst case;
        any conversation branch exceeding it would 4xx mid-run.
        """
        kept: dict[str, list[WekaTrace]] = {}
        max_seen = 0
        max_osl = self.user_config.input.synthesis.max_osl
        for trace_id, wekas in data.items():
            peak = _trace_peak_context_length(wekas[0], max_osl=max_osl)
            if peak > max_seen:
                max_seen = peak
            if peak <= max_ctx:
                kept[trace_id] = wekas

        total = len(data)
        dropped = total - len(kept)
        if dropped:
            _logger.info(
                "--max-context-length=%d: dropped %d/%d traces exceeding the "
                "limit (largest observed: %d tokens).",
                max_ctx,
                dropped,
                total,
                max_seen,
            )
        else:
            _logger.info(
                "--max-context-length=%d: all %d traces within limit "
                "(largest: %d tokens).",
                max_ctx,
                total,
                max_seen,
            )
        if not kept:
            raise DatasetLoaderError(
                f"All {total} traces exceed --max-context-length={max_ctx} "
                "tokens; nothing left to benchmark. Raise the limit or use "
                "a smaller-context dataset."
            )
        return kept

    def _cap_output(self, req: _NormalRequestT) -> int:
        max_osl = self.user_config.input.synthesis.max_osl
        if max_osl is not None and req.output_length > max_osl:
            return max_osl
        return req.output_length

    def _trace_idle_gap_cap_seconds(self) -> float | None:
        """Optional per-trace idle-gap cap; robust to MagicMock test configs."""
        value = getattr(self.user_config.loadgen, "trace_idle_gap_cap_seconds", None)
        if isinstance(value, int | float):
            return float(value)
        return None

    def _build_reconstruction_plans(
        self, data: dict[str, list[WekaTrace]]
    ) -> tuple[list[_ParentPlan], list[_ChildPlan]]:
        parent_plans: list[_ParentPlan] = []
        child_plans: list[_ChildPlan] = []
        for trace_id, wekas in data.items():
            trace = wekas[0]
            trace_bs = self._block_size_for_trace(trace)
            normals: list[tuple[int, _NormalRequestT]] = []
            subagents: list[tuple[int, WekaSubagentEntry]] = []
            for idx, req in enumerate(trace.requests):
                if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
                    if self._request_passes_filters(req):
                        normals.append((idx, req))
                else:  # WekaSubagentEntry
                    sa_index = len(subagents)
                    subagents.append((idx, req))
                    child_plans.extend(
                        _expand_subagent_to_child_plans(
                            trace_id, sa_index, req, trace_bs
                        )
                    )
            parent_plans.append(
                _ParentPlan(trace_id, normals, subagents, block_size=trace_bs)
            )
        return parent_plans, child_plans

    def _build_trace_idle_timing_by_trace(
        self, parent_plans: list[_ParentPlan], child_plans: list[_ChildPlan]
    ) -> dict[str, _TraceIdleTiming]:
        trace_idle_gap_cap_seconds = self._trace_idle_gap_cap_seconds()
        if trace_idle_gap_cap_seconds is None:
            return {}
        return {
            plan.trace_id: _build_trace_idle_timing(
                plan=plan,
                child_plans=child_plans,
                cap_seconds=trace_idle_gap_cap_seconds,
            )
            for plan in parent_plans
        }

    def _build_model_map(self, trace: WekaTrace) -> dict[str, str]:
        """Map trace-side model names to ``endpoint.model_names``.

        The trace's "main" model (first parent request, falling back to the
        first request of the first subagent for parent-less traces) maps to
        ``endpoint.model_names[0]``. Other distinct trace models map to
        ``endpoint.model_names[1..]`` in order of first appearance, with
        modulo wrap when distinct trace models exceed configured models.
        Identity mapping is returned when ``endpoint.model_names`` is empty.
        """
        configured = self.user_config.endpoint.model_names
        if not configured:
            return {}

        main_model: str | None = None
        for req in trace.requests:
            if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
                main_model = req.model
                break
        if main_model is None:
            for req in trace.requests:
                if isinstance(req, WekaSubagentEntry) and req.requests:
                    main_model = req.requests[0].model
                    break
        if main_model is None:
            return {}

        ordered: list[str] = [main_model]
        seen: set[str] = {main_model}
        for req in trace.requests:
            if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
                if req.model not in seen:
                    seen.add(req.model)
                    ordered.append(req.model)
            elif isinstance(req, WekaSubagentEntry):
                for creq in req.requests:
                    if creq.model not in seen:
                        seen.add(creq.model)
                        ordered.append(creq.model)

        n = len(configured)
        return {m: configured[i % n] for i, m in enumerate(ordered)}

    def _decode_block_tokens(self, hash_ids: list[int]) -> list[int]:
        """Concatenate per-hash-id Qwen token blocks into a single token list.

        The caller MUST clear ``self.prompt_generator._cache`` and call
        ``self.prompt_generator._hash_id_corpus_rng.set_trace_id(scope)``
        before any sequence of calls within a single conversation scope.

        Within that scope the int-keyed cache is valid: every
        ``(current_trace_id, hash_id) -> tokens`` mapping is deterministic
        via ``reseed_for_hash_id``. The ``hash_id_scope: "local"`` contract
        means we never need two scopes' cache content alive simultaneously,
        so int keys + per-scope clear is sufficient and bounds memory.
        """
        pg = self.prompt_generator
        rng = pg._hash_id_corpus_rng
        bs = self._block_size
        corpus = pg._tokenized_corpus
        corpus_size = pg._corpus_size
        cache = pg._cache
        tokens: list[int] = []
        for h in hash_ids:
            cached = cache.get(h)
            if cached is None:
                rng.reseed_for_hash_id(h)
                # Mirror PromptGenerator._sample_tokens: randrange over the
                # full corpus and wrap the slice if it overflows.
                start = rng.randrange(corpus_size)
                end = start + bs
                cached = corpus[start:end]
                if end > corpus_size:
                    cached = cached + corpus[: end - corpus_size]
                cache[h] = cached
            tokens.extend(cached)
        return tokens

    def _decode_tokens_to_text(self, tokens: list[int]) -> str:
        """Decode a Qwen token list to text (no special-token insertion)."""
        return self.prompt_generator.tokenizer.decode(tokens)

    def convert_to_conversations(
        self, data: dict[str, list[WekaTrace]]
    ) -> list[Conversation]:
        """Build one root + one-per-subagent Conversation per trace.

        Subagent markers become SPAWN branches on the preceding parent turn
        plus a SPAWN_JOIN TurnPrerequisite on the following parent turn.
        Terminal subagents (with no parent turn after them) become background
        branches (is_background=True, no prereq).
        """
        self._delay_cap_tracker.reset()

        # Track subagents whose branch was dropped during the second pass;
        # their child conversations must also be pruned.
        dropped_per_trace: dict[str, set[int]] = {}

        max_ctx = self.user_config.input.max_context_length
        if max_ctx is not None:
            data = self._filter_traces_by_max_context(data, max_ctx)

        parent_plans, child_plans = self._build_reconstruction_plans(data)

        # Per-trace model rewrite map. Built once here, applied in both the
        # serial and parallel reconstruction paths so workers don't need
        # access to UserConfig.
        model_map_per_trace: dict[str, dict[str, str]] = {
            trace_id: self._build_model_map(wekas[0])
            for trace_id, wekas in data.items()
        }

        import time as _time

        ignore_delays = self.user_config.input.ignore_trace_delays
        think_time_only = self.user_config.input.use_think_time_only
        cap_seconds = self.user_config.loadgen.inter_turn_delay_cap_seconds
        trace_idle_gap_cap_seconds = self._trace_idle_gap_cap_seconds()
        trace_idle_timing_by_trace = self._build_trace_idle_timing_by_trace(
            parent_plans, child_plans
        )
        turn_cap_seconds = (
            None if trace_idle_gap_cap_seconds is not None else cap_seconds
        )
        self._delay_cap_tracker.cap_seconds = turn_cap_seconds

        _t0 = _time.monotonic()
        _t1 = _time.monotonic()
        _n_plans = len(parent_plans)

        parallel_threshold = Environment.DATASET.WEKA_PARALLEL_THRESHOLD
        configured_workers = Environment.DATASET.WEKA_PARALLEL_WORKERS
        use_parallel = (
            self.prompt_generator is not None
            and _n_plans >= parallel_threshold
            and configured_workers != 1
        )

        try:
            if use_parallel:
                conversations = self._reconstruct_parallel(
                    parent_plans=parent_plans,
                    child_plans=child_plans,
                    data=data,
                    ignore_delays=ignore_delays,
                    think_time_only=think_time_only,
                    cap_seconds=turn_cap_seconds,
                    configured_workers=configured_workers,
                    t_start=_t1,
                    model_map_per_trace=model_map_per_trace,
                    trace_idle_timing_by_trace=trace_idle_timing_by_trace,
                )
            else:
                conversations = self._reconstruct_serial(
                    parent_plans=parent_plans,
                    child_plans=child_plans,
                    data=data,
                    dropped_per_trace=dropped_per_trace,
                    ignore_delays=ignore_delays,
                    think_time_only=think_time_only,
                    cap_seconds=turn_cap_seconds,
                    t_start=_t1,
                    model_map_per_trace=model_map_per_trace,
                    trace_idle_timing_by_trace=trace_idle_timing_by_trace,
                )
        finally:
            # Don't hold trace content past this call. The caller may process
            # many traces; per-scope clears bound peak memory but the final
            # clear ensures no leftover scope leaks back to other code paths
            # that share the same PromptGenerator.
            self.prompt_generator._cache.clear()

        from aiperf.common.models import DatasetMetadata
        from aiperf.common.validators.orchestrator_v1 import (
            validate_for_orchestrator_v1,
        )

        sampling = self.get_preferred_sampling_strategy()
        metadata = DatasetMetadata(
            conversations=[c.to_metadata() for c in conversations],
            sampling_strategy=sampling,
        )
        validate_for_orchestrator_v1(metadata)
        self._delay_cap_tracker.log_summary(logger_name=__name__)
        _logger.info(
            f"WekaTraceLoader: reconstructed {len(conversations)} conversation(s) "
            f"in {_time.monotonic() - _t1:.1f}s "
            f"(total load+synth+reconstruct: {_time.monotonic() - _t0:.1f}s)"
        )
        return conversations

    def _reconstruct_serial(
        self,
        *,
        parent_plans: list[_ParentPlan],
        child_plans: list[_ChildPlan],
        data: dict[str, list[WekaTrace]],
        dropped_per_trace: dict[str, set[int]],
        ignore_delays: bool,
        think_time_only: bool,
        cap_seconds: float | None,
        t_start: float,
        model_map_per_trace: dict[str, dict[str, str]],
        trace_idle_timing_by_trace: dict[str, _TraceIdleTiming],
    ) -> list[Conversation]:
        """In-process serial reconstruction."""
        import time as _time

        from aiperf.common.enums import (
            ConversationBranchMode,
            PrerequisiteKind,
        )
        from aiperf.common.models import (
            ConversationBranchInfo,
            Turn,
            TurnPrerequisite,
        )
        from aiperf.dataset.loader.weka_synth_buf import (
            ConversationReconstructor,
        )

        conversations: list[Conversation] = []
        n_plans = len(parent_plans)
        log_every_plan = max(1, n_plans // 10)

        for _plan_idx, plan in enumerate(parent_plans, 1):
            # ``hash_id_scope: "local"`` requires per-trace cache + RNG reset to
            # prevent cross-trace hash_id aliasing inflating KV-cache hit rates.
            pg = self.prompt_generator
            pg._cache.clear()
            pg._hash_id_corpus_rng.set_trace_id(plan.trace_id)

            # Sync the instance attribute so the ``_decode_block_tokens``
            # closure (which reads ``self._block_size``) sees the per-trace
            # value resolved by ``_block_size_for_trace``.
            self._block_size = plan.block_size

            model_map = model_map_per_trace.get(plan.trace_id, {})

            # raw_messages carries delta-encoded segments per turn; the
            # endpoint accumulates across turns at request time, with
            # ``reset_context`` flagging non-monotonic LCP cuts.
            trace = data[plan.trace_id][0]
            trace_idle_timing = trace_idle_timing_by_trace.get(plan.trace_id)
            conv = Conversation(
                session_id=plan.trace_id,
                context_mode=self._resolved_context_mode(),
            )
            recon = ConversationReconstructor(
                block_size=plan.block_size,
                decode_block_tokens=self._decode_block_tokens,
                sample_partial_tail_tokens=self.sample_partial_tail_tokens,
                decode_tokens_to_text=self._decode_tokens_to_text,
                bpe_stable_terminator_tokens=self.bpe_stable_terminator_tokens,
                emit_assistant_segments=not self._use_live_assistant,
            )

            # First pass: emit turns from normal requests; track outer-index → turn-pos.
            outer_to_turn_pos: dict[int, int] = {}
            parent_seen_hash_ids: set[int] = set()
            for k, (outer_idx, req) in enumerate(plan.normals):
                seed = f"{plan.trace_id}:turn_{k}:partial_tail"
                if k == 0:
                    recon.init_turn_0(
                        hash_ids=req.hash_ids,
                        in_tokens=req.input_length,
                        tool_tokens=trace.tool_tokens,
                        system_tokens=trace.system_tokens,
                        seed=seed,
                    )
                else:
                    prev_req = plan.normals[k - 1][1]
                    recon.advance_turn(
                        prev_hash_ids=prev_req.hash_ids,
                        prev_in_tokens=prev_req.input_length,
                        prev_out_tokens=prev_req.output_length,
                        curr_hash_ids=req.hash_ids,
                        curr_in_tokens=req.input_length,
                        seed=seed,
                    )

                # Turn.timestamp/delay are in milliseconds; weka traces record seconds.
                if trace_idle_timing is not None:
                    timing = trace_idle_timing.parent_by_outer_idx[outer_idx]
                    t_ms = timing.timestamp_seconds * 1000.0
                    delay_ms = timing.delay_ms
                else:
                    t_ms = req.t * 1000.0
                    if k == 0:
                        delay_ms = None
                    elif think_time_only and req.think_time is not None:
                        delay_ms = req.think_time * 1000.0
                    else:
                        delay_ms = t_ms - plan.normals[k - 1][1].t * 1000.0
                if delay_ms is not None:
                    delay_ms = self._delay_cap_tracker.clamp(delay_ms)
                    # Floor at 0: a negative inter-turn delay (corrupt
                    # think_time, or a non-monotonic timestamp gap) would tell
                    # the load generator to dispatch a request in the past.
                    delay_ms = max(delay_ms, 0.0)
                delta = recon.turn_delta()
                theoretical_hit_blocks = _count_seen_prefix_blocks(
                    req.hash_ids, parent_seen_hash_ids
                )
                theoretical_total_blocks = len(req.hash_ids)
                parent_seen_hash_ids.update(req.hash_ids)
                conv.turns.append(
                    Turn(
                        timestamp=None if ignore_delays else t_ms,
                        delay=None if ignore_delays else delay_ms,
                        model=model_map.get(req.model, req.model),
                        max_tokens=self._cap_output(req),
                        raw_messages=delta.delta_messages,
                        reset_context=delta.reset_context,
                        theoretical_prefix_cache_hit_blocks=theoretical_hit_blocks,
                        theoretical_prefix_cache_total_blocks=theoretical_total_blocks,
                    )
                )
                outer_to_turn_pos[outer_idx] = len(conv.turns) - 1

            # Group subagents by spawning parent turn and the first later parent
            # turn whose timestamp is at or after that subagent's recorded end.
            # This preserves tiered joins: short children can gate the next main
            # turn while longer siblings gate a later turn or run background.
            #
            # Examples:
            #   parent[0] t=0
            #   subagent A ends t=6
            #   subagent B ends t=12.5
            #   subagent C ends t=24
            #   parent[1] t=6
            #   parent[2] t=20
            #
            #   A joins parent[1] because parent[1].t >= A.end.
            #   B joins parent[2] because parent[1].t < B.end <= parent[2].t.
            #   C is background because no later parent turn reaches C.end.
            #
            # Additional examples:
            #   Shared join group:
            #     parent[0] t=0
            #     subagent A ends t=4
            #     subagent B ends t=5
            #     parent[1] t=6
            #     => A and B share group (parent[0], parent[1]); parent[1]
            #        waits for both.
            #
            #   Tiered siblings:
            #     parent[0] t=0
            #     subagent A ends t=4
            #     subagent B ends t=9
            #     parent[1] t=6
            #     parent[2] t=12
            #     => A gates parent[1]; B keeps running through parent[1] and
            #        gates parent[2].
            #
            #   No spawning parent:
            #     subagent A marker t=1 appears before the first retained
            #     parent turn
            #     parent[0] t=5
            #     => A is dropped because no parent turn can spawn it.
            #
            #   Equality joins:
            #     parent[0] t=0
            #     subagent A ends t=10
            #     parent[1] t=10
            #     => A joins parent[1] within _JOIN_EPSILON_SECONDS.
            groups: dict[
                tuple[int, int | None], list[tuple[int, WekaSubagentEntry]]
            ] = defaultdict(list)
            group_order: list[tuple[int, int | None]] = []
            dropped_subagent_indices: set[int] = set()
            child_sids_by_subagent: dict[int, list[str]] = defaultdict(list)
            for cp in child_plans:
                if cp.parent_trace_id == plan.trace_id:
                    child_sids_by_subagent[cp.subagent_index].append(cp.session_id)
            if trace_idle_timing is not None:
                outer_to_t: dict[int, float] = {
                    outer_idx: trace_idle_timing.parent_by_outer_idx[
                        outer_idx
                    ].timestamp_seconds
                    for outer_idx, _ in plan.normals
                }
            else:
                outer_to_t = {outer_idx: req.t for outer_idx, req in plan.normals}

            for subagent_index, (sa_outer_idx, sa_entry) in enumerate(plan.subagents):
                preceding = max(
                    (pos for oi, pos in outer_to_turn_pos.items() if oi < sa_outer_idx),
                    default=None,
                )
                if preceding is None:
                    _logger.info(
                        f"Dropping subagent '{sa_entry.agent_id}' from trace "
                        f"{plan.trace_id}: no preceding parent turn"
                    )
                    dropped_subagent_indices.add(subagent_index)
                    continue

                if trace_idle_timing is not None:
                    sa_end_t = trace_idle_timing.subagent_end_by_outer_idx[sa_outer_idx]
                    sa_start_t = trace_idle_timing.subagent_start_by_outer_idx[
                        sa_outer_idx
                    ]
                else:
                    sa_end_t = _sa_end_seconds(sa_entry)
                    sa_start_t = sa_entry.t
                join_turn: int | None = None
                for oi, pos in sorted(outer_to_turn_pos.items()):
                    if oi <= sa_outer_idx:
                        continue
                    if outer_to_t[oi] + _JOIN_EPSILON_SECONDS >= sa_end_t:
                        join_turn = pos
                        break

                key = (preceding, join_turn)
                if key not in groups:
                    group_order.append(key)
                groups[key].append((subagent_index, sa_entry, sa_start_t))

            for preceding, join_turn in group_order:
                entries = groups[(preceding, join_turn)]
                child_sids: list[str] = []
                for subagent_index, e, _sa_start_t in entries:
                    subagent_child_sids = child_sids_by_subagent[subagent_index]
                    child_sids.extend(subagent_child_sids)
                    if len(subagent_child_sids) > 1:
                        _logger.info(
                            f"Trace {plan.trace_id}: subagent '{e.agent_id}' has "
                            f"{len(subagent_child_sids)} parallel inner-request streams; "
                            f"emitting as sibling child conversations."
                        )
                branch_id = f"{plan.trace_id}:spawn:{entries[0][1].agent_id}"
                is_background = join_turn is None
                conv.branches.append(
                    ConversationBranchInfo(
                        branch_id=branch_id,
                        child_conversation_ids=child_sids,
                        mode=ConversationBranchMode.SPAWN,
                        is_background=is_background,
                        # Mapped spawn time (raw entry.t when no idle-gap warp),
                        # so the branch start shares the turns' compressed timeline.
                        start_timestamp_ms=min(s for _, _, s in entries) * 1000.0,
                    )
                )
                conv.turns[preceding].branch_ids.append(branch_id)
                if join_turn is not None:
                    conv.turns[join_turn].prerequisites.append(
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN,
                            branch_id=branch_id,
                        )
                    )
            dropped_per_trace[plan.trace_id] = dropped_subagent_indices
            conversations.append(conv)
            if _plan_idx % log_every_plan == 0 or _plan_idx == n_plans:
                elapsed = _time.monotonic() - t_start
                rate = _plan_idx / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * _plan_idx / n_plans
                _logger.info(
                    f"WekaTraceLoader: reconstructed "
                    f"{_plan_idx}/{n_plans} ({pct:.0f}%) parent conversations "
                    f"in {elapsed:.1f}s ({rate:.1f} traces/s)"
                )

        for cp in child_plans:
            if cp.subagent_index in dropped_per_trace.get(cp.parent_trace_id, set()):
                continue
            child_model_map = model_map_per_trace.get(cp.parent_trace_id, {})
            # ``hash_id_scope: "local"`` is one namespace per trace FILE: a
            # subagent shares its parent trace's scope so a hash_id reused
            # across parent and subagent (or across siblings) decodes to the
            # same tokens, reproducing the real cross-agent shared prefix.
            pg = self.prompt_generator
            pg._cache.clear()
            pg._hash_id_corpus_rng.set_trace_id(cp.parent_trace_id)
            # Sync for ``_decode_block_tokens``; see parent loop above.
            self._block_size = cp.block_size

            child_recon = ConversationReconstructor(
                block_size=cp.block_size,
                decode_block_tokens=self._decode_block_tokens,
                sample_partial_tail_tokens=self.sample_partial_tail_tokens,
                decode_tokens_to_text=self._decode_tokens_to_text,
                bpe_stable_terminator_tokens=self.bpe_stable_terminator_tokens,
                emit_assistant_segments=not self._use_live_assistant,
            )
            child_conv = Conversation(
                session_id=cp.session_id,
                context_mode=self._resolved_context_mode(),
                is_root=False,
                agent_depth=1,
                parent_conversation_id=cp.parent_trace_id,
            )
            child_seen_hash_ids: set[int] = set()
            for k, creq in enumerate(cp.stream_requests):
                seed = f"{cp.session_id}:turn_{k}:partial_tail"
                if k == 0:
                    child_recon.init_turn_0(
                        hash_ids=creq.hash_ids,
                        in_tokens=creq.input_length,
                        tool_tokens=cp.entry.tool_tokens,
                        system_tokens=cp.entry.system_tokens,
                        seed=seed,
                    )
                else:
                    prev_creq = cp.stream_requests[k - 1]
                    child_recon.advance_turn(
                        prev_hash_ids=prev_creq.hash_ids,
                        prev_in_tokens=prev_creq.input_length,
                        prev_out_tokens=prev_creq.output_length,
                        curr_hash_ids=creq.hash_ids,
                        curr_in_tokens=creq.input_length,
                        seed=seed,
                    )
                trace_idle_timing = trace_idle_timing_by_trace.get(cp.parent_trace_id)
                if trace_idle_timing is not None:
                    timing = trace_idle_timing.child_by_session_request[
                        (cp.session_id, k)
                    ]
                    t_ms = timing.timestamp_seconds * 1000.0
                    child_delay_ms = timing.delay_ms
                else:
                    t_ms = creq.t * 1000.0
                    if k == 0:
                        child_delay_ms = None
                    elif think_time_only and creq.think_time is not None:
                        child_delay_ms = creq.think_time * 1000.0
                    else:
                        child_delay_ms = t_ms - cp.stream_requests[k - 1].t * 1000.0
                if child_delay_ms is not None:
                    child_delay_ms = self._delay_cap_tracker.clamp(child_delay_ms)
                child_delta = child_recon.turn_delta()
                theoretical_hit_blocks = _count_seen_prefix_blocks(
                    creq.hash_ids, child_seen_hash_ids
                )
                theoretical_total_blocks = len(creq.hash_ids)
                child_seen_hash_ids.update(creq.hash_ids)
                child_conv.turns.append(
                    Turn(
                        timestamp=None if ignore_delays else t_ms,
                        delay=None if ignore_delays else child_delay_ms,
                        model=child_model_map.get(creq.model, creq.model),
                        max_tokens=creq.output_length,
                        raw_messages=child_delta.delta_messages,
                        reset_context=child_delta.reset_context,
                        theoretical_prefix_cache_hit_blocks=theoretical_hit_blocks,
                        theoretical_prefix_cache_total_blocks=theoretical_total_blocks,
                    )
                )
            conversations.append(child_conv)

        return conversations

    def _build_parallel_reconstruction_tasks(
        self,
        *,
        parent_plans: list[_ParentPlan],
        child_plans: list[_ChildPlan],
        data: dict[str, list[WekaTrace]],
        ignore_delays: bool,
        think_time_only: bool,
        cap_seconds: float | None,
        model_map_per_trace: dict[str, dict[str, str]],
        trace_idle_timing_by_trace: dict[str, _TraceIdleTiming],
    ):
        from aiperf.dataset.loader.weka_parallel_convert import (
            _WekaNormalRequestPayload,
            _WekaTraceTask,
        )

        # Drop the same child_plans the serial path drops at line ~1172.
        # _build_trace_idle_timing only populates timing for active subagents
        # (via _child_plans_for_active_subagents), so without this skip the
        # lookup below KeyErrors on any subagent that appears before the
        # first normal parent turn -- a real condition in the 256k-capped
        # corpus where reshifted timelines can leave the first subagent
        # without a preceding normal.
        dropped_per_trace: dict[str, set[int]] = {
            plan.trace_id: _dropped_subagent_indices(plan) for plan in parent_plans
        }

        children_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sids_by_subagent: dict[tuple[str, int], list[str]] = defaultdict(list)
        for cp in child_plans:
            if cp.subagent_index in dropped_per_trace.get(cp.parent_trace_id, set()):
                continue
            trace_idle_timing = trace_idle_timing_by_trace.get(cp.parent_trace_id)
            requests_dicts: list[_WekaNormalRequestPayload] = []
            for k, creq in enumerate(cp.stream_requests):
                req_payload: _WekaNormalRequestPayload = {
                    "hash_ids": list(creq.hash_ids),
                    "input_length": creq.input_length,
                    "output_length": creq.output_length,
                    "model": creq.model,
                    "t": creq.t,
                    "think_time": getattr(creq, "think_time", None),
                }
                if trace_idle_timing is not None:
                    timing = trace_idle_timing.child_by_session_request[
                        (cp.session_id, k)
                    ]
                    req_payload["effective_t"] = timing.timestamp_seconds
                    req_payload["effective_delay_ms"] = timing.delay_ms
                requests_dicts.append(req_payload)
            children_by_trace[cp.parent_trace_id].append(
                {
                    "session_id": cp.session_id,
                    "parent_trace_id": cp.parent_trace_id,
                    "subagent_index": cp.subagent_index,
                    "agent_id": cp.entry.agent_id,
                    "tool_tokens": cp.entry.tool_tokens,
                    "system_tokens": cp.entry.system_tokens,
                    "requests": requests_dicts,
                }
            )
            sids_by_subagent[(cp.parent_trace_id, cp.subagent_index)].append(
                cp.session_id
            )

        tasks: list[_WekaTraceTask] = []
        for plan in parent_plans:
            trace = data[plan.trace_id][0]
            tasks.append(
                _WekaTraceTask(
                    trace_id=plan.trace_id,
                    parent={
                        "normals": self._parallel_parent_normals(
                            plan, trace_idle_timing_by_trace
                        ),
                        "subagents": self._parallel_subagents(
                            plan, sids_by_subagent, trace_idle_timing_by_trace
                        ),
                        "tool_tokens": trace.tool_tokens,
                        "system_tokens": trace.system_tokens,
                    },
                    children=children_by_trace.get(plan.trace_id, []),
                    cap_seconds=cap_seconds,
                    ignore_delays=ignore_delays,
                    think_time_only=think_time_only,
                    model_map=model_map_per_trace.get(plan.trace_id, {}),
                    emit_assistant_segments=not self._use_live_assistant,
                    block_size=plan.block_size,
                )
            )
        return tasks

    def _parallel_parent_normals(
        self,
        plan: _ParentPlan,
        trace_idle_timing_by_trace: dict[str, _TraceIdleTiming],
    ):
        from aiperf.dataset.loader.weka_parallel_convert import (
            _WekaNormalRequestPayload,
        )

        trace_idle_timing = trace_idle_timing_by_trace.get(plan.trace_id)
        normals_dicts: list[tuple[int, _WekaNormalRequestPayload]] = []
        for outer_idx, req in plan.normals:
            req_payload: _WekaNormalRequestPayload = {
                "hash_ids": list(req.hash_ids),
                "input_length": req.input_length,
                "output_length": req.output_length,
                "model": req.model,
                "t": req.t,
                "think_time": getattr(req, "think_time", None),
                "capped_output_length": self._cap_output(req),
            }
            if trace_idle_timing is not None:
                timing = trace_idle_timing.parent_by_outer_idx[outer_idx]
                req_payload["effective_t"] = timing.timestamp_seconds
                req_payload["effective_delay_ms"] = timing.delay_ms
            normals_dicts.append((outer_idx, req_payload))
        return normals_dicts

    def _parallel_subagents(
        self,
        plan: _ParentPlan,
        sids_by_subagent: dict[tuple[str, int], list[str]],
        trace_idle_timing_by_trace: dict[str, _TraceIdleTiming],
    ):
        from aiperf.dataset.loader.weka_parallel_convert import (
            _WekaSubagentMarkerPayload,
        )

        trace_idle_timing = trace_idle_timing_by_trace.get(plan.trace_id)
        subagents_dicts: list[tuple[int, _WekaSubagentMarkerPayload]] = []
        for sa_index, (outer_idx, sa) in enumerate(plan.subagents):
            sa_payload: _WekaSubagentMarkerPayload = {
                "agent_id": sa.agent_id,
                "tool_tokens": sa.tool_tokens,
                "system_tokens": sa.system_tokens,
                "child_session_ids": sids_by_subagent.get(
                    (plan.trace_id, sa_index), []
                ),
                "sa_end_seconds": _sa_end_seconds(sa),
                "t": sa.t,
            }
            if trace_idle_timing is not None:
                sa_payload["effective_sa_end_seconds"] = (
                    trace_idle_timing.subagent_end_by_outer_idx[outer_idx]
                )
                # Mapped spawn time -> _process_task reads it for the branch
                # start_timestamp (e.get("effective_t", e["t"])); without it the
                # parallel branch start silently reverts to raw seconds under a warp.
                sa_payload["effective_t"] = (
                    trace_idle_timing.subagent_start_by_outer_idx[outer_idx]
                )
            subagents_dicts.append((outer_idx, sa_payload))
        return subagents_dicts

    def _reconstruct_parallel(
        self,
        *,
        parent_plans: list[_ParentPlan],
        child_plans: list[_ChildPlan],
        data: dict[str, list[WekaTrace]],
        ignore_delays: bool,
        think_time_only: bool,
        cap_seconds: float | None,
        configured_workers: int,
        t_start: float,
        model_map_per_trace: dict[str, dict[str, str]],
        trace_idle_timing_by_trace: dict[str, _TraceIdleTiming],
    ) -> list[Conversation]:
        """Per-trace parallel reconstruction across a multiprocessing Pool.

        Workers share the tokenized corpus via shared memory and run an
        exact-replica of :meth:`_decode_block_tokens` /
        :meth:`sample_partial_tail_tokens` / :meth:`_decode_tokens_to_text`
        against fresh per-scope cache + RNG. Output is byte-identical to
        :meth:`_reconstruct_serial`.
        """
        import os
        import time as _time

        from aiperf.common.enums import (
            ConversationBranchMode,
            PrerequisiteKind,
        )
        from aiperf.common.models import (
            ConversationBranchInfo,
            Turn,
            TurnPrerequisite,
        )
        from aiperf.dataset.loader.weka_parallel_convert import (
            run_parallel_weka_reconstruction,
        )

        tasks = self._build_parallel_reconstruction_tasks(
            parent_plans=parent_plans,
            child_plans=child_plans,
            data=data,
            ignore_delays=ignore_delays,
            think_time_only=think_time_only,
            cap_seconds=cap_seconds,
            model_map_per_trace=model_map_per_trace,
            trace_idle_timing_by_trace=trace_idle_timing_by_trace,
        )

        n_plans = len(tasks)
        if configured_workers > 0:
            num_workers = min(configured_workers, n_plans)
        else:
            num_workers = min((os.cpu_count() or 4) - 1, 16, n_plans)
        num_workers = max(1, num_workers)

        pg = self.prompt_generator
        _logger.info(
            f"WekaTraceLoader: spawning {num_workers} worker process(es) for "
            f"parallel reconstruction of {n_plans} trace(s)"
        )
        results = run_parallel_weka_reconstruction(
            tasks,
            tokenizer_name=self._tokenizer_name,
            corpus=pg._tokenized_corpus,
            base_seed=pg._hash_id_corpus_rng.seed,
            block_size=self._block_size,
            bpe_stable_terminator_tokens=self.bpe_stable_terminator_tokens,
            trust_remote_code=self._trust_remote_code,
            revision=self._tokenizer_revision or "main",
            num_workers=num_workers,
        )
        _logger.info(
            f"WekaTraceLoader: workers finished in {_time.monotonic() - t_start:.1f}s; "
            f"assembling Conversation objects"
        )

        conversations: list[Conversation] = []
        # Two-pass append to match the serial path's ordering: all parent
        # conversations first (in trace order), then all children (also in
        # trace order). Tests assert byte-identical output across paths.
        parent_convs: list[Conversation] = []
        for result in results:
            self._delay_cap_tracker.capped_count += result.get("capped_count", 0)
            observed = result.get("max_observed_ms", 0.0)
            if observed > self._delay_cap_tracker.max_observed_ms:
                self._delay_cap_tracker.max_observed_ms = observed
            trace_id = result["trace_id"]
            for agent_id in result["dropped_agent_ids"]:
                _logger.info(
                    f"Dropping subagent '{agent_id}' from trace {trace_id}: "
                    f"no preceding parent turn"
                )
            parent_conv = Conversation(
                session_id=trace_id,
                context_mode=self._resolved_context_mode(),
            )
            for t_dict in result["parent_turns"]:
                parent_conv.turns.append(
                    Turn(
                        timestamp=t_dict["timestamp"],
                        delay=t_dict["delay"],
                        model=t_dict["model"],
                        max_tokens=t_dict["max_tokens"],
                        raw_messages=t_dict["raw_messages"],
                        reset_context=t_dict["reset_context"],
                        theoretical_prefix_cache_hit_blocks=t_dict[
                            "theoretical_prefix_cache_hit_blocks"
                        ],
                        theoretical_prefix_cache_total_blocks=t_dict[
                            "theoretical_prefix_cache_total_blocks"
                        ],
                    )
                )
            for branch in result["branches"]:
                parent_conv.branches.append(
                    ConversationBranchInfo(
                        branch_id=branch["branch_id"],
                        child_conversation_ids=branch["child_session_ids"],
                        mode=ConversationBranchMode.SPAWN,
                        is_background=branch["is_background"],
                        start_timestamp_ms=branch.get("start_timestamp"),
                    )
                )
                parent_conv.turns[branch["preceding_turn"]].branch_ids.append(
                    branch["branch_id"]
                )
                if branch["following_turn"] is not None:
                    parent_conv.turns[branch["following_turn"]].prerequisites.append(
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN,
                            branch_id=branch["branch_id"],
                        )
                    )
            parent_convs.append(parent_conv)
        conversations.extend(parent_convs)

        for result in results:
            for child in result["children"]:
                child_conv = Conversation(
                    session_id=child["session_id"],
                    context_mode=self._resolved_context_mode(),
                    is_root=child["is_root"],
                    agent_depth=child["agent_depth"],
                    parent_conversation_id=child.get(
                        "parent_conversation_id", result["trace_id"]
                    ),
                )
                for t_dict in child["turns"]:
                    child_conv.turns.append(
                        Turn(
                            timestamp=t_dict["timestamp"],
                            delay=t_dict["delay"],
                            model=t_dict["model"],
                            max_tokens=t_dict["max_tokens"],
                            raw_messages=t_dict["raw_messages"],
                            reset_context=t_dict["reset_context"],
                            theoretical_prefix_cache_hit_blocks=t_dict[
                                "theoretical_prefix_cache_hit_blocks"
                            ],
                            theoretical_prefix_cache_total_blocks=t_dict[
                                "theoretical_prefix_cache_total_blocks"
                            ],
                        )
                    )
                conversations.append(child_conv)

        return conversations
