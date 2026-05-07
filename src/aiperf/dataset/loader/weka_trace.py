# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WekaTraceLoader: native AIPerf loader for kv-cache-tester agentic traces.

Accepts a single JSON file or a directory of per-conversation JSON files.
Each trace emits one root Conversation plus one child Conversation per
``type: "subagent"`` entry, linked via SPAWN + SPAWN_JOIN prerequisites.
"""

from __future__ import annotations

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


@dataclass
class _ParentPlan:
    trace_id: str
    normals: list[tuple[int, _NormalRequestT]]
    subagents: list[tuple[int, WekaSubagentEntry]]


@dataclass
class _ChildPlan:
    session_id: str
    parent_trace_id: str
    subagent_index: int
    entry: WekaSubagentEntry


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
            self._block_size = user_block_size
        elif default_block_size is not None:
            self._block_size = default_block_size
        else:
            self._block_size = 64  # matches Weka traces' default
        self._delay_cap_tracker = DelayCapTracker(
            cap_seconds=user_config.loadgen.inter_turn_delay_cap_seconds
        )
        self._use_live_assistant = Environment.DATASET.WEKA_LIVE_ASSISTANT_RESPONSES

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
        """Drop traces whose peak recorded ``input_length`` exceeds ``max_ctx``.

        Uses the per-request ``input_length`` recorded in the WEKA trace
        (cumulative context at that turn) so no client-side re-tokenization
        is required. The peak across requests is the conversation's worst
        case; any conversation exceeding it would 4xx mid-run.
        """
        kept: dict[str, list[WekaTrace]] = {}
        max_seen = 0
        for trace_id, wekas in data.items():
            peak = max(
                (
                    req.input_length
                    for req in wekas[0].requests
                    if isinstance(req, WekaNormalRequest | WekaStreamingRequest)
                ),
                default=0,
            )
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

        parent_plans: list[_ParentPlan] = []
        child_plans: list[_ChildPlan] = []
        # Track subagents whose branch was dropped during the second pass;
        # their child conversations must also be pruned.
        dropped_per_trace: dict[str, set[str]] = {}

        max_ctx = self.user_config.input.max_context_length
        if max_ctx is not None:
            data = self._filter_traces_by_max_context(data, max_ctx)

        for trace_id, wekas in data.items():
            trace = wekas[0]
            normals: list[tuple[int, _NormalRequestT]] = []
            subagents: list[tuple[int, WekaSubagentEntry]] = []
            for idx, req in enumerate(trace.requests):
                if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
                    if not self._request_passes_filters(req):
                        continue
                    normals.append((idx, req))
                else:  # WekaSubagentEntry
                    sa_index = len(subagents)
                    subagents.append((idx, req))
                    child_sid = f"{trace_id}::sa:{req.agent_id}"
                    child_plans.append(
                        _ChildPlan(
                            session_id=child_sid,
                            parent_trace_id=trace_id,
                            subagent_index=sa_index,
                            entry=req,
                        )
                    )
            parent_plans.append(_ParentPlan(trace_id, normals, subagents))

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
                    cap_seconds=cap_seconds,
                    configured_workers=configured_workers,
                    t_start=_t1,
                    model_map_per_trace=model_map_per_trace,
                )
            else:
                conversations = self._reconstruct_serial(
                    parent_plans=parent_plans,
                    child_plans=child_plans,
                    data=data,
                    dropped_per_trace=dropped_per_trace,
                    ignore_delays=ignore_delays,
                    think_time_only=think_time_only,
                    cap_seconds=cap_seconds,
                    t_start=_t1,
                    model_map_per_trace=model_map_per_trace,
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
        dropped_per_trace: dict[str, set[str]],
        ignore_delays: bool,
        think_time_only: bool,
        cap_seconds: float | None,
        t_start: float,
        model_map_per_trace: dict[str, dict[str, str]],
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

            model_map = model_map_per_trace.get(plan.trace_id, {})

            # raw_messages carries delta-encoded segments per turn; the
            # endpoint accumulates across turns at request time, with
            # ``reset_context`` flagging non-monotonic LCP cuts.
            trace = data[plan.trace_id][0]
            conv = Conversation(
                session_id=plan.trace_id,
                context_mode=self._resolved_context_mode(),
            )
            recon = ConversationReconstructor(
                block_size=self._block_size,
                decode_block_tokens=self._decode_block_tokens,
                sample_partial_tail_tokens=self.sample_partial_tail_tokens,
                decode_tokens_to_text=self._decode_tokens_to_text,
                bpe_stable_terminator_tokens=self.bpe_stable_terminator_tokens,
                emit_assistant_segments=not self._use_live_assistant,
            )

            # First pass: emit turns from normal requests; track outer-index → turn-pos.
            outer_to_turn_pos: dict[int, int] = {}
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
                t_ms = req.t * 1000.0
                if k == 0:
                    delay_ms: float | None = None
                elif think_time_only and req.think_time is not None:
                    delay_ms = req.think_time * 1000.0
                else:
                    delay_ms = t_ms - plan.normals[k - 1][1].t * 1000.0
                if delay_ms is not None:
                    delay_ms = self._delay_cap_tracker.clamp(delay_ms)
                delta = recon.turn_delta()
                conv.turns.append(
                    Turn(
                        timestamp=None if ignore_delays else t_ms,
                        delay=None if ignore_delays else delay_ms,
                        model=model_map.get(req.model, req.model),
                        max_tokens=self._cap_output(req),
                        raw_messages=delta.delta_messages,
                        reset_context=delta.reset_context,
                    )
                )
                outer_to_turn_pos[outer_idx] = len(conv.turns) - 1

            # Group subagents by (preceding, following) anchor pair so adjacent
            # subagents collapse into one multi-child branch.
            groups: dict[tuple[int | None, int | None], list[WekaSubagentEntry]] = (
                defaultdict(list)
            )
            group_order: list[tuple[int | None, int | None]] = []
            dropped_sa_agent_ids: set[str] = set()

            for sa_outer_idx, sa_entry in plan.subagents:
                preceding = max(
                    (pos for oi, pos in outer_to_turn_pos.items() if oi < sa_outer_idx),
                    default=None,
                )
                following = min(
                    (pos for oi, pos in outer_to_turn_pos.items() if oi > sa_outer_idx),
                    default=None,
                )
                if preceding is None:
                    _logger.info(
                        f"Dropping subagent '{sa_entry.agent_id}' from trace "
                        f"{plan.trace_id}: no preceding parent turn"
                    )
                    dropped_sa_agent_ids.add(sa_entry.agent_id)
                    continue
                key = (preceding, following)
                if key not in groups:
                    group_order.append(key)
                groups[key].append(sa_entry)

            for preceding, following in group_order:
                entries = groups[(preceding, following)]
                child_sids = [f"{plan.trace_id}::sa:{e.agent_id}" for e in entries]
                branch_id = f"{plan.trace_id}:spawn:{entries[0].agent_id}"
                is_background = following is None
                conv.branches.append(
                    ConversationBranchInfo(
                        branch_id=branch_id,
                        child_conversation_ids=child_sids,
                        mode=ConversationBranchMode.SPAWN,
                        is_background=is_background,
                    )
                )
                conv.turns[preceding].branch_ids.append(branch_id)
                if following is not None:
                    conv.turns[following].prerequisites.append(
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN,
                            branch_id=branch_id,
                        )
                    )
            dropped_per_trace[plan.trace_id] = dropped_sa_agent_ids
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
            if cp.entry.agent_id in dropped_per_trace.get(cp.parent_trace_id, set()):
                continue
            child_model_map = model_map_per_trace.get(cp.parent_trace_id, {})
            # Subagent has its own scope: tool_tokens/system_tokens differ from
            # the parent, and its block_cache must not leak across subagents.
            pg = self.prompt_generator
            pg._cache.clear()
            pg._hash_id_corpus_rng.set_trace_id(cp.session_id)

            child_recon = ConversationReconstructor(
                block_size=self._block_size,
                decode_block_tokens=self._decode_block_tokens,
                sample_partial_tail_tokens=self.sample_partial_tail_tokens,
                decode_tokens_to_text=self._decode_tokens_to_text,
                bpe_stable_terminator_tokens=self.bpe_stable_terminator_tokens,
                emit_assistant_segments=not self._use_live_assistant,
            )
            child_conv = Conversation(
                session_id=cp.session_id,
                context_mode=self._resolved_context_mode(),
            )
            for k, creq in enumerate(cp.entry.requests):
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
                    prev_creq = cp.entry.requests[k - 1]
                    child_recon.advance_turn(
                        prev_hash_ids=prev_creq.hash_ids,
                        prev_in_tokens=prev_creq.input_length,
                        prev_out_tokens=prev_creq.output_length,
                        curr_hash_ids=creq.hash_ids,
                        curr_in_tokens=creq.input_length,
                        seed=seed,
                    )
                t_ms = creq.t * 1000.0
                if k == 0:
                    child_delay_ms: float | None = None
                elif think_time_only and creq.think_time is not None:
                    child_delay_ms = creq.think_time * 1000.0
                else:
                    child_delay_ms = t_ms - cp.entry.requests[k - 1].t * 1000.0
                if child_delay_ms is not None:
                    child_delay_ms = self._delay_cap_tracker.clamp(child_delay_ms)
                child_delta = child_recon.turn_delta()
                child_conv.turns.append(
                    Turn(
                        timestamp=None if ignore_delays else t_ms,
                        delay=None if ignore_delays else child_delay_ms,
                        model=child_model_map.get(creq.model, creq.model),
                        max_tokens=creq.output_length,
                        raw_messages=child_delta.delta_messages,
                        reset_context=child_delta.reset_context,
                    )
                )
            conversations.append(child_conv)

        return conversations

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
            _WekaChildPayload,
            _WekaNormalRequestPayload,
            _WekaSubagentMarkerPayload,
            _WekaTraceTask,
            run_parallel_weka_reconstruction,
        )

        # Partition child plans by parent trace_id.
        children_by_trace: dict[str, list[_WekaChildPayload]] = defaultdict(list)
        for cp in child_plans:
            requests_dicts: list[_WekaNormalRequestPayload] = [
                {
                    "hash_ids": list(creq.hash_ids),
                    "input_length": creq.input_length,
                    "output_length": creq.output_length,
                    "model": creq.model,
                    "t": creq.t,
                    "think_time": getattr(creq, "think_time", None),
                }
                for creq in cp.entry.requests
            ]
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

        tasks: list[_WekaTraceTask] = []
        for plan in parent_plans:
            trace = data[plan.trace_id][0]
            normals_dicts: list[tuple[int, _WekaNormalRequestPayload]] = [
                (
                    outer_idx,
                    {
                        "hash_ids": list(req.hash_ids),
                        "input_length": req.input_length,
                        "output_length": req.output_length,
                        "model": req.model,
                        "t": req.t,
                        "think_time": getattr(req, "think_time", None),
                        "capped_output_length": self._cap_output(req),
                    },
                )
                for outer_idx, req in plan.normals
            ]
            subagents_dicts: list[tuple[int, _WekaSubagentMarkerPayload]] = [
                (
                    outer_idx,
                    {
                        "agent_id": sa.agent_id,
                        "tool_tokens": sa.tool_tokens,
                        "system_tokens": sa.system_tokens,
                    },
                )
                for outer_idx, sa in plan.subagents
            ]
            tasks.append(
                _WekaTraceTask(
                    trace_id=plan.trace_id,
                    parent={
                        "normals": normals_dicts,
                        "subagents": subagents_dicts,
                        "tool_tokens": trace.tool_tokens,
                        "system_tokens": trace.system_tokens,
                    },
                    children=children_by_trace.get(plan.trace_id, []),
                    cap_seconds=cap_seconds,
                    ignore_delays=ignore_delays,
                    think_time_only=think_time_only,
                    model_map=model_map_per_trace.get(plan.trace_id, {}),
                    emit_assistant_segments=not self._use_live_assistant,
                )
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
                    )
                )
            for branch in result["branches"]:
                parent_conv.branches.append(
                    ConversationBranchInfo(
                        branch_id=branch["branch_id"],
                        child_conversation_ids=branch["child_session_ids"],
                        mode=ConversationBranchMode.SPAWN,
                        is_background=branch["is_background"],
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
                        )
                    )
                conversations.append(child_conv)

        return conversations
