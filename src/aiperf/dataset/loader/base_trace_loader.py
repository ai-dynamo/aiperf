# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Text, Turn
from aiperf.config.dataset.defaults import InputTokensDefaults
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.hash_ids_synthesis import (
    HashIdsPromptRequest,
    HashIdsPromptSynthesisMixin,
)
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.synthesizer import Synthesizer
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

TraceT = TypeVar("TraceT")


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file content (first 16 hex chars).

    Falls back to hashing the filepath string if the file cannot be read.
    Used as the per-file ``trace_id`` scope for ``HashIdRandomGenerator`` so
    that two different trace files with overlapping ``hash_id`` values
    produce different content.
    """
    try:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except (OSError, TypeError):
        return hashlib.sha256(str(filepath).encode()).hexdigest()[:16]


def _has_meaningful_synthesis(synthesis: Any) -> bool:
    """Return True when SynthesisConfig has any non-default transform set.

    Trace loaders only invoke the Synthesizer when the user actually asked
    for a transformation. Defaults: speedup_ratio=1.0,
    prefix_len_multiplier=1.0, prefix_root_multiplier=1,
    prompt_len_multiplier=1.0, output_len_multiplier=1.0.
    """
    if synthesis is None:
        return False
    return (
        getattr(synthesis, "speedup_ratio", 1.0) != 1.0
        or getattr(synthesis, "prefix_len_multiplier", 1.0) != 1.0
        or getattr(synthesis, "prefix_root_multiplier", 1) != 1
        or getattr(synthesis, "prompt_len_multiplier", 1.0) != 1.0
        or getattr(synthesis, "output_len_multiplier", 1.0) != 1.0
    )


class BaseTraceDatasetLoader(
    HashIdsPromptSynthesisMixin, BaseFileLoader, Generic[TraceT]
):
    """Base class for trace dataset loaders with hash_ids-based prompt generation.

    Provides common infrastructure for loading trace-format datasets
    (Mooncake, Bailian, etc.) including shared initialization, timestamp
    filtering, two-phase prompt generation with parallel decode, and
    synthesis integration.

    Subclasses must implement:
    - `can_load`: data format detection
    - `load_dataset`: JSONL parsing and session grouping
    - `_synthesis_exclude_fields`: fields to strip before synthesis
    - `_reconstruct_traces`: rebuild typed traces from synthesized dicts
    """

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        prompt_generator: PromptGenerator,
        run: BenchmarkRun | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, run=run, **kwargs)
        self.prompt_generator = prompt_generator
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        self._delay_cap_tracker = DelayCapTracker(
            cap_seconds=getattr(
                self.run.cfg.get_default_dataset(),
                "inter_turn_delay_cap_seconds",
                None,
            )
        )
        self._trace_id: str = ""

        # Fixed-schedule timestamp window lives on FixedSchedulePhase entries.
        # Read from the first profiling phase that exposes it (if any).
        start_offset: int | None = None
        end_offset: int | None = None
        for phase in self.run.cfg.phases:
            phase_start = getattr(phase, "start_offset", None)
            phase_end = getattr(phase, "end_offset", None)
            if phase_start is not None or phase_end is not None:
                start_offset = phase_start
                end_offset = phase_end
                break
        self._start_offset = start_offset
        self._end_offset = end_offset

        # Synthesis lives on FileDataset.synthesis; max_isl filters traces and
        # max_osl caps final traces after optional synthesis.
        dataset = self.run.cfg.get_default_dataset()
        synthesis = getattr(dataset, "synthesis", None)
        self._synthesis = synthesis
        self._max_isl = getattr(synthesis, "max_isl", None) if synthesis else None
        self._max_osl = getattr(synthesis, "max_osl", None) if synthesis else None

        # Use the resolved tokenizer name so worker processes can load from cache
        # without needing alias resolution or network access.
        tokenizer_cfg = self.run.cfg.tokenizer
        model_names = self.run.cfg.get_model_names()
        self._tokenizer_name = (
            getattr(prompt_generator.tokenizer, "resolved_name", None)
            or (tokenizer_cfg.name if tokenizer_cfg is not None else None)
            or (model_names[0] if model_names else "")
        )
        self._trust_remote_code = (
            tokenizer_cfg.trust_remote_code if tokenizer_cfg is not None else False
        )
        self._tokenizer_revision = (
            tokenizer_cfg.revision if tokenizer_cfg is not None else "main"
        )

        # Precedence: user block_size (--isl-block-size) > plugin metadata default
        # > hardcoded fallback. The hash-id trace loaders (mooncake/bailian/...)
        # decode each hash_id into a block of this many tokens. The user override
        # lands on FileDataset.block_size (routed by the converter for non-weka
        # traces); a synthetic-via-file dataset carries it on prompts.block_size.
        prompts = getattr(dataset, "prompts", None)
        user_block_size = getattr(dataset, "block_size", None)
        if user_block_size is None and prompts is not None:
            user_block_size = getattr(prompts, "block_size", None)
        if user_block_size is not None:
            self._block_size = user_block_size
        elif default_block_size is not None:
            self._block_size = default_block_size
        else:
            self._block_size = InputTokensDefaults.BLOCK_SIZE

    # ------------------------------------------------------------------
    # Shared class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Trace datasets use sequential sampling to preserve timestamp order."""
        return DatasetSamplingStrategy.SEQUENTIAL

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _parse_trace(self, record: dict) -> TraceT:
        """Parse a single record dict into the trace's typed model."""
        ...

    def _preprocess_trace(self, trace: TraceT) -> None:
        """Optional hook for per-trace pre-processing (e.g. unit conversion).

        Called after parsing but before filtering. Default is a no-op.
        """
        pass

    @abstractmethod
    def _group_traces(self, items: list[TraceT]) -> dict[str, list[TraceT]]:
        """Group flat trace entries into sessions keyed by session ID."""
        ...

    # ------------------------------------------------------------------
    # Timestamp / filtering helpers
    # ------------------------------------------------------------------

    def _timestamp_within_offsets(self, timestamp: int | float) -> bool:
        """Check if a timestamp falls within configured offsets."""
        return (self._start_offset is None or timestamp >= self._start_offset) and (
            self._end_offset is None or timestamp <= self._end_offset
        )

    def _filter_and_cap_trace(self, trace: TraceT) -> bool:
        """Apply timestamp-window and max_isl filters.

        Returns `True` if the trace should be kept, `False` to skip.
        """
        timestamp = getattr(trace, "timestamp", None)
        if timestamp is not None and not self._timestamp_within_offsets(timestamp):
            self._skipped_traces += 1
            return False

        input_length = getattr(trace, "input_length", None)
        if (
            self._max_isl is not None
            and input_length is not None
            and input_length > self._max_isl
        ):
            self._skipped_max_isl += 1
            return False

        return True

    def _cap_grouped_traces_max_osl(
        self, data: dict[str, list[TraceT]]
    ) -> dict[str, list[TraceT]]:
        """Apply max_osl to final traces after optional synthesis."""
        if self._max_osl is None:
            return data

        for traces in data.values():
            for trace in traces:
                output_length = getattr(trace, "output_length", None)
                if output_length is not None and output_length > self._max_osl:
                    self._capped_max_osl += 1
                    trace.output_length = self._max_osl  # type: ignore[attr-defined]

        return data

    def _log_filtering_summary(self) -> None:
        """Emit info-level messages for any skipped or capped traces."""
        if self._skipped_traces > 0:
            self.info(
                f"Skipped {self._skipped_traces:,} traces because they were "
                f"before the start offset of {self._start_offset} or "
                f"after the end offset of {self._end_offset}"
            )
        if self._skipped_max_isl > 0:
            self.info(
                f"Skipped {self._skipped_max_isl:,} traces because input_length "
                f"exceeded max_isl of {self._max_isl}"
            )
        if self._capped_max_osl > 0:
            self.info(
                f"{self._capped_max_osl:,} traces exceeded max_osl of "
                f"{self._max_osl} and were capped to {self._max_osl}"
            )

    # ------------------------------------------------------------------
    # load_dataset — template method
    # ------------------------------------------------------------------

    def _init_trace_scope(self) -> None:
        """Set up per-file trace_id scope and clear stale block content.

        Called from :meth:`load_dataset`. Computes a content hash of the
        trace file and uses it as the ``trace_id`` for
        :class:`HashIdRandomGenerator` so the same ``hash_id`` in two
        different files produces different tokens. Also clears the
        :class:`PromptGenerator` int-keyed block cache because it is scoped
        to the previous file's ``trace_id``.

        In inline-records mode ``self.filename`` is ``None``;
        :func:`_compute_file_hash` falls back to hashing the string ``"None"``
        so this stays crash-free and deterministic without a file on disk.
        """
        self._trace_id = _compute_file_hash(self.filename)
        pg = self.prompt_generator
        pg._hash_id_corpus_rng.set_trace_id(self._trace_id)
        pg._cache.clear()
        self.debug(lambda: f"Trace ID {self._trace_id} for {self.filename}")

    def load_dataset(self) -> dict[str, list[TraceT]]:
        """Load, filter, group, and optionally synthesize trace data.

        Template method that delegates format-specific work to subclass hooks:
        :meth:`_parse_trace`, :meth:`_preprocess_trace`, and
        :meth:`_group_traces`.
        """
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        self._delay_cap_tracker.reset()
        self._init_trace_scope()
        items: list[TraceT] = []

        for record_dict in self._iter_record_dicts():
            trace = self._parse_trace(record_dict)
            self._preprocess_trace(trace)
            if not self._filter_and_cap_trace(trace):
                continue
            items.append(trace)

        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} traces "
                f"across {len(data):,} sessions "
                f"from {self.filename if self.filename else '<inline records>'}"
            )
        )

        if _has_meaningful_synthesis(self._synthesis):
            data = self._apply_synthesis(data)

        data = self._cap_grouped_traces_max_osl(data)
        self._log_filtering_summary()

        return data

    # ------------------------------------------------------------------
    # convert_to_conversations — two-phase prompt generation
    # ------------------------------------------------------------------

    def _get_text_input(self, trace: TraceT) -> str | None:
        """Return pre-existing text input, or `None` to use hash_ids generation.

        Override for traces that carry literal prompts (e.g. `MooncakeTrace.text_input`).
        Default: checks for a `text_input` attribute via getattr.
        """
        return getattr(trace, "text_input", None)

    def _infer_context_mode(
        self, traces: list[TraceT]
    ) -> ConversationContextMode | None:
        """Infer context_mode from trace data when not explicitly set.

        Override in subclasses to auto-detect based on trace content.
        Default returns None (falls through to global DELTAS_WITHOUT_RESPONSES default).
        """
        return None

    def _build_turn(self, trace: TraceT, prompt: str) -> Turn:
        """Build a :class:`Turn` from trace data and a generated prompt.

        Default implementation extracts `timestamp`, `delay`, `output_length`,
        and `extra` via getattr, which works for both Mooncake and Bailian traces.
        """
        # extra="allow" surfaces stray row keys as attributes; only a DECLARED
        # request_body field (e.g. BasetenTrace) may override the row's extra.
        request_body = (
            getattr(trace, "request_body", None)
            if "request_body" in type(trace).model_fields
            else None
        )
        return Turn(
            timestamp=getattr(trace, "timestamp", None),
            delay=self._delay_cap_tracker.clamp(getattr(trace, "delay", None)),
            texts=[Text(name="text", contents=[prompt])],
            max_tokens=getattr(trace, "output_length", None),
            extra_body=request_body or getattr(trace, "extra", None),
        )

    def convert_to_conversations(
        self, data: dict[str, list[TraceT]]
    ) -> list[Conversation]:
        """Convert trace sessions to :class:`Conversation` objects.

        Uses a two-phase approach for optimal performance:

        1. Collect hash_ids synthesis requests (skipping traces that carry a
           literal ``text_input``).
        2. Batch-synthesize prompts via the shared mixin, then assemble final
           :class:`Conversation` objects.
        """
        # Phase 1: Collect synthesis requests, short-circuiting literal prompts.
        requests: list[HashIdsPromptRequest] = []
        conversations_data: dict[str, list[tuple[TraceT, str | None]]] = {}

        for session_id, traces in data.items():
            conversations_data[session_id] = []
            for idx, trace in enumerate(traces):
                text_input = self._get_text_input(trace)
                if text_input is not None:
                    conversations_data[session_id].append((trace, text_input))
                    continue
                hash_ids: list[int] = getattr(trace, "hash_ids", None) or []
                input_length: int = getattr(trace, "input_length", 0)
                key = f"{session_id}:{idx}"
                requests.append(
                    HashIdsPromptRequest(
                        key=key, hash_ids=hash_ids, input_length=input_length
                    )
                )
                conversations_data[session_id].append((trace, None))

        prompts_by_key = self.synthesize_prompts_from_hash_ids(requests)

        # Phase 2: Build final conversation objects.
        conversations: list[Conversation] = []
        for session_id, trace_prompt_pairs in conversations_data.items():
            traces_in_session = [trace for trace, _ in trace_prompt_pairs]
            context_mode = self._infer_context_mode(traces_in_session)

            conversation = Conversation(
                session_id=session_id, context_mode=context_mode
            )
            for idx, (trace, existing) in enumerate(trace_prompt_pairs):
                prompt = (
                    existing
                    if existing is not None
                    else prompts_by_key[f"{session_id}:{idx}"]
                )
                conversation.turns.append(self._build_turn(trace, prompt))
            conversations.append(conversation)

        self._delay_cap_tracker.log_summary(logger_name=type(self).__module__)
        return conversations

    # ------------------------------------------------------------------
    # Opt-in parallel-convert path (multi-process, full session pipeline)
    # ------------------------------------------------------------------

    def convert_to_conversations_parallel(
        self,
        data: dict[str, list[TraceT]],
        *,
        num_workers: int | None = None,
        batch_size: int = 100,
    ) -> Iterator[Conversation]:
        """Yield Conversations using a multi-process worker pool.

        Opt-in alternative to the default in-process
        :meth:`convert_to_conversations`. Workers share the tokenized corpus
        via shared memory, each holds its own
        :class:`HashIdRandomGenerator` seeded with the same
        ``(base_seed, trace_id)``, and run reseed + sample + decode entirely
        in-worker. For the exact-tile and last-block-partial layouts emitted
        by Mooncake/Bailian/BurstGPT, output is byte-identical to the
        in-process path.

        Note: this path uses the simpler exact-tile / last-block-partial
        layout only (no prefix-only-with-tail support). Loaders whose
        ``input_length`` may exceed ``len(hash_ids) * block_size`` should
        keep using the in-process path.
        """
        from aiperf.dataset.loader.parallel_convert import parallel_convert

        sessions: list[tuple[str, list[dict]]] = []
        for sid, traces in data.items():
            sessions.append(
                (
                    sid,
                    [
                        {
                            "hash_ids": getattr(t, "hash_ids", None) or [],
                            "input_length": getattr(t, "input_length", 0),
                            "output_length": getattr(t, "output_length", None),
                            "timestamp": getattr(t, "timestamp", None),
                            "delay": getattr(t, "delay", None),
                            "text_input": self._get_text_input(t),
                        }
                        for t in traces
                    ],
                )
            )

        pg = self.prompt_generator
        yield from parallel_convert(
            sessions,
            tokenizer_name=self._tokenizer_name,
            corpus=pg._tokenized_corpus,
            base_seed=pg._hash_id_corpus_rng.seed,
            block_size=self._block_size,
            sep_token=pg.tokenizer.block_separation_token_id,
            trace_id=self._trace_id,
            trust_remote_code=self._trust_remote_code,
            revision=self._tokenizer_revision,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Synthesis — shared orchestration with subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _synthesis_exclude_fields(self) -> frozenset[str]:
        """Fields to exclude when serializing traces for the Synthesizer."""
        ...

    def _synthesis_dump_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for `model_dump` during synthesis serialization.

        Override to add e.g. `by_alias=True` for aliased fields.
        """
        return {}

    @abstractmethod
    def _reconstruct_traces(
        self, originals: list[TraceT], synth_dicts: list[dict[str, Any]]
    ) -> list[TraceT]:
        """Rebuild typed trace objects from synthesized dicts.

        Args:
            originals: The original traces for this session (for metadata recovery).
            synth_dicts: The synthesized dicts from the Synthesizer.
        """
        ...

    def _apply_synthesis(
        self, data: dict[str, list[TraceT]]
    ) -> dict[str, list[TraceT]]:
        """Apply synthesis transformations to traces in-memory."""
        params = SynthesisParams.from_synthesis_config(
            self._synthesis, block_size=self._block_size
        )

        exclude = self._synthesis_exclude_fields()
        dump_kwargs = self._synthesis_dump_kwargs()
        dict_data = {
            sid: [
                t.model_dump(exclude=exclude, exclude_none=True, **dump_kwargs)  # type: ignore[union-attr]
                for t in traces
            ]
            for sid, traces in data.items()
        }

        synthesized = Synthesizer(params=params).synthesize_grouped_traces(dict_data)

        return {
            sid: self._reconstruct_traces(data.get(sid, []), synth_traces)
            for sid, synth_traces in synthesized.items()
        }
