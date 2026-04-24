# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Text, Turn
from aiperf.config.defaults import InputTokensDefaults
from aiperf.dataset.generator.parallel_decode import parallel_decode
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.synthesizer import Synthesizer
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig, BenchmarkRun
    from aiperf.config.dataset import SynthesisConfig

TraceT = TypeVar("TraceT")


def _extract_phase_offsets(config: BenchmarkConfig) -> tuple[int | None, int | None]:
    """Extract start_offset and end_offset from the first fixed_schedule phase."""
    for phase in config.phases.values():
        start = getattr(phase, "start_offset", None)
        end = getattr(phase, "end_offset", None)
        if start is not None or end is not None:
            return start, end
    return None, None


def _get_file_dataset_synthesis(config: BenchmarkConfig) -> SynthesisConfig | None:
    """Get synthesis config from the default dataset if it's a FileDataset."""
    from aiperf.config.dataset import FileDataset

    dataset = config.get_default_dataset()
    if isinstance(dataset, FileDataset):
        return dataset.synthesis
    return None


def _synthesis_should_apply(synthesis: SynthesisConfig | None) -> bool:
    """Check if synthesis should be triggered based on non-default values."""
    if synthesis is None:
        return False
    return (
        synthesis.speedup_ratio != 1.0
        or synthesis.prefix_len_multiplier != 1.0
        or synthesis.prefix_root_multiplier != 1
        or synthesis.prompt_len_multiplier != 1.0
    )


class BaseTraceDatasetLoader(BaseFileLoader, Generic[TraceT]):
    """Base class for trace dataset loaders with hash_ids-based prompt generation.

    Provides common infrastructure for loading trace-format datasets
    (Mooncake, Bailian, etc.) including shared initialization, timestamp
    filtering, 3-phase prompt generation with parallel decode, and
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
        filename: str,
        prompt_generator: PromptGenerator,
        run: BenchmarkRun,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, run=run, **kwargs)
        self.prompt_generator = prompt_generator
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0

        config = run.cfg
        # Phase offsets for timestamp filtering
        self._start_offset, self._end_offset = _extract_phase_offsets(config)

        # Synthesis config from the file dataset
        self._synthesis_config = _get_file_dataset_synthesis(config)
        self._max_isl = (
            self._synthesis_config.max_isl if self._synthesis_config else None
        )
        self._max_osl = (
            self._synthesis_config.max_osl if self._synthesis_config else None
        )

        # Use the resolved tokenizer name so worker processes can load from cache
        # without needing alias resolution or network access.
        tokenizer_cfg = config.tokenizer
        self._tokenizer_name = (
            prompt_generator.tokenizer.resolved_name
            or (tokenizer_cfg.name if tokenizer_cfg else None)
            or config.get_model_names()[0]
        )
        self._trust_remote_code = (
            tokenizer_cfg.trust_remote_code if tokenizer_cfg else False
        )
        self._tokenizer_revision = tokenizer_cfg.revision if tokenizer_cfg else "main"

        # Precedence: user CLI --isl-block-size > plugin metadata default > hardcoded fallback
        dataset = config.get_default_dataset()
        prompts = getattr(dataset, "prompts", None)
        user_block_size = prompts.block_size if prompts else None
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
    def _parse_trace(self, line: str) -> TraceT:
        """Parse a single JSONL line into a typed trace object."""
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
        """Apply timestamp-window, max_isl, and max_osl filters.

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

        output_length = getattr(trace, "output_length", None)
        if (
            self._max_osl is not None
            and output_length is not None
            and output_length > self._max_osl
        ):
            self._capped_max_osl += 1
            trace.output_length = self._max_osl  # type: ignore[attr-defined]

        return True

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

    def load_dataset(self) -> dict[str, list[TraceT]]:
        """Load, filter, group, and optionally synthesize trace data.

        Template method that delegates format-specific work to subclass hooks:
        :meth:`_parse_trace`, :meth:`_preprocess_trace`, and
        :meth:`_group_traces`.
        """
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        items: list[TraceT] = []

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue

                trace = self._parse_trace(line)
                self._preprocess_trace(trace)

                if not self._filter_and_cap_trace(trace):
                    continue

                items.append(trace)

        self._log_filtering_summary()

        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} traces "
                f"across {len(data):,} sessions from {self.filename}"
            )
        )

        if _synthesis_should_apply(self._synthesis_config):
            data = self._apply_synthesis(data)

        return data

    # ------------------------------------------------------------------
    # convert_to_conversations — 3-phase prompt generation
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

        Default implementation extracts `timestamp`, `delay`, `output_length`
        via getattr, which works for both Mooncake and Bailian traces.
        """
        return Turn(
            timestamp=getattr(trace, "timestamp", None),
            delay=getattr(trace, "delay", None),
            texts=[Text(name="text", contents=[prompt])],
            max_tokens=getattr(trace, "output_length", None),
        )

    def _resolve_trace_prompt(
        self,
        trace: TraceT,
        session_id: str,
        idx: int,
        pending_decodes: list[tuple[str, int, list[int], tuple]],
    ) -> tuple[TraceT, str | None]:
        """Resolve the prompt for a single trace, appending to pending_decodes on cache miss."""
        text_input = self._get_text_input(trace)
        if text_input is not None:
            return (trace, text_input)

        hash_ids: list[int] = getattr(trace, "hash_ids", None) or []
        input_length: int = getattr(trace, "input_length", 0)

        if not hash_ids:
            prompt = self.prompt_generator.generate(
                mean=input_length, stddev=0, hash_ids=[]
            )
            return (trace, prompt)

        cache_key = (tuple(hash_ids), input_length, self._block_size)
        if cache_key in self.prompt_generator._decoded_cache:
            prompt = self.prompt_generator._decoded_cache[cache_key]
            return (trace, prompt)

        tokens = self.prompt_generator._build_token_sequence(
            input_length, hash_ids, self._block_size
        )
        pending_decodes.append((session_id, idx, tokens, cache_key))
        return (trace, None)

    def _fill_pending_decodes(
        self,
        pending_decodes: list[tuple[str, int, list[int], tuple]],
        conversations_data: dict[str, list[tuple[TraceT, str | None]]],
        num_conversations: int,
    ) -> None:
        """Batch-decode all cache misses and fill resolved prompts back into conversations_data."""
        if not pending_decodes:
            return

        self.debug(
            lambda: f"Parallel decoding {len(pending_decodes)} prompts "
            f"({num_conversations} conversations)"
        )
        token_sequences = [p[2] for p in pending_decodes]
        decoded_prompts = parallel_decode(
            token_sequences,
            self._tokenizer_name,
            trust_remote_code=self._trust_remote_code,
            revision=self._tokenizer_revision,
        )

        for (session_id, idx, _, cache_key), prompt in zip(
            pending_decodes, decoded_prompts, strict=True
        ):
            self.prompt_generator._decoded_cache[cache_key] = prompt
            trace, _ = conversations_data[session_id][idx]
            conversations_data[session_id][idx] = (trace, prompt)

    def _assemble_conversations(
        self,
        conversations_data: dict[str, list[tuple[TraceT, str | None]]],
    ) -> list[Conversation]:
        """Assemble final Conversation objects from resolved trace/prompt pairs."""
        conversations: list[Conversation] = []
        for session_id, trace_prompt_pairs in conversations_data.items():
            traces_in_session = [trace for trace, _ in trace_prompt_pairs]
            context_mode = self._infer_context_mode(traces_in_session)

            conversation = Conversation(
                session_id=session_id, context_mode=context_mode
            )
            for trace, prompt in trace_prompt_pairs:
                conversation.turns.append(self._build_turn(trace, prompt))
            conversations.append(conversation)
        return conversations

    def convert_to_conversations(
        self, data: dict[str, list[TraceT]]
    ) -> list[Conversation]:
        """Convert trace sessions to :class:`Conversation` objects.

        Uses a three-phase approach for optimal performance:

        1. Build token sequences, checking the string cache first.
        2. Batch parallel decode for all cache misses.
        3. Assemble final :class:`Conversation` objects.
        """
        # Phase 1: Build token sequences and identify cache misses
        pending_decodes: list[tuple[str, int, list[int], tuple]] = []
        conversations_data: dict[str, list[tuple[TraceT, str | None]]] = {}

        for session_id, traces in data.items():
            conversations_data[session_id] = [
                self._resolve_trace_prompt(trace, session_id, idx, pending_decodes)
                for idx, trace in enumerate(traces)
            ]

        # Phase 2: Batch parallel decode for all cache misses
        self._fill_pending_decodes(pending_decodes, conversations_data, len(data))

        # Phase 3: Build final conversation objects
        return self._assemble_conversations(conversations_data)

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
            self._synthesis_config, block_size=self._block_size
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
