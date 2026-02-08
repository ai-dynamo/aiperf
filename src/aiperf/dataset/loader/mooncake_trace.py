# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.generator.parallel_decode import parallel_decode
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.synthesizer import Synthesizer
from aiperf.plugin.enums import DatasetSamplingStrategy


class MooncakeTraceDatasetLoader(BaseFileLoader):
    """A dataset loader that loads Mooncake trace data from a file.

    Loads Mooncake trace data from a file and converts the data into
    a list of conversations for dataset manager.

    Each line in the file represents a single trace entry and will be
    converted to a separate conversation with a unique session ID.

    Example:
    Fixed schedule version
    ```json
    {"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
    ```

    Multi-turn version
    ```json
    {"session_id": "abc-123", "input_length": 300, "output_length": 40},
    {"session_id": "abc-123", "delay": 2, "input_length": 150, "output_length": 20}
    ```
    """

    def __init__(
        self,
        *,
        filename: str,
        prompt_generator: PromptGenerator,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self.prompt_generator = prompt_generator
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        self._start_offset = user_config.input.fixed_schedule_start_offset
        self._end_offset = user_config.input.fixed_schedule_end_offset
        self._max_isl = user_config.input.synthesis.max_isl
        self._max_osl = user_config.input.synthesis.max_osl

        # Store tokenizer name and block size for parallel decode
        self._tokenizer_name = (
            user_config.tokenizer.name or user_config.endpoint.model_names[0]
        )
        self._block_size = user_config.input.prompt.input_tokens.block_size

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        For mooncake trace data, simply validate the data against the MooncakeTrace model.
        This will handle all of the validation logic for the different input combinations.
        """
        if data is None:
            return False

        try:
            MooncakeTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for MooncakeTrace."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        """Load Mooncake trace data from a file.

        Returns:
            A dictionary of session_id and list of Mooncake trace data.
        """
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                trace_data = MooncakeTrace.model_validate_json(line)

                # Skip traces before or after the fixed schedule offset
                if (
                    trace_data.timestamp is not None
                    and not self._timestamp_within_offsets(trace_data.timestamp)
                ):
                    self._skipped_traces += 1
                    continue

                # Filter by max_isl if configured
                if (
                    self._max_isl is not None
                    and trace_data.input_length is not None
                    and trace_data.input_length > self._max_isl
                ):
                    self._skipped_max_isl += 1
                    continue

                # Cap by max_osl if configured
                if (
                    self._max_osl is not None
                    and trace_data.output_length is not None
                    and trace_data.output_length > self._max_osl
                ):
                    self._capped_max_osl += 1
                    # Only cap it, do not skip the trace
                    trace_data.output_length = self._max_osl

                session_id = trace_data.session_id or self.session_id_generator.next()
                data[session_id].append(trace_data)

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
                f"{self._capped_max_osl:,} traces exceeded max_osl of {self._max_osl} and were capped to {self._max_osl}"
            )
        self.debug(lambda: f"Loaded {len(data):,} traces from {self.filename}")

        # Apply synthesis if needed
        synthesis_config = self.user_config.input.synthesis
        if synthesis_config.should_synthesize():
            data = self._apply_synthesis(data)

        return data

    def _timestamp_within_offsets(self, timestamp: int) -> bool:
        return (self._start_offset is None or timestamp >= self._start_offset) and (
            self._end_offset is None or timestamp <= self._end_offset
        )

    def convert_to_conversations(
        self, data: dict[str, list[MooncakeTrace]]
    ) -> list[Conversation]:
        """Convert all the Mooncake trace data to conversation objects.

        Uses a three-phase approach for optimal performance:
        1. Phase 1: Build token sequences, checking string cache first
        2. Phase 2: Batch parallel decode for cache misses
        3. Phase 3: Assemble final conversation objects

        Args:
            data: A dictionary of session_id and list of Mooncake trace data.

        Returns:
            A list of conversations.
        """
        # Phase 1: Build token sequences and identify cache misses
        # pending_decodes: list of (session_id, trace_idx, tokens, cache_key)
        pending_decodes: list[tuple[str, int, list[int], tuple]] = []
        # conversations_data: session_id -> list of (trace, prompt or None)
        conversations_data: dict[str, list[tuple[MooncakeTrace, str | None]]] = {}

        for session_id, traces in data.items():
            conversations_data[session_id] = []
            for idx, trace in enumerate(traces):
                if trace.text_input is not None:
                    # Already a string, no decode needed
                    conversations_data[session_id].append((trace, trace.text_input))
                else:
                    hash_ids = trace.hash_ids or []
                    if hash_ids:
                        # Check string cache first
                        cache_key = (
                            tuple(hash_ids),
                            trace.input_length,
                            self._block_size,
                        )
                        if cache_key in self.prompt_generator._decoded_cache:
                            # Cache hit - use cached prompt
                            prompt = self.prompt_generator._decoded_cache[cache_key]
                            conversations_data[session_id].append((trace, prompt))
                        else:
                            # Cache miss - build tokens for batch decode
                            tokens = self.prompt_generator._build_token_sequence(
                                trace.input_length, hash_ids, self._block_size
                            )
                            pending_decodes.append((session_id, idx, tokens, cache_key))
                            conversations_data[session_id].append(
                                (trace, None)
                            )  # Placeholder
                    else:
                        # No hash_ids - use normal generation (already optimized)
                        prompt = self.prompt_generator.generate(
                            mean=trace.input_length, stddev=0, hash_ids=[]
                        )
                        conversations_data[session_id].append((trace, prompt))

        # Phase 2: Batch parallel decode for all cache misses
        if pending_decodes:
            self.debug(
                lambda: f"Parallel decoding {len(pending_decodes)} prompts "
                f"({len(data)} conversations)"
            )
            token_sequences = [p[2] for p in pending_decodes]
            decoded_prompts = parallel_decode(token_sequences, self._tokenizer_name)

            # Fill in placeholders and update cache
            for (session_id, idx, _, cache_key), prompt in zip(
                pending_decodes, decoded_prompts, strict=True
            ):
                # Update decoded cache for future reuse
                self.prompt_generator._decoded_cache[cache_key] = prompt
                # Update placeholder in conversations_data
                trace, _ = conversations_data[session_id][idx]
                conversations_data[session_id][idx] = (trace, prompt)

        # Phase 3: Build final conversation objects
        conversations = []
        for session_id, trace_prompt_pairs in conversations_data.items():
            conversation = Conversation(session_id=session_id)
            for trace, prompt in trace_prompt_pairs:
                turn = Turn(
                    timestamp=trace.timestamp,
                    delay=trace.delay,
                    texts=[Text(name="text", contents=[prompt])],
                    max_tokens=trace.output_length,
                )
                conversation.turns.append(turn)
            conversations.append(conversation)

        return conversations

    def _apply_synthesis(
        self, data: dict[str, list[MooncakeTrace]]
    ) -> dict[str, list[MooncakeTrace]]:
        """Apply synthesis transformations to mooncake traces in-memory.

        Args:
            data: Dictionary of session_id to list of MooncakeTrace objects.

        Returns:
            Dictionary of session_id to list of synthesized MooncakeTrace objects.
        """
        params = SynthesisParams.from_synthesis_config(
            self.user_config.input.synthesis, block_size=self._block_size
        )

        # Convert to dicts for synthesizer (exclude discriminator field "type")
        dict_data = {
            sid: [t.model_dump(exclude={"type"}, exclude_none=True) for t in traces]
            for sid, traces in data.items()
        }
        synthesized = Synthesizer(params=params).synthesize_grouped_traces(dict_data)

        return {
            sid: [MooncakeTrace.model_validate(t) for t in traces]
            for sid, traces in synthesized.items()
        }
