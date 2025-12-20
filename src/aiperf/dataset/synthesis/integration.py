# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration utilities for synthesis with AIPerf dataset pipeline."""

import json
from pathlib import Path

from aiperf.common.config import SynthesisConfig
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator import PromptGenerator
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher
from aiperf.dataset.synthesis.synthesizer import Synthesizer


class SynthesisIntegration(AIPerfLoggerMixin):
    """Integrates trace synthesis into the AIPerf dataset pipeline.

    Bridges between AIPerf's Conversation model and mooncake trace format,
    enabling synthesis transformations to be applied during profiling.
    """

    def __init__(
        self,
        synthesis_config: SynthesisConfig,
        tokenizer: Tokenizer,
        prompt_generator: PromptGenerator,
    ) -> None:
        """Initialize synthesis integration.

        Args:
            synthesis_config: Configuration for synthesis parameters.
            tokenizer: Tokenizer for text processing.
            prompt_generator: Generator for creating prompts from hash_ids.
        """
        super().__init__(config=None, tokenizer=tokenizer)
        self.synthesis_config = synthesis_config
        self.tokenizer = tokenizer
        self.prompt_generator = prompt_generator
        self._rolling_hasher = RollingHasher(block_size=synthesis_config.block_size)

    def synthesize_conversations(
        self,
        conversations: list[Conversation],
        is_synthetic_data: bool = False,
    ) -> tuple[list[Conversation], list[dict]]:
        """Apply synthesis to conversations.

        Args:
            conversations: Input conversations to synthesize.
            is_synthetic_data: If True, generate hash_ids before synthesis.
                Note: hash_ids are always generated if missing, regardless of this flag.

        Returns:
            Tuple of (synthesized conversations, synthesized traces for file output).
        """
        # Convert to mooncake format
        # Always generate hash_ids if missing - needed for prefix-aware synthesis
        traces = self._conversations_to_mooncake_traces(
            conversations,
            generate_hash_ids=True,
        )

        self.info(
            f"Converting {len(conversations)} conversations to {len(traces)} traces for synthesis"
        )

        # Create synthesis params from config
        params = SynthesisParams(
            speedup_ratio=self.synthesis_config.speedup_ratio,
            prefix_len_multiplier=self.synthesis_config.prefix_len_multiplier,
            prefix_root_multiplier=self.synthesis_config.prefix_root_multiplier,
            prompt_len_multiplier=self.synthesis_config.prompt_len_multiplier,
            max_isl=self.synthesis_config.max_isl,
            block_size=self.synthesis_config.block_size,
        )

        # Run synthesis
        synthesizer = Synthesizer(params=params)
        synthesized_traces = synthesizer.synthesize_traces(traces)

        self.info(f"Synthesis complete: {len(synthesized_traces)} synthesized traces")

        # Convert back to conversations
        synthesized_conversations = self._mooncake_traces_to_conversations(
            synthesized_traces
        )

        return synthesized_conversations, synthesized_traces

    def write_synthesized_traces(
        self,
        traces: list[dict],
        output_path: Path,
    ) -> None:
        """Write synthesized traces to JSONL file.

        Args:
            traces: Synthesized trace dictionaries.
            output_path: Path to output file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

        self.info(f"Wrote {len(traces)} synthesized traces to {output_path}")

    def _conversations_to_mooncake_traces(
        self,
        conversations: list[Conversation],
        generate_hash_ids: bool = False,
    ) -> list[dict]:
        """Convert Conversations to mooncake trace format.

        Args:
            conversations: List of Conversation objects.
            generate_hash_ids: If True, generate hash_ids for traces without them.

        Returns:
            List of mooncake trace dictionaries.
        """
        traces = []
        for conversation in conversations:
            for turn in conversation.turns:
                trace = self._turn_to_mooncake_trace(
                    turn,
                    conversation.session_id,
                    generate_hash_ids,
                )
                traces.append(trace)
        return traces

    def _turn_to_mooncake_trace(
        self,
        turn: Turn,
        session_id: str,
        generate_hash_ids: bool,
    ) -> dict:
        """Convert a single Turn to mooncake trace format.

        Args:
            turn: Turn object to convert.
            session_id: Session ID for the conversation.
            generate_hash_ids: Whether to generate hash_ids from text content.

        Returns:
            Mooncake trace dictionary.
        """
        # Extract text content for tokenization
        text_content = ""
        if turn.texts:
            for text in turn.texts:
                if text.contents:
                    text_content = " ".join(text.contents)
                    break

        # Calculate input_length via tokenization
        input_length = len(self.tokenizer.encode(text_content)) if text_content else 0

        trace: dict = {
            "input_length": input_length,
            "output_length": turn.max_tokens or 64,
            "session_id": session_id,
        }

        if turn.timestamp is not None:
            trace["timestamp"] = turn.timestamp
        if turn.delay is not None:
            trace["delay"] = turn.delay

        # Generate hash_ids if requested (for synthetic data)
        if generate_hash_ids and text_content:
            hash_ids = self._generate_hash_ids(text_content)
            if hash_ids:
                trace["hash_ids"] = hash_ids

        return trace

    def _generate_hash_ids(self, text: str) -> list[int]:
        """Generate hash_ids for text content using rolling hasher.

        Args:
            text: Text content to hash.

        Returns:
            List of hash IDs representing text blocks.
        """
        tokens = self.tokenizer.encode(text)
        block_size = self.synthesis_config.block_size

        # Split into blocks
        blocks = []
        for i in range(0, len(tokens), block_size):
            block_tokens = tokens[i : i + block_size]
            block_text = self.tokenizer.decode(block_tokens)
            blocks.append(block_text)

        if not blocks:
            return []

        self._rolling_hasher.reset()
        return self._rolling_hasher.hash_blocks(blocks)

    def _mooncake_traces_to_conversations(
        self,
        traces: list[dict],
    ) -> list[Conversation]:
        """Convert mooncake traces back to Conversations.

        Args:
            traces: List of mooncake trace dictionaries.

        Returns:
            List of Conversation objects.
        """
        # Group traces by session_id
        sessions: dict[str, list[dict]] = {}
        for trace in traces:
            session_id = trace.get("session_id", "default")
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(trace)

        conversations = []
        for session_id, session_traces in sessions.items():
            conversation = Conversation(session_id=session_id)
            for trace in session_traces:
                turn = self._mooncake_trace_to_turn(trace)
                conversation.turns.append(turn)
            conversations.append(conversation)

        return conversations

    def _mooncake_trace_to_turn(self, trace: dict) -> Turn:
        """Convert a single mooncake trace to a Turn.

        Args:
            trace: Mooncake trace dictionary.

        Returns:
            Turn object.
        """
        # Generate prompt from hash_ids or input_length
        hash_ids = trace.get("hash_ids")
        input_length = trace.get("input_length", 512)

        if hash_ids:
            prompt = self.prompt_generator.generate(
                mean=input_length,
                stddev=0,
                hash_ids=hash_ids,
            )
        else:
            prompt = self.prompt_generator.generate(
                mean=input_length,
                stddev=0,
            )

        turn = Turn(
            timestamp=trace.get("timestamp"),
            delay=trace.get("delay"),
            texts=[Text(name="text", contents=[prompt])],
            max_tokens=trace.get("output_length"),
        )

        return turn
