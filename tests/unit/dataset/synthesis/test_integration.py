# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SynthesisIntegration class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.config import SynthesisConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.synthesis.integration import SynthesisIntegration


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    # encode returns list of token ids (simple: 1 token per char)
    tokenizer.encode = lambda text: list(range(len(text)))
    tokenizer.decode = lambda tokens: "x" * len(tokens)
    return tokenizer


@pytest.fixture
def mock_prompt_generator() -> MagicMock:
    """Create a mock prompt generator."""
    generator = MagicMock()
    generator.generate = MagicMock(return_value="generated prompt text")
    return generator


@pytest.fixture
def synthesis_config() -> SynthesisConfig:
    """Create a synthesis config for testing."""
    return SynthesisConfig(
        speedup_ratio=1.0,
        prefix_len_multiplier=1.0,
        prefix_root_multiplier=1,
        prompt_len_multiplier=1.0,
        block_size=16,
    )


@pytest.fixture
def integration(
    synthesis_config: SynthesisConfig,
    mock_tokenizer: MagicMock,
    mock_prompt_generator: MagicMock,
) -> SynthesisIntegration:
    """Create a SynthesisIntegration instance for testing."""
    return SynthesisIntegration(
        synthesis_config=synthesis_config,
        tokenizer=mock_tokenizer,
        prompt_generator=mock_prompt_generator,
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    """Create a sample conversation for testing."""
    return Conversation(
        session_id="test-session-1",
        turns=[
            Turn(
                texts=[Text(name="text", contents=["Hello, how are you?"])],
                max_tokens=50,
                timestamp=1000,
                delay=100,
            ),
            Turn(
                texts=[Text(name="text", contents=["I need help with coding."])],
                max_tokens=100,
                timestamp=2000,
            ),
        ],
    )


class TestSynthesisIntegration:
    """Tests for SynthesisIntegration class."""

    def test_synthesize_conversations(
        self, integration: SynthesisIntegration, sample_conversation: Conversation
    ) -> None:
        """Test synthesize_conversations returns synthesized conversations and traces."""
        conversations = [sample_conversation]

        result_conversations, result_traces = integration.synthesize_conversations(
            conversations
        )

        assert len(result_conversations) >= 1
        assert len(result_traces) >= 1
        assert isinstance(result_conversations[0], Conversation)
        assert isinstance(result_traces[0], dict)

    def test_write_synthesized_traces(self, integration: SynthesisIntegration) -> None:
        """Test write_synthesized_traces writes traces to file."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2, 3]},
            {"input_length": 150, "output_length": 30, "hash_ids": [1, 2]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "traces.jsonl"
            integration.write_synthesized_traces(traces, output_path)

            assert output_path.exists()
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_conversations_to_mooncake_traces(
        self, integration: SynthesisIntegration, sample_conversation: Conversation
    ) -> None:
        """Test _conversations_to_mooncake_traces converts correctly."""
        conversations = [sample_conversation]

        traces = integration._conversations_to_mooncake_traces(
            conversations, generate_hash_ids=True
        )

        assert len(traces) == 2  # 2 turns in conversation
        assert all("input_length" in t for t in traces)
        assert all("output_length" in t for t in traces)
        assert all("session_id" in t for t in traces)

    def test_turn_to_mooncake_trace(self, integration: SynthesisIntegration) -> None:
        """Test _turn_to_mooncake_trace converts a single turn."""
        turn = Turn(
            texts=[Text(name="text", contents=["Test prompt content"])],
            max_tokens=64,
            timestamp=5000,
            delay=500,
        )

        trace = integration._turn_to_mooncake_trace(
            turn, session_id="test-session", generate_hash_ids=True
        )

        assert trace["session_id"] == "test-session"
        assert trace["output_length"] == 64
        assert trace["timestamp"] == 5000
        assert trace["delay"] == 500
        assert "input_length" in trace

    def test_turn_to_mooncake_trace_without_hash_ids(
        self, integration: SynthesisIntegration
    ) -> None:
        """Test _turn_to_mooncake_trace without generating hash_ids."""
        turn = Turn(
            texts=[Text(name="text", contents=["Test"])],
            max_tokens=32,
        )

        trace = integration._turn_to_mooncake_trace(
            turn, session_id="test", generate_hash_ids=False
        )

        assert "hash_ids" not in trace
        assert trace["output_length"] == 32

    def test_generate_hash_ids(self, integration: SynthesisIntegration) -> None:
        """Test _generate_hash_ids generates hash IDs from text."""
        # Text long enough to produce multiple blocks (block_size=16)
        text = "a" * 50  # Should produce ~3 blocks

        hash_ids = integration._generate_hash_ids(text)

        assert isinstance(hash_ids, list)
        assert len(hash_ids) >= 1
        assert all(isinstance(h, int) for h in hash_ids)

    def test_generate_hash_ids_empty_text(
        self, integration: SynthesisIntegration
    ) -> None:
        """Test _generate_hash_ids with empty text returns empty list."""
        hash_ids = integration._generate_hash_ids("")

        assert hash_ids == []

    def test_mooncake_traces_to_conversations(
        self, integration: SynthesisIntegration
    ) -> None:
        """Test _mooncake_traces_to_conversations groups by session_id."""
        traces = [
            {"input_length": 100, "output_length": 20, "session_id": "session-1"},
            {"input_length": 150, "output_length": 30, "session_id": "session-1"},
            {"input_length": 120, "output_length": 25, "session_id": "session-2"},
        ]

        conversations = integration._mooncake_traces_to_conversations(traces)

        assert len(conversations) == 2
        session_ids = {c.session_id for c in conversations}
        assert session_ids == {"session-1", "session-2"}

    def test_mooncake_trace_to_turn(
        self,
        integration: SynthesisIntegration,
        mock_prompt_generator: MagicMock,
    ) -> None:
        """Test _mooncake_trace_to_turn creates Turn from trace."""
        trace = {
            "input_length": 512,
            "output_length": 64,
            "hash_ids": [1, 2, 3],
            "timestamp": 1000,
            "delay": 500,
        }

        turn = integration._mooncake_trace_to_turn(trace)

        assert isinstance(turn, Turn)
        assert turn.max_tokens == 64
        assert turn.timestamp == 1000
        assert turn.delay == 500
        mock_prompt_generator.generate.assert_called()

    def test_mooncake_trace_to_turn_without_hash_ids(
        self,
        integration: SynthesisIntegration,
        mock_prompt_generator: MagicMock,
    ) -> None:
        """Test _mooncake_trace_to_turn without hash_ids."""
        trace = {
            "input_length": 256,
            "output_length": 32,
        }

        turn = integration._mooncake_trace_to_turn(trace)

        assert isinstance(turn, Turn)
        assert turn.max_tokens == 32
        # Should call generate without hash_ids
        mock_prompt_generator.generate.assert_called()
