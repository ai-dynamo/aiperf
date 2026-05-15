# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.dataset.loader.accuracy_dataset_loader import AccuracyDatasetLoader
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_user_config(system_prompt: str | None = None) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(
            benchmark=AccuracyBenchmarkType.MMLU,
            system_prompt=system_prompt,
        ),
    )


def _make_problem(
    prompt: str = "What is 2+2?", ground_truth: str = "A"
) -> BenchmarkProblem:
    return BenchmarkProblem(
        prompt=prompt,
        ground_truth=ground_truth,
        task="math",
    )


class TestAccuracyDatasetLoaderSystemPrompt:
    def test_system_prompt_prepended_to_texts_for_completions(self) -> None:
        """system_prompt is prepended to turn.texts so the completions endpoint receives it."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt="Answer concisely.")
        )
        conversations = loader._convert_to_conversations(
            [_make_problem(prompt="What is 2+2?")]
        )

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Answer concisely.\n\nWhat is 2+2?"

    def test_system_prompt_also_in_raw_messages_for_chat(self) -> None:
        """system_prompt is the first raw_message so the chat endpoint also receives it."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt="Answer concisely.")
        )
        conversations = loader._convert_to_conversations([_make_problem()])

        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        assert turn.raw_messages[0] == {
            "role": "system",
            "content": "Answer concisely.",
        }

    def test_no_system_prompt_leaves_texts_unchanged(self) -> None:
        """Without a system_prompt, turn.texts contains only the problem prompt."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt=None)
        )
        conversations = loader._convert_to_conversations(
            [_make_problem(prompt="What is 2+2?")]
        )

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "What is 2+2?"

    def test_no_system_prompt_raw_messages_has_no_system_role(self) -> None:
        """Without a system_prompt, raw_messages contains only the user turn."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt=None)
        )
        conversations = loader._convert_to_conversations([_make_problem()])

        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        roles = [m["role"] for m in turn.raw_messages]
        assert "system" not in roles

    @pytest.mark.asyncio
    async def test_load_raises_on_empty_problems(self) -> None:
        """load() raises ValueError when the benchmark returns no problems."""
        loader = AccuracyDatasetLoader(user_config=_make_user_config())

        with (
            patch(
                "aiperf.dataset.loader.accuracy_dataset_loader.load_benchmark_problems",
                return_value=[],
            ),
            pytest.raises(ValueError, match="returned 0 problems"),
        ):
            await loader.load()

    def test_system_prompt_applied_to_all_problems(self) -> None:
        """system_prompt is prepended to every problem's texts, not just the first."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt="Be brief.")
        )
        problems = [_make_problem(prompt=f"Q{i}") for i in range(3)]
        conversations = loader._convert_to_conversations(problems)

        for i, conv in enumerate(conversations):
            assert conv.turns[0].texts[0].contents[0] == f"Be brief.\n\nQ{i}"


class TestAccuracyDatasetLoaderRawMessages:
    def test_prebuilt_raw_messages_passed_through_unchanged(self) -> None:
        """When problem.raw_messages is set, Turn.raw_messages receives them as-is."""
        loader = AccuracyDatasetLoader(user_config=_make_user_config())
        problem = BenchmarkProblem(
            prompt="What is 2+2?",
            ground_truth="A",
            task="math",
            raw_messages=[
                {"role": "user", "content": "Q1?"},
                {"role": "assistant", "content": " A"},
                {"role": "user", "content": "What is 2+2?"},
            ],
        )

        conversations = loader._convert_to_conversations([problem])

        turn = conversations[0].turns[0]
        assert turn.raw_messages == [
            {"role": "user", "content": "Q1?"},
            {"role": "assistant", "content": " A"},
            {"role": "user", "content": "What is 2+2?"},
        ]

    def test_accuracy_ground_truth_stamped_on_conversation(self) -> None:
        """_convert_to_conversations stamps problem.ground_truth onto Conversation."""
        loader = AccuracyDatasetLoader(user_config=_make_user_config())
        problem = _make_problem(ground_truth="C")

        conversations = loader._convert_to_conversations([problem])

        assert conversations[0].accuracy_ground_truth == "C"

    def test_accuracy_task_stamped_on_conversation(self) -> None:
        """_convert_to_conversations stamps problem.task onto Conversation."""
        loader = AccuracyDatasetLoader(user_config=_make_user_config())
        problem = BenchmarkProblem(
            prompt="Q?", ground_truth="B", task="abstract_algebra"
        )

        conversations = loader._convert_to_conversations([problem])

        assert conversations[0].accuracy_task == "abstract_algebra"

    def test_metadata_propagates_accuracy_fields(self) -> None:
        """Conversation.metadata() carries accuracy fields into ConversationMetadata."""
        loader = AccuracyDatasetLoader(user_config=_make_user_config())
        problem = BenchmarkProblem(prompt="Q?", ground_truth="D", task="virology")

        conversations = loader._convert_to_conversations([problem])
        meta = conversations[0].metadata()

        assert meta.accuracy_ground_truth == "D"
        assert meta.accuracy_task == "virology"

    def test_system_prompt_prepended_to_prebuilt_raw_messages(self) -> None:
        """system_prompt is inserted at index 0 of pre-built raw_messages."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt="Be concise.")
        )
        problem = BenchmarkProblem(
            prompt="What is 2+2?",
            ground_truth="A",
            task="math",
            raw_messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
        )

        conversations = loader._convert_to_conversations([problem])

        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        assert turn.raw_messages[0] == {"role": "system", "content": "Be concise."}
        assert turn.raw_messages[1] == {"role": "user", "content": "What is 2+2?"}


class TestDefaultSystemPromptResolution:
    """Per-benchmark ``default_system_prompt`` metadata injection.

    The trt-llm AIME recipe sets a ``Please reason step by step, and
    put your final answer within \\boxed{}.`` system prompt via its
    ``aime_test.json``. We mirror that by reading
    ``default_system_prompt`` from plugin metadata when the user
    hasn't passed ``--accuracy-system-prompt``. Documented in
    ``docs/accuracy/accuracy-benchmarking.md`` so users see what
    aiperf injects on their behalf.
    """

    def test_user_override_wins_over_metadata_default(self) -> None:
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt="user-supplied")
        )
        problem = _make_problem()
        with patch(
            "aiperf.dataset.loader.accuracy_dataset_loader.plugins.get_metadata",
            return_value={"default_system_prompt": "metadata-default"},
        ):
            conversations = loader._convert_to_conversations([problem])
        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        assert turn.raw_messages[0] == {
            "role": "system",
            "content": "user-supplied",
        }

    def test_metadata_default_used_when_user_unset(self) -> None:
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt=None)
        )
        problem = _make_problem()
        with patch(
            "aiperf.dataset.loader.accuracy_dataset_loader.plugins.get_metadata",
            return_value={
                "default_system_prompt": "Please reason step by step, "
                "and put your final answer within \\boxed{}."
            },
        ):
            conversations = loader._convert_to_conversations([problem])
        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        assert turn.raw_messages[0]["role"] == "system"
        assert "reason step by step" in turn.raw_messages[0]["content"]

    def test_no_system_prompt_when_neither_set(self) -> None:
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt=None)
        )
        problem = _make_problem()
        with patch(
            "aiperf.dataset.loader.accuracy_dataset_loader.plugins.get_metadata",
            return_value={},
        ):
            conversations = loader._convert_to_conversations([problem])
        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        roles = [m["role"] for m in turn.raw_messages]
        assert "system" not in roles

    def test_empty_string_metadata_default_treated_as_none(self) -> None:
        """Empty-string metadata default is dropped (not injected as a
        zero-length system message)."""
        loader = AccuracyDatasetLoader(
            user_config=_make_user_config(system_prompt=None)
        )
        problem = _make_problem()
        with patch(
            "aiperf.dataset.loader.accuracy_dataset_loader.plugins.get_metadata",
            return_value={"default_system_prompt": ""},
        ):
            conversations = loader._convert_to_conversations([problem])
        turn = conversations[0].turns[0]
        roles = [m["role"] for m in turn.raw_messages]
        assert "system" not in roles
