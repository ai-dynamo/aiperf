# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Injection of the verbatim ``--system-prompt`` into composed conversations.

Covers the three composers that build conversations (synthetic, custom/file,
public) and the prepend-vs-assign branch that keeps a dataset's own authored
system message intact.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import (
    DatasetSamplingStrategy,
    EndpointType,
    PublicDatasetType,
)
from tests.unit.dataset.composer.conftest import make_run
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info

SYSTEM_TEXT = "You are a production assistant."


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "system.txt"
    path.write_text(SYSTEM_TEXT)
    return path


@pytest.fixture
def multi_turn_file(tmp_path: Path) -> Path:
    """A dataset whose first turn is a system message the loader hoists."""
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"type": "multi_turn", "session_id": "s1", "turns": ['
        '{"role": "system", "text": "Dataset system."}, '
        '{"role": "user", "text": "hi"}]}\n'
    )
    return path


def _synthetic_cli(**kwargs: Any) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=3,
        prompt_input_tokens_mean=10,
        prompt_output_tokens_mean=5,
        **kwargs,
    )


class TestSyntheticComposerSystemPrompt:
    def test_sets_system_message_on_every_conversation(
        self, prompt_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        run = make_run(_synthetic_cli(system_prompt_file=str(prompt_file)))
        conversations = SyntheticDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert len(conversations) == 3
        assert all(c.system_message == SYSTEM_TEXT for c in conversations)

    def test_absent_without_the_flag(self, mock_tokenizer: Tokenizer) -> None:
        run = make_run(_synthetic_cli())
        conversations = SyntheticDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert all(c.system_message is None for c in conversations)

    def test_inline_text_matches_file_text(
        self, prompt_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        from_file = SyntheticDatasetComposer(
            run=make_run(_synthetic_cli(system_prompt_file=str(prompt_file))),
            tokenizer=mock_tokenizer,
        ).create_dataset()
        from_text = SyntheticDatasetComposer(
            run=make_run(_synthetic_cli(system_prompt=SYSTEM_TEXT)),
            tokenizer=mock_tokenizer,
        ).create_dataset()

        assert from_file[0].system_message == from_text[0].system_message


class TestCustomComposerSystemPrompt:
    def test_prepends_to_authored_system_message(
        self, prompt_file: Path, multi_turn_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        """The dataset's own system message is kept, not replaced."""
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(multi_turn_file),
                system_prompt_file=str(prompt_file),
            )
        )
        conversations = CustomDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert conversations[0].system_message == f"{SYSTEM_TEXT}\n\nDataset system."

    def test_authored_system_message_untouched_without_the_flag(
        self, multi_turn_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        run = make_run(
            CLIConfig(model_names=["test-model"], input_file=str(multi_turn_file))
        )
        conversations = CustomDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert conversations[0].system_message == "Dataset system."

    def test_assigns_when_dataset_has_no_system_message(
        self, prompt_file: Path, tmp_path: Path, mock_tokenizer: Tokenizer
    ) -> None:
        data = tmp_path / "plain.jsonl"
        data.write_text('{"text": "hello"}\n')
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(data),
                system_prompt_file=str(prompt_file),
            )
        )
        conversations = CustomDatasetComposer(
            run=run, tokenizer=mock_tokenizer
        ).create_dataset()

        assert conversations[0].system_message == SYSTEM_TEXT

    def test_injected_without_a_tokenizer(
        self, prompt_file: Path, tmp_path: Path
    ) -> None:
        """Verbatim text needs no tokenizer, unlike the synthetic prefix path."""
        data = tmp_path / "plain.jsonl"
        data.write_text('{"text": "hello"}\n')
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(data),
                system_prompt_file=str(prompt_file),
            )
        )
        conversations = CustomDatasetComposer(run=run, tokenizer=None).create_dataset()

        assert conversations[0].system_message == SYSTEM_TEXT


class TestSpeedBenchSystemPrompt:
    """SPEED-Bench keeps its leading system message as a dispatched turn.

    ``SpeedBenchLoader._hoist_leading_system_message`` is False, so unlike
    ``multi_turn`` the dataset's system message never reaches
    ``conversation.system_message``. The composer therefore has nothing to
    prepend to, and the merge has to happen at the endpoint layer instead --
    the one production loader that exercises that path.
    """

    DATASET_SYSTEM = "SPEED-Bench system message."

    @pytest.fixture
    def speed_bench_file(self, tmp_path: Path) -> Path:
        row = {
            "question_id": "speed-coding-1".ljust(32, "0"),
            "category": "coding",
            "messages": [
                {"role": "system", "content": self.DATASET_SYSTEM},
                {"role": "user", "content": "Write a function to reverse a string."},
            ],
        }
        path = tmp_path / "speedbench.jsonl"
        path.write_bytes(orjson.dumps(row) + b"\n")
        return path

    def _compose(
        self,
        speed_bench_file: Path,
        prompt_file: Path | None,
        tokenizer: Tokenizer | None,
    ) -> list[Conversation]:
        kwargs = {}
        if prompt_file is not None:
            kwargs["system_prompt_file"] = str(prompt_file)
        run = make_run(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(speed_bench_file),
                custom_dataset_type="speed_bench_coding",
                **kwargs,
            )
        )
        return CustomDatasetComposer(run=run, tokenizer=tokenizer).create_dataset()

    @staticmethod
    def _format(conversation: Conversation, turns: list[Turn]) -> list[dict[str, Any]]:
        endpoint = ChatEndpoint(model_endpoint=create_model_endpoint(EndpointType.CHAT))
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint,
            turns=turns,
            system_message=conversation.system_message,
        )
        return endpoint.format_payload(request_info)["messages"]

    def test_composer_does_not_merge_unhoisted_system_turn(
        self, speed_bench_file: Path, prompt_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        """The dataset's system message stays a turn; only the custom text is lifted."""
        conversation = self._compose(speed_bench_file, prompt_file, mock_tokenizer)[0]

        assert conversation.system_message == SYSTEM_TEXT
        assert self.DATASET_SYSTEM not in conversation.system_message

    def test_endpoint_merges_into_one_system_message(
        self, speed_bench_file: Path, prompt_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        conversation = self._compose(speed_bench_file, prompt_file, mock_tokenizer)[0]
        messages = self._format(conversation, conversation.turns)

        systems = [m for m in messages if m["role"] == "system"]
        assert len(systems) == 1
        assert systems[0]["content"] == f"{SYSTEM_TEXT}\n\n{self.DATASET_SYSTEM}"

    def test_prefix_not_duplicated_across_repeated_formats(
        self, speed_bench_file: Path, prompt_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        """Turn state is shared across credits, so the merge must not restack.

        Regression guard for the copy-in-_format_messages: mutating the rendered
        system message in place would compound the prefix on every later request
        in the same session.
        """
        conversation = self._compose(speed_bench_file, prompt_file, mock_tokenizer)[0]

        for _ in range(3):
            messages = self._format(conversation, conversation.turns)
            systems = [m for m in messages if m["role"] == "system"]
            assert len(systems) == 1
            assert systems[0]["content"].count(SYSTEM_TEXT) == 1

    def test_dataset_system_turn_untouched_without_the_flag(
        self, speed_bench_file: Path, mock_tokenizer: Tokenizer
    ) -> None:
        conversation = self._compose(speed_bench_file, None, mock_tokenizer)[0]
        messages = self._format(conversation, conversation.turns)

        assert conversation.system_message is None
        systems = [m for m in messages if m["role"] == "system"]
        assert len(systems) == 1
        assert systems[0]["content"] == self.DATASET_SYSTEM


@pytest.mark.asyncio
class TestPublicComposerSystemPrompt:
    """Public (HF-backed) datasets reach the same injection point.

    ``PublicDatasetComposer.create_dataset_async`` calls
    ``_finalize_conversations`` like the other composers, but it is the only one
    on the async path. The loader is mocked so this stays offline.
    """

    @staticmethod
    def _public_cli(**kwargs: Any) -> CLIConfig:
        return CLIConfig(
            model_names=["test-model"],
            conversation_num_dataset_entries=2,
            public_dataset=PublicDatasetType.AIMO,
            **kwargs,
        )

    @staticmethod
    async def _compose(
        cli: CLIConfig, conversations: list[Conversation]
    ) -> list[Conversation]:
        mock_loader = AsyncMock()
        mock_loader.load_dataset = AsyncMock(return_value={"dataset": []})
        mock_loader.convert_to_conversations = AsyncMock(return_value=conversations)

        mock_loader_class = MagicMock()
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_loader_class.return_value = mock_loader

        composer = PublicDatasetComposer(run=make_run(cli), tokenizer=None)
        with (
            patch(
                "aiperf.dataset.composer.public.plugins.get_class",
                return_value=mock_loader_class,
            ),
            patch(
                "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
                return_value=MagicMock(
                    hf_dataset_name="test/dataset",
                    hf_split="train",
                    hf_subset=None,
                    prompt_column="problem",
                    multi_turn=False,
                    streaming=False,
                    is_trace=False,
                ),
            ),
        ):
            return await composer.create_dataset_async()

    @staticmethod
    def _conversations(system_message: str | None = None) -> list[Conversation]:
        return [
            Conversation(
                session_id=f"conv-{i}",
                system_message=system_message,
                turns=[Turn(texts=[Text(contents=[f"What is {i} + {i}?"])])],
            )
            for i in range(2)
        ]

    async def test_sets_system_message_on_every_conversation(
        self, prompt_file: Path
    ) -> None:
        result = await self._compose(
            self._public_cli(system_prompt_file=str(prompt_file)),
            self._conversations(),
        )

        assert len(result) == 2
        assert all(c.system_message == SYSTEM_TEXT for c in result)

    async def test_absent_without_the_flag(self) -> None:
        result = await self._compose(self._public_cli(), self._conversations())

        assert all(c.system_message is None for c in result)

    async def test_prepends_to_loader_supplied_system_message(
        self, prompt_file: Path
    ) -> None:
        """A public loader that already set system_message keeps its text."""
        result = await self._compose(
            self._public_cli(system_prompt_file=str(prompt_file)),
            self._conversations(system_message="Dataset system."),
        )

        assert all(
            c.system_message == f"{SYSTEM_TEXT}\n\nDataset system." for c in result
        )
