# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation gates for the verbatim ``--system-prompt`` / ``--system-prompt-file``.

The feature is the content-valued replacement for
``prefix_prompts.shared_system_length``: both fill ``Conversation.system_message``,
so it inherits that field's exclusivity rules rather than defining its own.
"""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import DatasetType
from aiperf.config.dataset import FileDataset, PublicDataset, SyntheticDataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import PublicDatasetType

SYSTEM_TEXT = "You are a production assistant."


@pytest.fixture
def prompt_file(tmp_path):
    """A readable system-prompt file.

    ``tmp_path`` resolves through /private on macOS, so it has no symlinked
    parent and survives ``safe_read_template_path``'s symlink rejection.
    """
    path = tmp_path / "system.txt"
    path.write_text(SYSTEM_TEXT)
    return path


def _cli(**kwargs) -> CLIConfig:
    return CLIConfig(
        model_names=["mock-model"],
        url="http://localhost:8000",
        request_count=1,
        **kwargs,
    )


def _benchmark(**kwargs):
    return convert_cli_to_aiperf(_cli(**kwargs)).benchmark


# ---------------------------------------------------------------------------
# Mixin-level resolution
# ---------------------------------------------------------------------------


class TestSystemPromptResolution:
    def test_inline_text_resolves(self):
        dataset = SyntheticDataset(
            name="main", type=DatasetType.SYNTHETIC, system_prompt=SYSTEM_TEXT
        )
        assert dataset.resolved_system_prompt == SYSTEM_TEXT

    def test_file_resolves(self, prompt_file):
        dataset = SyntheticDataset(
            name="main",
            type=DatasetType.SYNTHETIC,
            system_prompt_file=str(prompt_file),
        )
        assert dataset.resolved_system_prompt == SYSTEM_TEXT

    def test_unset_resolves_to_none(self):
        dataset = SyntheticDataset(name="main", type=DatasetType.SYNTHETIC)
        assert dataset.resolved_system_prompt is None

    def test_file_read_once_at_validation_time(self, prompt_file):
        """Later edits to the file do not change an already-validated config."""
        dataset = SyntheticDataset(
            name="main",
            type=DatasetType.SYNTHETIC,
            system_prompt_file=str(prompt_file),
        )
        prompt_file.write_text("mutated after validation")
        assert dataset.resolved_system_prompt == SYSTEM_TEXT

    @pytest.mark.parametrize(
        "dataset_cls,extra",
        [
            param(SyntheticDataset, {"type": DatasetType.SYNTHETIC}, id="synthetic"),
            param(
                FileDataset,
                {"type": DatasetType.FILE, "path": "/etc/hosts"},
                id="file",
            ),
            param(
                PublicDataset,
                {
                    "type": DatasetType.PUBLIC,
                    "dataset": PublicDatasetType.SHAREGPT,
                },
                id="public",
            ),
        ],
    )  # fmt: skip
    def test_every_dataset_variant_carries_the_field(self, dataset_cls, extra):
        dataset = dataset_cls(name="main", system_prompt=SYSTEM_TEXT, **extra)
        assert dataset.resolved_system_prompt == SYSTEM_TEXT


class TestSystemPromptRejections:
    def test_text_and_file_are_mutually_exclusive(self, prompt_file):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SyntheticDataset(
                name="main",
                type=DatasetType.SYNTHETIC,
                system_prompt=SYSTEM_TEXT,
                system_prompt_file=str(prompt_file),
            )

    @pytest.mark.parametrize(
        "text",
        [
            param("", id="empty"),
            param("   ", id="spaces"),
            param("\n\t ", id="whitespace"),
        ],
    )  # fmt: skip
    def test_blank_text_rejected(self, text):
        with pytest.raises(ValidationError, match="empty or whitespace-only"):
            SyntheticDataset(
                name="main", type=DatasetType.SYNTHETIC, system_prompt=text
            )

    def test_blank_file_rejected(self, tmp_path):
        blank = tmp_path / "blank.txt"
        blank.write_text("   \n")
        with pytest.raises(ValidationError, match="empty or whitespace-only"):
            SyntheticDataset(
                name="main",
                type=DatasetType.SYNTHETIC,
                system_prompt_file=str(blank),
            )

    def test_missing_file_rejected_at_startup(self, tmp_path):
        with pytest.raises(ValidationError, match="could not be read"):
            SyntheticDataset(
                name="main",
                type=DatasetType.SYNTHETIC,
                system_prompt_file=str(tmp_path / "does_not_exist.txt"),
            )


# ---------------------------------------------------------------------------
# BenchmarkConfig-level compatibility gates
# ---------------------------------------------------------------------------


class TestSystemPromptCompatibility:
    def test_resolves_through_cli_for_synthetic(self, prompt_file):
        benchmark = _benchmark(
            prompt_input_tokens_mean=100, system_prompt_file=str(prompt_file)
        )
        assert benchmark.get_system_prompt() == SYSTEM_TEXT

    def test_resolves_through_cli_for_file_dataset(self, prompt_file, tmp_path):
        data = tmp_path / "data.jsonl"
        data.write_text('{"text": "hello"}\n')
        benchmark = _benchmark(
            input_file=str(data), system_prompt_file=str(prompt_file)
        )
        assert benchmark.get_system_prompt() == SYSTEM_TEXT

    def test_resolves_through_cli_for_public_dataset(self, prompt_file):
        benchmark = _benchmark(
            public_dataset="sharegpt", system_prompt_file=str(prompt_file)
        )
        assert benchmark.get_system_prompt() == SYSTEM_TEXT

    def test_conflicts_with_shared_system_prompt_length(self):
        with pytest.raises(ValidationError, match="shared-system-prompt-length"):
            _benchmark(
                prompt_input_tokens_mean=100,
                system_prompt=SYSTEM_TEXT,
                prompt_prefix_shared_system_length=200,
            )

    def test_conflicts_with_prefix_pool(self):
        """Inherited from shared_system_length, which fills the same slot."""
        with pytest.raises(ValidationError, match="num-prefix-prompts"):
            _benchmark(
                prompt_input_tokens_mean=100,
                system_prompt=SYSTEM_TEXT,
                prompt_prefix_pool_size=50,
                prompt_prefix_length=100,
            )

    def test_allowed_with_user_context_length(self):
        """The two-tier shape, with verbatim text for the shared tier."""
        benchmark = _benchmark(
            prompt_input_tokens_mean=100,
            system_prompt=SYSTEM_TEXT,
            prompt_prefix_user_context_length=64,
        )
        assert benchmark.get_system_prompt() == SYSTEM_TEXT

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            param("completions", id="completions"),
            param("embeddings", id="embeddings"),
            param("nim_rankings", id="rankings"),
        ],
    )  # fmt: skip
    def test_rejected_on_endpoints_without_a_system_role(self, endpoint_type):
        with pytest.raises(ValidationError, match="not supported by endpoint"):
            _benchmark(
                prompt_input_tokens_mean=100,
                system_prompt=SYSTEM_TEXT,
                endpoint_type=endpoint_type,
            )

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            param("chat", id="chat"),
            param("responses", id="responses"),
            param("messages", id="messages"),
            param("chat_embeddings", id="chat_embeddings"),
        ],
    )  # fmt: skip
    def test_allowed_on_endpoints_that_consume_system_message(self, endpoint_type):
        benchmark = _benchmark(
            prompt_input_tokens_mean=100,
            system_prompt=SYSTEM_TEXT,
            endpoint_type=endpoint_type,
        )
        assert benchmark.get_system_prompt() == SYSTEM_TEXT

    def test_satisfies_warmup_isolation_system_requirement(self):
        """A verbatim prompt is a real system message, so the marker has a slot."""
        benchmark = _benchmark(
            prompt_input_tokens_mean=100,
            system_prompt=SYSTEM_TEXT,
            cache_bust="warmup_isolation_system",
        )
        assert benchmark.get_system_prompt() == SYSTEM_TEXT

    def test_warmup_isolation_system_still_rejected_without_any_system_prompt(self):
        with pytest.raises(ValidationError, match="requires a shared system prompt"):
            _benchmark(
                prompt_input_tokens_mean=100,
                cache_bust="warmup_isolation_system",
            )
