# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, ValidationError, model_validator

from aiperf.common.models import AIPerfBaseModel, Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class SpeedBenchRow(AIPerfBaseModel):
    """Schema for a single SPEED-Bench JSONL row.

    SPEED-Bench distributes prompts as JSONL where each line carries a
    ``question_id`` (a 32-character identifier), a ``category`` naming the
    semantic domain or entropy tier, and a ``messages`` array of OpenAI-style
    ``role``/``content`` dictionaries. Auxiliary fields (``sub_category``,
    ``source``, ``src_id``, ``difficulty``, ``multiturn``) are tolerated but
    unused.

    The public ``nvidia/SPEED-Bench`` dataset ships placeholder content; the
    real prompts must be fetched from the source with ``specdec_bench``.
    ``validate_messages_structure`` rejects the placeholder sentinel so a
    benchmark against un-fetched data fails loudly instead of silently
    measuring against sentinel text.
    """

    TURNS_PLACEHOLDER: ClassVar[str] = (
        "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
    )

    question_id: str = Field(
        description="Unique identifier for the question", min_length=32, max_length=32
    )
    category: str = Field(description="Category of the question", min_length=1)
    messages: list[dict[str, Any]] = Field(
        description="List of OpenAI-style role/content messages in the conversation",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_messages_structure(self) -> SpeedBenchRow:
        """Require every message to be a dict with non-empty string role/content.

        Rejects the ``TURNS_PLACEHOLDER`` sentinel so un-fetched public rows
        cannot pass validation.
        """
        if not all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and bool(message["role"].strip())
            and isinstance(message.get("content"), str)
            and bool(message["content"].strip())
            and message["content"] != self.TURNS_PLACEHOLDER
            for message in self.messages
        ):
            raise ValueError(
                "messages must be a non-empty list of dictionaries with role and "
                "content fields, and the content must not be the placeholder string"
            )
        return self


def is_speed_bench_row(data: object) -> bool:
    """Return whether ``data`` matches the SPEED-Bench JSONL row shape.

    Example:
        >>> is_speed_bench_row(
        ...     {
        ...         "question_id": "0123456789abcdef0123456789abcdef",
        ...         "category": "coding",
        ...         "messages": [{"role": "user", "content": "Implement quicksort."}],
        ...     }
        ... )
        True
    """
    if not isinstance(data, dict):
        return False

    try:
        SpeedBenchRow.model_validate(data)
        return True
    except ValidationError:
        return False


class SpeedBenchLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for nvidia/SPEED-Bench.

    SPEED-Bench (SPEculative Evaluation Dataset) provides prompts for
    benchmarking speculative decoding across diverse semantic domains and
    input sequence lengths. Each row is a :class:`SpeedBenchRow` with a
    ``question_id``, a ``category``, and a ``messages`` array of OpenAI-style
    ``role``/``content`` dictionaries. Per-message roles are preserved onto the
    resulting :class:`~aiperf.common.models.Turn` objects so the chat endpoint
    dispatches each message under its authored role.

    By default (``multi_turn=True``) all messages become turns; with
    ``multi_turn=False`` only the first message is used. ``multi_turn`` is wired
    through the public-dataset loader-metadata mechanism
    (``composer/public.py``), so it can be flipped per plugin entry.

    When ``category`` is set in plugin metadata, only rows matching that
    category are loaded. This enables per-category acceptance-rate measurement
    by running one category at a time against a speculative-decoding-enabled
    server. Splits are selected via ``hf_subset`` (qualitative,
    throughput_{1,2,8,16,32}k) in plugin metadata rather than by filename.

    **Qualitative subset categories** (80 samples each):
    coding, humanities, math, multilingual, qa, rag, reasoning, roleplay,
    stem, summarization, writing

    **Throughput subset categories** (512 samples each per ISL bucket):
    low_entropy, mixed, high_entropy

    Example plugins.yaml entries::

        speed_bench_qualitative:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchLoader
          metadata:
            hf_dataset_name: nvidia/SPEED-Bench
            hf_split: test
            hf_subset: qualitative

        speed_bench_coding:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchLoader
          metadata:
            hf_dataset_name: nvidia/SPEED-Bench
            hf_split: test
            hf_subset: qualitative
            category: coding
    """

    def __init__(
        self,
        run: BenchmarkRun,
        category: str | None = None,
        *,
        multi_turn: bool = True,
        **kwargs: Any,
    ) -> None:
        self.category = category
        self.multi_turn = multi_turn
        super().__init__(run=run, **kwargs)

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert validated SPEED-Bench rows into role-tagged conversations.

        Each row maps to one :class:`~aiperf.common.models.Conversation` whose
        ``session_id`` is the row's ``question_id`` and whose turns preserve the
        per-message ``role``. When ``self.category`` is set, non-matching rows
        are skipped; when ``self.multi_turn`` is False, only the first message
        is used.

        Raises:
            pydantic.ValidationError: If any row fails :class:`SpeedBenchRow`
                validation (malformed row or placeholder content). Failing loud
                is intentional: benchmarking against sentinel text is worse than
                a hard error.
        """
        dataset = data["dataset"]
        conversations: list[Conversation] = []
        max_conversations = self._max_conversations()

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            speed_bench_row = SpeedBenchRow.model_validate(row)

            if self.category and speed_bench_row.category != self.category:
                continue

            messages = (
                speed_bench_row.messages
                if self.multi_turn
                else speed_bench_row.messages[:1]
            )

            conversations.append(
                Conversation(
                    session_id=speed_bench_row.question_id,
                    turns=[
                        Turn(
                            texts=[Text(contents=[message["content"]])],
                            role=message["role"],
                        )
                        for message in messages
                    ],
                )
            )

        if self.category and not conversations:
            self.warning(
                lambda: (
                    f"SPEED-Bench category filter {self.category!r} matched no rows. "
                    f"Verify the configured category exists in this subset."
                )
            )

        return conversations
