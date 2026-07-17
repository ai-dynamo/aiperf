# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, ValidationError, model_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.loader.models import MultiTurn, SingleTurn
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class SpeedBenchRow(AIPerfBaseModel):
    """Defines the schema for Speed-Bench row data.

    Each entry represents a single line in the Speed-Bench JSONL file, which contains the following fields:
    - question_id: Unique identifier for the question
    - category: Category of the question
    - messages: List of messages in the conversation
    """

    TURNS_PLACEHOLDER: ClassVar[str] = (
        "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
    )

    question_id: str = Field(
        description="Unique identifier for the question", min_length=32, max_length=32
    )
    category: str = Field(description="Category of the question", min_length=1)
    messages: list[dict[str, Any]] = Field(
        description="List of messages in the conversation", min_length=1
    )

    @model_validator(mode="after")
    def validate_messages_structure(self) -> SpeedBenchRow:
        """Validate the messages field structure."""
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
                "messages must be a non-empty list of dictionaries with role and content fields, and the content must not be the placeholder string"
            )
        return self


def is_speed_bench_row(data: object) -> bool:
    """Return whether data matches the SPEED-Bench JSONL row shape."""
    if not isinstance(data, dict):
        return False

    try:
        SpeedBenchRow.model_validate(data)
        return True
    except ValidationError:
        return False


class SpeedBenchLoader(MultiTurnDatasetLoader):
    """HuggingFace dataset loader for nvidia/SPEED-Bench.

    SPEED-Bench (SPEculative Evaluation Dataset) provides prompts for
    benchmarking speculative decoding across diverse semantic domains and
    input sequence lengths. Each JSONL row contains a ``question_id``, a
    ``category`` identifying the semantic domain or entropy tier, and a
    ``messages`` array of OpenAI-style ``role``/``content`` dictionaries.
    By default all messages are used with ``multi_turn=True``,
    otherwise only the first message is used.

    When ``category`` is set in plugin metadata, only rows matching that
    category are loaded. This enables per-category acceptance rate
    measurement by running one category at a time against a
    speculative-decoding-enabled server.

    **Qualitative subset categories** (80 samples each):
    coding, humanities, math, multilingual, qa, rag, reasoning, roleplay,
    stem, summarization, writing

    **Throughput subset categories** (512 samples each per ISL bucket):
    low_entropy, mixed, high_entropy

    Example plugins.yaml entries::

        speed_bench_qualitative:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchQualitativeLoader

        speed_bench_coding:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchQualitativeCategoryLoader
          metadata:
            category: coding

        speed_bench_throughput_1k_mixed:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchThroughput1KCategoryLoader
          metadata:
            category: mixed
    """

    def __init__(
        self,
        filename: str,
        run: BenchmarkRun | None = None,
        category: str | None = None,
        *,
        multi_turn: bool = True,
        **kwargs: Any,
    ) -> None:
        self.category = category
        self.multi_turn = multi_turn
        super().__init__(filename=filename, run=run, **kwargs)

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return whether a JSON object matches the SPEED-Bench JSONL shape."""
        return is_speed_bench_row(data)

    def load_dataset(self) -> dict[str, list[MultiTurn]]:
        """Load SPEED-Bench multi-turn data from a JSONL file.

        Each line is mapped to a ``MultiTurn`` where ``session_id`` is taken
        from the line's ``question_id``, and ``turns`` is built from the
        ``messages`` array by converting each ``{"role", "content"}`` entry
        into a ``SingleTurn(role=..., text=...)``.

        When ``self.category`` is set, lines whose ``category`` field does not
        match are skipped. If the filter eliminates every row, a warning is
        emitted to surface a likely category/file mismatch rather than
        silently returning an empty dataset.

        When ``self.multi_turn`` is set, all turns in the row are used,
        otherwise only the first turn is used.

        Returns:
            A dictionary mapping session_id (the SPEED-Bench question_id) to
            a list of MultiTurn objects.
        """
        data: dict[str, list[MultiTurn]] = defaultdict(list)

        for row in self._iter_record_dicts():
            loaded_line = SpeedBenchRow.model_validate(row)

            if self.category and loaded_line.category != self.category:
                continue

            messages = (
                loaded_line.messages if self.multi_turn else loaded_line.messages[:1]
            )

            multi_turn_data = MultiTurn(
                session_id=loaded_line.question_id,
                turns=[
                    SingleTurn(text=message["content"], role=message["role"])
                    for message in messages
                ],
            )

            data[multi_turn_data.session_id].append(multi_turn_data)

        if self.category and not data:
            self.warning(
                lambda: (
                    f"SPEED-Bench category filter {self.category!r} matched no rows "
                    f"in {self.filename}. Verify the configured category exists in "
                    f"this dataset."
                )
            )

        return data


class SpeedBenchSplitLoader(SpeedBenchLoader):
    """Base loader for a concrete SPEED-Bench JSONL split."""

    split_filename: ClassVar[str]

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        if filename is None or Path(filename).name != cls.split_filename:
            return False

        return super().can_load(data, filename)


class SpeedBenchQualitativeLoader(SpeedBenchSplitLoader):
    """Loader for the SPEED-Bench qualitative split."""

    split_filename = "qualitative.jsonl"


class SpeedBenchThroughput1KLoader(SpeedBenchSplitLoader):
    """Loader for the SPEED-Bench throughput 1K split."""

    split_filename = "throughput_1k.jsonl"


class SpeedBenchThroughput2KLoader(SpeedBenchSplitLoader):
    """Loader for the SPEED-Bench throughput 2K split."""

    split_filename = "throughput_2k.jsonl"


class SpeedBenchThroughput8KLoader(SpeedBenchSplitLoader):
    """Loader for the SPEED-Bench throughput 8K split."""

    split_filename = "throughput_8k.jsonl"


class SpeedBenchThroughput16KLoader(SpeedBenchSplitLoader):
    """Loader for the SPEED-Bench throughput 16K split."""

    split_filename = "throughput_16k.jsonl"


class SpeedBenchThroughput32KLoader(SpeedBenchSplitLoader):
    """Loader for the SPEED-Bench throughput 32K split."""

    split_filename = "throughput_32k.jsonl"
