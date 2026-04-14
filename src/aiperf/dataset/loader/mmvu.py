# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class MMVUDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for the MMVU video understanding benchmark.

    Combines the question with formatted multiple-choice options and attaches
    the video URL as a video turn input — matching vLLM's benchmark format.

    Example plugins.yaml entry::

        mmvu:
          class: aiperf.dataset.loader.mmvu:MMVUDatasetLoader
          metadata:
            hf_dataset_name: yale-nlp/MMVU
            hf_split: validation
            video_column: video
    """

    def __init__(
        self,
        user_config: UserConfig,
        video_column: str = "video",
        **kwargs,
    ) -> None:
        self.video_column = video_column
        super().__init__(user_config=user_config, **kwargs)

    @staticmethod
    def _format_prompt(row: dict[str, Any]) -> str:
        """Build prompt from question and choices, matching vLLM benchmark format."""
        question = (row.get("question") or "").strip()
        choices = row.get("choices", {})
        if isinstance(choices, dict):
            choices_str = " ".join(f"{k}.{v}" for k, v in choices.items() if v)
            if question and choices_str:
                return f"{question} {choices_str}"
            return question or choices_str
        return question

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each MMVU row into a single-turn video Conversation."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0
        max_conversations = self._max_conversations()

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            prompt = self._format_prompt(row)
            if not prompt:
                skipped += 1
                continue

            videos = self._extract_videos(row, self.video_column)
            if not videos:
                self.warning(
                    f"Row has no video in column '{self.video_column}' — skipping."
                )
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[prompt])],
                            videos=videos,
                        )
                    ],
                )
            )

        if skipped > 0 and not conversations:
            self.warning(f"All {skipped} rows were skipped — no conversations loaded.")
        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations
