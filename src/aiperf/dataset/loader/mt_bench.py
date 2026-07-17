# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class MTBenchDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace loader for MT-Bench prompts (HuggingFaceH4/mt_bench_prompts).

    Each row's prompt column is a list of strings (one per user turn, usually
    two), so each row becomes one multi-turn Conversation of bare user Turns.
    AIPerf's UserSession dispatches the turns sequentially and, under the default
    DELTAS_WITHOUT_RESPONSES context mode, feeds the live assistant reply back as
    history between turns - the FastChat / Spec-Bench MT-Bench protocol.

    Example plugins.yaml entry::

        spec_al_mtbench:
          class: aiperf.dataset.loader.mt_bench:MTBenchDatasetLoader
          metadata:
            hf_dataset_name: HuggingFaceH4/mt_bench_prompts
            hf_split: train
    """

    # mt_bench_prompts stores the per-turn prompt list under this column.
    PROMPT_COLUMN = "prompt"

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each MT-Bench row into a multi-turn Conversation."""
        dataset = data["dataset"]
        conversations: list[Conversation] = []
        skipped = 0
        max_conversations = self._max_conversations()

        column_validated = False
        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            if not column_validated:
                column_validated = True
                if self.PROMPT_COLUMN not in row:
                    raise DatasetLoaderError(
                        f"Column '{self.PROMPT_COLUMN}' not found in dataset "
                        f"'{self.hf_dataset_name}'. Available columns: "
                        f"{list(row.keys())}."
                    )

            turns_raw = row.get(self.PROMPT_COLUMN)
            if not isinstance(turns_raw, list):
                skipped += 1
                continue

            conv_turns: list[Turn] = []
            for t in turns_raw:
                text = str(t).strip() if t else ""
                if text:
                    conv_turns.append(Turn(texts=[Text(contents=[text])]))
            if not conv_turns:
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=conv_turns,
                )
            )

        if skipped > 0 and not conversations:
            self.warning(
                f"All {skipped} rows skipped - no conversations loaded. "
                f"Check that '{self.PROMPT_COLUMN}' holds non-empty prompt lists."
            )
        self.debug(
            lambda: f"Converted {len(conversations)} MT-Bench rows (skipped {skipped})"
        )
        return conversations
