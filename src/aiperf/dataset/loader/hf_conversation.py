# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class HFConversationDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for conversation-array datasets.

    Handles datasets where each row stores messages as a list of dicts.
    By default extracts only the first user message as the prompt (single-turn).
    With ``multi_turn=True``, builds one Conversation per row with multiple
    turns from consecutive user→assistant (or human→gpt) pairs—aligned with
    :class:`~aiperf.dataset.loader.sharegpt.ShareGPTLoader` semantics for HF
    conversation columns.

    Optionally attaches an image per row when image_column is configured (same
    image on every turn in multi-turn mode, matching single-turn behavior).

    Normalises two common dataset quirks automatically:
    - List-of-lists wrapping (VisionArena): each turn is wrapped in its own list
    - Image placeholder tokens (LLaVA): ``<image>`` tokens are stripped from text

    Example plugins.yaml entry::

        vision_arena:
          class: aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader
          metadata:
            hf_dataset_name: lmarena-ai/VisionArena-Chat
            hf_split: train
            conversation_column: conversation
            message_content_key: content
            image_column: images

        llava_onevision:
          class: aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader
          metadata:
            hf_dataset_name: lmms-lab/LLaVA-OneVision-Data
            hf_split: train
            hf_subset: sharegpt4o
            conversation_column: conversations
            message_content_key: value
            image_column: image

        my_multi_turn_hf:
          class: aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader
          metadata:
            hf_dataset_name: org/chat-dataset
            hf_split: train
            conversation_column: messages
            message_content_key: content
            multi_turn: true
    """

    def __init__(
        self,
        run: BenchmarkRun | None = None,
        *,
        conversation_column: str,
        message_content_key: str = "content",
        image_column: str | None = None,
        multi_turn: bool = False,
        tokenizer: Tokenizer | None = None,
        **kwargs,
    ) -> None:
        self.conversation_column = conversation_column
        self.message_content_key = message_content_key
        self.image_column = image_column
        self.multi_turn = multi_turn
        self.tokenizer = tokenizer
        from aiperf.dataset.loader.base_loader import _default_test_run

        active_run = run or _default_test_run()
        dataset = active_run.cfg.get_default_dataset()
        prompts = getattr(dataset, "prompts", None)
        osl = getattr(prompts, "osl", None) if prompts else None
        if isinstance(osl, dict):
            self.output_tokens_mean = osl.get("mean")
        else:
            self.output_tokens_mean = getattr(osl, "mean", None) if osl else None
        super().__init__(run=run, **kwargs)

    def _text_from_message_dict(self, message: dict[str, Any]) -> str | None:
        value = message.get(self.message_content_key)
        if not isinstance(value, str):
            return None
        return value.replace("<image>", "").strip() or None

    def _normalize_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Flatten list-of-lists (VisionArena) to a list of message dicts."""
        normalized: list[dict[str, Any]] = []
        for item in messages:
            msg = item[0] if isinstance(item, list) and item else item
            if isinstance(msg, dict) and msg:
                normalized.append(msg)
        return normalized

    def _extract_first_message(self, messages: list[Any]) -> str | None:
        """Extract the text of the first user message, handling dataset-specific quirks.

        Prefers the first message with a ``user`` or ``human`` role. Falls back
        to the literal first message when no role fields are present (backward
        compatible with untagged datasets). Unwraps list-of-lists turns
        (VisionArena) and strips ``<image>`` placeholder tokens (LLaVA).
        """
        normalized = self._normalize_messages(messages)
        if not normalized:
            return None
        for msg in normalized:
            role = (msg.get("role") or msg.get("from") or "").lower()
            if role in ("user", "human"):
                return self._text_from_message_dict(msg)
        return self._text_from_message_dict(normalized[0])

    def _sharegpt_style_pairs(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[str, str]] | None:
        """Return human→gpt pairs if messages use ShareGPT ``from`` roles."""
        uses_roles = any(c.get("from") in ("human", "gpt") for c in messages)
        if not uses_roles:
            return None
        role_msgs = [c for c in messages if c.get("from") in ("human", "gpt")]
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(role_msgs) - 1:
            if (
                role_msgs[i].get("from") == "human"
                and role_msgs[i + 1].get("from") == "gpt"
            ):
                pa = self._text_from_message_dict(role_msgs[i])
                pb = self._text_from_message_dict(role_msgs[i + 1])
                if pa and pb:
                    pairs.append((pa, pb))
                i += 2
            else:
                i += 1
        return pairs

    def _openai_style_pairs(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[str, str]]:
        """Pair consecutive user→assistant messages (OpenAI-style roles)."""
        chat_messages = [
            msg for msg in messages if (msg.get("role") or "").lower() != "system"
        ]
        pairs: list[tuple[str, str]] = []
        idx = 0
        while idx < len(chat_messages) - 1:
            current, next_msg = chat_messages[idx], chat_messages[idx + 1]
            current_role = (current.get("role") or "").lower()
            next_role = (next_msg.get("role") or "").lower()
            if current_role == "user" and next_role == "assistant":
                prompt = self._text_from_message_dict(current)
                completion = self._text_from_message_dict(next_msg)
                if prompt and completion:
                    pairs.append((prompt, completion))
                idx += 2
            else:
                idx += 1
        return pairs

    def _prompt_completion_pairs(self, messages: list[Any]) -> list[tuple[str, str]]:
        """Match ShareGPTLoader pairing: ShareGPT ``from`` roles, else OpenAI roles."""
        normalized = self._normalize_messages(messages)
        if len(normalized) < 2:
            return []

        sg = self._sharegpt_style_pairs(normalized)
        if sg:
            return sg
        return self._openai_style_pairs(normalized)

    def _pairs_pass_validation(self, pairs: list[tuple[str, str]]) -> list[int] | None:
        """Validate pairs and return completion token lengths, or None if invalid.

        Returns ``None`` when any pair fails validation. When no tokenizer is
        configured, validation is skipped and an empty list is returned so the
        caller falls back to ``max_tokens=None``.
        """
        if self.tokenizer is None:
            return []
        completion_lengths: list[int] = []
        for prompt, completion in pairs:
            prompt_length = len(self.tokenizer.encode(prompt))
            completion_length = len(self.tokenizer.encode(completion))
            if not self.is_valid_sequence(
                prompt_len=prompt_length,
                output_len=completion_length,
                skip_min_output_len_check=self.output_tokens_mean is not None,
            ):
                return None
            completion_lengths.append(completion_length)
        return completion_lengths

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each dataset row into one Conversation (single- or multi-turn)."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0
        max_conversations = self._max_conversations()
        if self.multi_turn and self.tokenizer is None:
            self.warning(
                "multi_turn=True but no tokenizer is configured; sequence-length "
                "validation will be skipped and Turn.max_tokens will be None for "
                "every turn. Configure a tokenizer to enable prompt/output-length "
                "filtering and max_tokens propagation."
            )

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            messages = row.get(self.conversation_column) or []

            images = (
                self._extract_images(row, self.image_column)
                if self.image_column
                else []
            )

            if self.multi_turn:
                pairs = self._prompt_completion_pairs(messages)
                if not pairs:
                    skipped += 1
                    continue
                completion_lengths = self._pairs_pass_validation(pairs)
                if completion_lengths is None:
                    skipped += 1
                    continue
                turns = [
                    Turn(
                        texts=[Text(contents=[prompt])],
                        images=images if idx == 0 else [],
                        max_tokens=completion_lengths[idx]
                        if completion_lengths
                        else None,
                    )
                    for idx, (prompt, _) in enumerate(pairs)
                ]
                conversations.append(
                    Conversation(
                        session_id=self.session_id_generator.next(),
                        turns=turns,
                    )
                )
                continue

            prompt = self._extract_first_message(messages)
            if not prompt:
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[prompt])],
                            images=images,
                        )
                    ],
                )
            )

        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations
