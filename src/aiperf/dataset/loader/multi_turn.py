# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ValidationError

from aiperf.common.enums import MediaType
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import MultiTurn, SingleTurn
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy


class MultiTurnDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads multi-turn data from a file.

    The multi-turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch_size > 1)

    NOTE: If the user specifies multiple multi-turn entries with same session ID,
    the loader will group them together. If the timestamps are specified, they will
    be sorted in ascending order later in the timing manager.

    Examples:
    1. Simple version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"text": "Hello", "image": "url", "delay": 0},
            {"text": "Hi there", "delay": 1000}
        ]
    }
    ```

    2. Batched version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"texts": ["Who are you?", "Hello world"], "images": ["/path/1.png", "/path/2.png"]},
            {"texts": ["What is in the image?", "What is AI?"], "images": ["/path/3.png", "/path/4.png"]}
        ]
    }
    ```

    3. Fixed schedule version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"timestamp": 0, "text": "What is deep learning?"},
            {"timestamp": 1000, "text": "Who are you?"}
        ]
    }
    ```

    4. Time delayed version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"delay": 0, "text": "What is deep learning?"},
            {"delay": 1000, "text": "Who are you?"}
        ]
    }
    ```

    5. full-featured version (multi-batch, multi-modal, multi-fielded, session-based, etc.)
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {
                "timestamp": 1234,
                "texts": [
                    {"name": "text_field_a", "contents": ["hello", "world"]},
                    {"name": "text_field_b", "contents": ["hi there"]}
                ],
                "images": [
                    {"name": "image_field_a", "contents": ["/path/1.png", "/path/2.png"]},
                    {"name": "image_field_b", "contents": ["/path/3.png"]}
                ]
            }
        ]
    }
    ```

    6. persistent system prompt version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"role": "system", "text": "You are a helpful assistant."},
            {"text": "What is deep learning?"},
            {"text": "Explain it for a five year old.", "delay": 1000}
        ]
    }
    ```
    A leading text-only ``role: "system"`` turn is hoisted into the
    conversation-level ``system_message`` rather than dispatched as its own
    (user-less) request. The endpoint then prepends it to every turn's
    message array, so the system prompt persists across all turns and does
    not consume a turn slot or skew per-turn metrics. Hoisting only applies to
    endpoints that send ``system_message`` (chat, responses, messages,
    chat_embeddings); on others the system turn is left in place rather than
    silently dropped. A conversation
    that leads with two or more consecutive ``role: "system"`` turns is
    un-hoisted: the endpoint only merges ``system_message`` into a rendered
    leading system message during warmup, so a hoisted prompt sitting in front
    of another leading system turn would never reach the wire in profiling
    while still counting toward ISL. Both system turns are dispatched normally
    instead, matching pre-hoist behavior.
    """

    _hoist_leading_system_message: ClassVar[bool] = True
    """Hoist a leading text-only system turn into ``conversation.system_message``.

    Subclasses representing fixed benchmarks (e.g. SpeedBench) set this to
    ``False`` so their authored message structure and per-turn metrics stay
    intact rather than silently losing the leading system turn.
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        For multi-turn data, simply validate the data against the MultiTurn model.
        This will handle all of the validation logic for the different input combinations.
        """
        if data is None:
            return False

        try:
            MultiTurn.model_validate(data)
            return True
        except ValidationError:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for MultiTurn."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[MultiTurn]]:
        """Load multi-turn data from a file or inline records.

        Each record represents a complete multi-turn conversation with its own
        session_id and multiple turns.
        """
        data: dict[str, list[MultiTurn]] = defaultdict(list)
        for record_dict in self._iter_record_dicts():
            multi_turn_data = MultiTurn.model_validate(record_dict)
            session_id = multi_turn_data.session_id or self.session_id_generator.next()
            data[session_id].append(multi_turn_data)
        return data

    def convert_to_conversations(
        self, data: dict[str, list[MultiTurn]]
    ) -> list[Conversation]:
        """Convert multi-turn data to conversation objects.

        Args:
            data: A dictionary mapping session_id to list of MultiTurn objects.

        Returns:
            A list of conversations.
        """
        hoist_enabled = (
            self._hoist_leading_system_message
            and self._endpoint_consumes_system_message()
        )
        if self.run.cfg.endpoint.uuid_and_strip:
            raise NotImplementedError(
                "--uuid-and-strip is not supported with "
                "--custom-dataset-type multi_turn. Load-time dedup of "
                "repeated images is only implemented for the single_turn "
                "loader. Use --custom-dataset-type single_turn (with "
                "session_id-grouped rows) for cache-reuse benchmarks."
            )

        conversations = []
        hoisted_count = 0
        for session_id, multi_turns in data.items():
            conversation = Conversation(session_id=session_id)
            hoisted: tuple[SingleTurn, dict[str, list[Any]]] | None = None

            # Process all MultiTurn objects for this session
            for multi_turn in multi_turns:
                for single_turn in multi_turn.turns:
                    media = self.convert_to_media_objects(single_turn)
                    if hoist_enabled and self._try_hoist_system_message(
                        conversation, single_turn, media
                    ):
                        hoisted = (single_turn, media)
                        continue
                    conversation.turns.append(self._build_turn(single_turn, media))

            # A session whose only turn was the hoisted system turn would leave
            # the conversation turn-less, which the scheduler cannot dispatch
            # (it raises "turn index out of range" and the run hangs). Un-hoist:
            # restore the system turn as a normal turn so the conversation stays
            # dispatchable, matching pre-hoist behavior for this degenerate input.
            if hoisted is not None and not conversation.turns:
                conversation.system_message = None
                conversation.turns.append(self._build_turn(*hoisted))
                hoisted = None

            # A second consecutive leading system turn fails the hoist guard and
            # stays as turn 0 (role="system"). The endpoint only merges
            # system_message into a rendered leading system message during
            # warmup, so in profiling the hoisted prompt never reaches the wire
            # while ISL still counts it. Un-hoist: restore the hoisted turn as
            # turn 0 so both system turns dispatch normally, matching pre-hoist
            # behavior for this ambiguous input.
            elif hoisted is not None and conversation.turns[0].role == "system":
                conversation.system_message = None
                conversation.turns.insert(0, self._build_turn(*hoisted))
                hoisted = None

            if hoisted is not None:
                hoisted_count += 1

            conversations.append(conversation)

        if hoisted_count:
            self.info(
                f"Hoisted leading system turn into system_message for "
                f"{hoisted_count}/{len(conversations)} conversation(s) "
                f"(endpoint '{self.run.cfg.endpoint.type}')"
            )
        return conversations

    def _endpoint_consumes_system_message(self) -> bool:
        """Whether the configured endpoint sends ``system_message`` on the wire.

        Hoisting is endpoint-blind, but only system-message-aware endpoints
        (chat, responses, messages, chat_embeddings) emit
        ``conversation.system_message``. On the others it would be silently
        dropped - and on completions the leading system turn would also bypass
        the "only supports one turn" error - so hoisting is gated on this
        capability and the system turn is left in place otherwise.
        """
        return plugins.get_endpoint_metadata(
            self.run.cfg.endpoint.type
        ).consumes_system_message

    @staticmethod
    def _build_turn(single_turn: SingleTurn, media: dict[str, list[Any]]) -> Turn:
        """Build a ``Turn`` from a parsed ``SingleTurn`` and its converted media."""
        return Turn(
            texts=media[MediaType.TEXT],
            images=media[MediaType.IMAGE],
            audios=media[MediaType.AUDIO],
            videos=media[MediaType.VIDEO],
            timestamp=single_turn.timestamp,
            delay=single_turn.delay,
            role=single_turn.role,
            max_tokens=single_turn.output_length,
            extra_body=single_turn.extra,
            raw_messages=single_turn.raw_messages,
            raw_tools=single_turn.raw_tools,
        )

    @staticmethod
    def _try_hoist_system_message(
        conversation: Conversation,
        single_turn: SingleTurn,
        media: dict[str, list[Any]],
    ) -> bool:
        """Lift a leading text-only system turn into ``conversation.system_message``.

        A ``role: "system"`` turn authored as the conversation's first turn
        belongs at the conversation level: the endpoint prepends
        ``system_message`` to every turn's message array, so the prompt
        persists across all turns. Dispatching it as a turn instead would emit
        a standalone, user-less request and inflate per-turn metrics by one.

        Only the leading turn is hoisted, and only when it carries text alone -
        a system turn with image/audio/video media is unusual, so it falls
        through to normal turn handling rather than silently dropping that
        media. The same applies to dispatch-time metadata (``timestamp``,
        ``delay``, ``output_length``, ``extra``): a conversation-level system
        message has no turn to carry them, so a system turn that sets any of
        them falls through to normal handling rather than silently dropping it.
        Returns ``True`` when the turn was consumed as the system message and
        must not be appended as a turn.
        """
        if (
            single_turn.role != "system"
            or conversation.turns
            or conversation.system_message is not None
            or single_turn.timestamp is not None
            or single_turn.delay is not None
            or single_turn.output_length is not None
            or single_turn.extra is not None
        ):
            return False
        if media[MediaType.IMAGE] or media[MediaType.AUDIO] or media[MediaType.VIDEO]:
            return False
        text = "\n".join(
            content
            for text_obj in media[MediaType.TEXT]
            for content in text_obj.contents
        )
        if not text:
            return False
        conversation.system_message = text
        return True
