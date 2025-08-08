# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections import defaultdict

from aiperf.common.config.user_config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.generator import PromptGenerator
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.protocol import CustomDatasetLoaderProtocol


@implements_protocol(CustomDatasetLoaderProtocol)
@CustomDatasetFactory.register(CustomDatasetType.MOONCAKE_TRACE)
class MooncakeTraceDatasetLoader(AIPerfLoggerMixin):
    """A dataset loader that loads Mooncake trace data from a file.

    Loads Mooncake trace data from a file and converts the data into
    a list of conversations for dataset manager.

    Each line in the file represents a single trace entry and will be
    converted to a separate conversation with a unique session ID.

    Example:
    Fixed schedule version (Each line is a distinct session. Multi-turn is NOT supported)
    ```json
    {"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
    ```
    """

    def __init__(
        self,
        filename: str,
        prompt_generator: PromptGenerator,
        user_config: UserConfig,
        **kwargs,
    ):
        self.filename = filename
        self.prompt_generator = prompt_generator
        self.user_config = user_config
        self._skipped_traces = 0
        self._start_offset = user_config.input.fixed_schedule_start_offset
        self._end_offset = user_config.input.fixed_schedule_end_offset
        super().__init__(user_config=user_config, **kwargs)

    def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        """Load Mooncake trace data from a file.

        Returns:
            A dictionary of session_id and list of Mooncake trace data.
        """
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                trace_data = MooncakeTrace.model_validate_json(line)

                if (
                    self._start_offset is not None
                    and trace_data.timestamp < self._start_offset
                ) or (
                    self._end_offset is not None
                    and trace_data.timestamp > self._end_offset
                ):
                    self._skipped_traces += 1
                    continue  # Skip traces before or after the fixed schedule offset

                session_id = str(uuid.uuid4())
                data[session_id].append(trace_data)

        if self._skipped_traces > 0:
            self.info(
                f"Skipped {self._skipped_traces:,} traces because they were before the start offset of {self._start_offset} or after the end offset of {self._end_offset}"
            )
        self.debug(lambda: f"Loaded {len(data):,} traces from {self.filename}")

        return data

    def convert_to_conversations(
        self, data: dict[str, list[MooncakeTrace]]
    ) -> list[Conversation]:
        """Convert all the Mooncake trace data to conversation objects.

        Args:
            data: A dictionary of session_id and list of Mooncake trace data.

        Returns:
            A list of conversations.
        """
        conversations = []
        for session_id, traces in data.items():
            conversation = Conversation(session_id=session_id)
            for trace in traces:
                prompt = self.prompt_generator.generate(
                    mean=trace.input_length,
                    stddev=0,
                    hash_ids=trace.hash_ids,
                )
                turn = Turn(
                    timestamp=trace.timestamp,
                    texts=[Text(name="text", contents=[prompt])],
                )
                conversation.turns.append(turn)
            conversations.append(conversation)
        return conversations
