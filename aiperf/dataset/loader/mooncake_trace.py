# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections import defaultdict

from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.generator import PromptGenerator
from aiperf.dataset.loader.models import MooncakeTrace


@CustomDatasetFactory.register(CustomDatasetType.MOONCAKE_TRACE)
class MooncakeTraceDatasetLoader:
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

    def __init__(self, filename: str, prompt_generator: PromptGenerator):
        self.filename = filename
        self.prompt_generator = prompt_generator

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
                session_id = str(uuid.uuid4())
                data[session_id].append(trace_data)

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
