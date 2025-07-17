# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections import defaultdict

from aiperf.common.dataset_models import Conversation, Text, Turn
from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.services.dataset.generator import PromptGenerator
from aiperf.services.dataset.loader.models import CustomData, TraceCustomData


@CustomDatasetFactory.register(CustomDatasetType.TRACE)
class TraceDatasetLoader:
    """A dataset loader that loads trace data from a file.

    Loads trace data (e.g. Mooncake trace) from a file
    and converts the data into a list of conversations for dataset manager.

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

    def load_dataset(self) -> dict[str, list[CustomData]]:
        """Load trace data from a file.

        Returns:
            A dictionary of session_id and list of trace data.
        """
        data: dict[str, list[TraceCustomData]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                trace_data = TraceCustomData.model_validate_json(line)
                session_id = str(uuid.uuid4())
                data[session_id].append(trace_data)

        return data

    def convert_to_conversations(
        self, data: dict[str, list[TraceCustomData]]
    ) -> list[Conversation]:
        """Convert all the trace data to conversations.

        Args:
            data: A dictionary of session_id and list of trace data.

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
                    text=[Text(name="text", content=[prompt])],
                )
                conversation.turns.append(turn)
            conversations.append(conversation)
        return conversations
