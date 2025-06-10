#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging
from collections import deque

from aiperf.services.dataset.composer.synthetic import SyntheticConversationComposer
from aiperf.services.dataset.config import CustomDataConfig
from aiperf.services.dataset.conversation import Conversation


class DatasetHandler:
    """Sits on top of the composer layer.

    - Generates all the dataset by calling the composers
    - Manages the conversation datasets (e.g. queue)
    """

    def __init__(self, config: CustomDataConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.num_conversations = config.num_conversations

        self.queue: deque[Conversation] = deque(maxlen=self.num_conversations)
        self.composer = SyntheticConversationComposer(config)

    # TODO: better name?
    def initialize(self):
        self.logger.debug(
            "Initializing dataset handler with %d conversations",
            self.num_conversations,
        )
        for _ in range(self.num_conversations):
            conversation = self.composer.create_conversation()
            self.queue.append(conversation)

    def get_conversation(self) -> Conversation:
        return self.queue.pop()


if __name__ == "__main__":
    import logging

    from aiperf.common.tokenizer import Tokenizer
    from aiperf.services.dataset.config import (
        PromptConfig,
        TurnConfig,
        TurnDelayConfig,
    )

    logging.basicConfig(level=logging.DEBUG)

    tokenizer = Tokenizer.from_pretrained("gpt2")
    config = CustomDataConfig(
        num_conversations=2,  # XXX CONVERSATION
        turn=TurnConfig(  # XXX TURN
            mean=5,
            stddev=3,
            delay=TurnDelayConfig(  # XXX TURN DELAY
                mean=100,
                stddev=3,
            ),
        ),
        prompt=PromptConfig(
            tokenizer=tokenizer,
            mean=10,
            stddev=3,
            # prefix_prompt=PrefixPromptConfig(  # XXX PREFIX PROMPT
            #    pool_size=4,
            #    length=10,
            # ),
            block_size=512,
        ),
        # image=ImageConfig(
        #    width_mean=2,
        #    height_mean=2,
        # ),
    )
    handler = DatasetHandler(config)
    handler.initialize()

    from rich import print

    for _ in range(config.num_conversations):
        print(handler.get_conversation())
