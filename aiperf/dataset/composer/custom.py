# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import InputConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ComposerType, CustomDatasetType
from aiperf.common.factories import ComposerFactory, CustomDatasetFactory
from aiperf.common.models import Conversation
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset import utils
from aiperf.dataset.composer.base import BaseDatasetComposer


@implements_protocol(ServiceProtocol)
@ComposerFactory.register(ComposerType.CUSTOM)
class CustomDatasetComposer(BaseDatasetComposer):
    def __init__(self, config: InputConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

    def create_dataset(self) -> list[Conversation]:
        """Create conversations from a file or directory.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        # TODO: (future) for K8s, we need to transfer file data from SC (across node)
        utils.check_file_exists(self.config.file)

        self._create_loader_instance(self.config.custom_dataset_type)
        dataset = self.loader.load_dataset()
        conversations = self.loader.convert_to_conversations(dataset)
        return conversations

    def _create_loader_instance(self, dataset_type: CustomDatasetType) -> None:
        """Initializes the dataset loader based on the custom dataset type.

        Args:
            dataset_type: The type of custom dataset to create.
        """
        kwargs = {"filename": self.config.file}
        if dataset_type == CustomDatasetType.MOONCAKE_TRACE:
            kwargs["prompt_generator"] = self.prompt_generator

        self.loader = CustomDatasetFactory.create_instance(dataset_type, **kwargs)
