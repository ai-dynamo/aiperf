# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from aiperf.common.enums import ComposerType
from aiperf.common.factories import FactoryMixin
from aiperf.services.dataset.composer.base import BaseDatasetComposer

logger = logging.getLogger(__name__)


class ComposerFactory(FactoryMixin[ComposerType, BaseDatasetComposer]):
    """Factory for registering and creating BaseDatasetComposer instances
    based on the specified composer type.

    Example:
    ```python
        # Register a new composer type
        @ComposerFactory.register(ComposerType.SYNTHETIC)
        class SyntheticDatasetComposer(BaseDatasetComposer):
            pass

        # Create a new composer instance
        composer = ComposerFactory.create_instance(
            ComposerType.SYNTHETIC,
            config=InputConfig(
                conversation=ConversationConfig(num=10),
                prompt=PromptConfig(batch_size=10),
            )
        )
    ```
    """
