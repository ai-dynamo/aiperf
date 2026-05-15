# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.dataset.loader.models import CustomDatasetT
from aiperf.plugin.enums import DatasetSamplingStrategy

LoaderProbeData = dict[str, Any]
"""First-line probe shape passed to ``can_load`` overrides.

Any of ``session_id``, ``turns``, ``messages``, ``data``, ``conversation_id``
may be present depending on the on-disk format. Loaders branch on which keys
exist to decide whether they recognise the file.
"""


class BaseLoader(AIPerfLoggerMixin, ABC):
    """Base class for loading data.

    This abstract class provides a base implementation for loading data.
    Subclasses must implement the load_dataset and convert_to_conversations methods.
    It includes a session ID generator that is used to generate unique session IDs
    for each conversation.

    Args:
        user_config: The user configuration.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(self, *, user_config: UserConfig, **kwargs):
        self.user_config = user_config
        super().__init__(user_config=user_config, **kwargs)
        # Create session ID generator (deterministic when seed is set)
        self.session_id_generator = SessionIDGenerator(
            seed=user_config.input.random_seed
        )

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        """Dataset-level default context mode for conversations without an explicit one.

        Override in subclasses when the dataset format implies a specific mode.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Dataset-level preferred sampling strategy for downstream conversation selection.

        Override in subclasses when the dataset format implies a specific strategy
        (e.g. raw payload replay loaders prefer SEQUENTIAL to preserve recorded order).
        Defaults to SHUFFLE for general datasets.
        """
        return DatasetSamplingStrategy.SHUFFLE

    @abstractmethod
    def load_dataset(self) -> dict[str, list[CustomDatasetT]]: ...

    @abstractmethod
    def convert_to_conversations(
        self, custom_data: dict[str, list[CustomDatasetT]]
    ) -> list[Conversation]: ...


class BaseFileLoader(BaseLoader):
    """Base class for loading data from a file.

    This abstract class provides a base implementation for loading data from a file.
    Subclasses must implement the load_dataset and convert_to_conversations methods.
    It includes a session ID generator that is used to generate unique session IDs
    for each conversation. It also includes a filename attribute that is used to
    load the data from a file.

    Args:
        filename: The path to the file to load.
        user_config: The user configuration.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(self, *, filename: str | Path, user_config: UserConfig, **kwargs):
        super().__init__(user_config=user_config, **kwargs)
        self.filename = Path(filename) if isinstance(filename, str) else filename


class BaseRawPayloadLoader(BaseFileLoader):
    """Base for loaders that produce verbatim raw_payload conversations.

    Provides shared defaults: MESSAGE_ARRAY_WITH_RESPONSES context mode and
    SEQUENTIAL sampling. Used by ``inputs_json`` and ``raw_payload`` loaders
    that replay pre-built API request payloads byte-for-byte.
    """

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
