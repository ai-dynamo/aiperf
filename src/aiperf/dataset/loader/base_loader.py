# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from aiperf.common.config.user_config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.dataset.loader.models import CustomDatasetT


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

    def __init__(self, *, filename: str, user_config: UserConfig, **kwargs):
        super().__init__(user_config=user_config, **kwargs)
        self.filename = filename
