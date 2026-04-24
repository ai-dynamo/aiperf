# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiperf.common.enums import ConversationContextMode
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.dataset.loader.models import CustomDatasetT

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class BaseLoader(AIPerfLoggerMixin, ABC):
    """Base class for loading data.

    This abstract class provides a base implementation for loading data.
    Subclasses must implement the load_dataset and convert_to_conversations methods.
    It includes a session ID generator that is used to generate unique session IDs
    for each conversation.

    Args:
        run: The benchmark run context.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(self, *, run: BenchmarkRun, **kwargs):
        self.run = run
        super().__init__(run=run, **kwargs)
        # Create session ID generator (deterministic when seed is set)
        dataset_config = run.cfg.get_default_dataset()
        self.session_id_generator = SessionIDGenerator(
            seed=dataset_config.random_seed or run.cfg.random_seed
        )

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        """Dataset-level default context mode for conversations without an explicit one.

        Override in subclasses when the dataset format implies a specific mode.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

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
        config: The AIPerf configuration.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(self, *, filename: str, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        self.filename = filename
