#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for all data generators.

    Provides a consistent interface for generating synthetic data while allowing
    each generator type to use its own specific configuration and runtime parameters.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_initialized: asyncio.Event = asyncio.Event()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the generator."""

    @abstractmethod
    async def generate(self, *args, **kwargs) -> str:
        """Generate synthetic data.

        Args:
            *args: Variable length argument list (subclass-specific)
            **kwargs: Arbitrary keyword arguments (subclass-specific)

        Returns:
            Generated data as a string (could be text, base64 encoded media, etc.)
        """

    async def wait_for_data_initialized(self) -> None:
        """Wait for the data to be initialized."""
        await self.data_initialized.wait()
