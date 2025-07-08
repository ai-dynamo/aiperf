# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod


class CreditIssuingStrategy(ABC):
    """
    Base class for credit issuing strategies.
    """

    def __init__(self, config, credit_drop_function):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.credit_drop_function = credit_drop_function

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass
