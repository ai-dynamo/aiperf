# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod


class CreditIssuingStrategy(ABC):
    """
    Base class for credit issuing strategies.
    """

    def __init__(self, config, stop_event, comms, service_id):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stop_event = stop_event
        self.comms = comms
        self.service_id = service_id

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass
