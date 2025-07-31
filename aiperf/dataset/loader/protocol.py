# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from aiperf.common.models import Conversation
from aiperf.dataset.loader.models import CustomData


class CustomDatasetLoaderProtocol(Protocol):
    def load_dataset(self) -> dict[str, list[CustomData]]:
        pass

    def convert_to_conversations(
        self, custom_data: dict[str, list[CustomData]]
    ) -> list[Conversation]:
        pass
