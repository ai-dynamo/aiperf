#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.config import DatasetConfig


class TestCustomDatasetComposer:
    @pytest.fixture
    def empty_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    @pytest.fixture
    def config(self, empty_tokenizer) -> DatasetConfig:
        return DatasetConfig(
            filename=Path("dummy.jsonl"),
            tokenizer=empty_tokenizer,
        )
