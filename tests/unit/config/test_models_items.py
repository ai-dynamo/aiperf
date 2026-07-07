# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field-constraint tests for ModelItem (restored parity with upstream main)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config._models_core import ModelItem


class TestModelItemName:
    def test_model_item_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ModelItem(name="")

    def test_model_item_nonempty_name_passes(self) -> None:
        item = ModelItem(name="meta-llama/Llama-3.1-8B-Instruct")
        assert item.name == "meta-llama/Llama-3.1-8B-Instruct"
