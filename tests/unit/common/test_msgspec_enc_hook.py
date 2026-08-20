# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.base_models import AIPerfBaseModel, _msgspec_enc_hook


class _Sample(AIPerfBaseModel):
    name: str


def test_enc_hook_encodes_extensible_str_enum():
    assert _msgspec_enc_hook(ModelSelectionStrategy.ROUND_ROBIN) == "round_robin"


def test_enc_hook_encodes_path_as_string():
    assert _msgspec_enc_hook(Path("/tmp/artifacts")) == "/tmp/artifacts"


def test_enc_hook_encodes_pydantic_model_as_dict():
    assert _msgspec_enc_hook(_Sample(name="aiperf-bench-7f2a")) == {
        "name": "aiperf-bench-7f2a"
    }


def test_enc_hook_rejects_unsupported_type():
    with pytest.raises(NotImplementedError, match="object"):
        _msgspec_enc_hook(object())
