# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path, PurePosixPath

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.base_models import AIPerfBaseModel, msgspec_enc_hook


class _Sample(AIPerfBaseModel):
    name: str


def test_enc_hook_encodes_extensible_str_enum():
    assert msgspec_enc_hook(ModelSelectionStrategy.ROUND_ROBIN) == "round_robin"


def test_enc_hook_encodes_path_as_string():
    # Deliberately not /tmp: a hardcoded world-writable temp path trips
    # bandit's S108 and this assertion needs a fixed literal to compare against.
    # PurePosixPath keeps that literal stable on Windows, where the native
    # Path renders backslash separators.
    assert msgspec_enc_hook(PurePosixPath("/var/aiperf/artifacts")) == (
        "/var/aiperf/artifacts"
    )


def test_enc_hook_encodes_native_path_as_string():
    path = Path("/var/aiperf/artifacts")
    assert msgspec_enc_hook(path) == str(path)


def test_enc_hook_encodes_pydantic_model_as_dict():
    assert msgspec_enc_hook(_Sample(name="aiperf-bench-7f2a")) == {
        "name": "aiperf-bench-7f2a"
    }


def test_enc_hook_rejects_unsupported_type():
    with pytest.raises(NotImplementedError, match="object"):
        msgspec_enc_hook(object())
