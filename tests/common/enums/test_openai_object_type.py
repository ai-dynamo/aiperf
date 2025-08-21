# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import OpenAIObjectType


@pytest.mark.parametrize("obj_type", list(OpenAIObjectType))
def test_openai_object_type(obj_type: str | None) -> None:
    assert obj_type is not None
    assert obj_type in OpenAIObjectType
