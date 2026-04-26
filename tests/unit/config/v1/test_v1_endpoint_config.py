# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 EndpointConfig - validator-free CLI input DTO."""

import inspect

from aiperf.config.v1._endpoint import EndpointConfig


def test_endpoint_config_round_trip():
    cfg = EndpointConfig.model_validate(
        {
            "model_names": ["x"],
            "url": ["http://localhost:8000"],
        }
    )
    assert cfg.model_names == ["x"]


def test_endpoint_config_has_no_validators():
    bad = [
        m
        for m in inspect.getmembers(EndpointConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not bad, f"EndpointConfig must have NO validators (found: {bad})"
