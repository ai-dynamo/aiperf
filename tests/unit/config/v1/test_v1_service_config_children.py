# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 ServiceConfig children: validator-free DTO contract.

ZMQ + workers nested classes carry CLI annotations only; AIPerfConfig owns all
validation. This file fences the contract for the ZMQ/Workers families.
"""

import inspect

from aiperf.config.v1._workers import WorkersConfig
from aiperf.config.v1._zmq import ZMQIPCConfig, ZMQTCPConfig


def test_zmq_tcp_round_trip():
    cfg = ZMQTCPConfig.model_validate({"host": "127.0.0.1"})
    assert cfg.host == "127.0.0.1"


def test_zmq_ipc_round_trip():
    cfg = ZMQIPCConfig.model_validate({})
    assert cfg is not None


def test_workers_config_round_trip():
    cfg = WorkersConfig.model_validate({"max": 8})
    assert cfg.max == 8


def test_no_validators_on_service_children():
    for cls in (ZMQTCPConfig, ZMQIPCConfig, WorkersConfig):
        bad = [
            m
            for m in inspect.getmembers(cls)
            if hasattr(m[1], "__pydantic_decorator_info__")
        ]
        assert not bad, f"{cls.__name__} must have NO validators (found: {bad})"
