# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests verifying UserConfig + ServiceConfig accept nested data after model_rebuild."""

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.config.v1._loadgen import LoadGeneratorConfig
from aiperf.config.v1._zmq import ZMQTCPConfig


def test_user_config_accepts_endpoint_data():
    uc = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"]},
        }
    )
    assert isinstance(uc.endpoint, EndpointConfig)
    assert uc.endpoint.model_names == ["m"]


def test_user_config_accepts_loadgen_data():
    uc = UserConfig.model_validate(
        {
            "loadgen": {"concurrency": 100},
        }
    )
    assert isinstance(uc.loadgen, LoadGeneratorConfig)
    assert uc.loadgen.concurrency == 100


def test_service_config_accepts_zmq_tcp():
    sc = ServiceConfig.model_validate(
        {
            "zmq_tcp": {"host": "127.0.0.1"},
        }
    )
    assert isinstance(sc.zmq_tcp, ZMQTCPConfig)
    assert sc.zmq_tcp.host == "127.0.0.1"


def test_user_config_full_round_trip():
    uc = UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["llama"]},
            "loadgen": {"concurrency": 10, "request_count": 100},
            "tokenizer": {"name": "gpt2"},
        }
    )
    assert uc.endpoint.model_names == ["llama"]
    assert uc.loadgen.concurrency == 10
    assert uc.tokenizer.name == "gpt2"
