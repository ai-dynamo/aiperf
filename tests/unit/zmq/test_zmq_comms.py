# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for zmq_comms.py - ZMQ communication classes.
"""

import pytest

from aiperf.common.config import ZMQTCPConfig
from aiperf.common.enums import CommAddress, LifecycleState
from aiperf.common.exceptions import InvalidStateError
from aiperf.plugin.enums import CommClientType
from aiperf.zmq.zmq_comms import (
    ZMQTCPCommunication,
)


class TestZMQTCPCommunication:
    """Test ZMQTCPCommunication class."""

    def test_init_with_default_config(self):
        """Test initialization with default TCP config."""
        comm = ZMQTCPCommunication()

        assert comm.config is not None
        assert isinstance(comm.config, ZMQTCPConfig)
        assert comm.context is not None
        assert comm.state == LifecycleState.CREATED

    def test_init_with_custom_config(self):
        """Test initialization with custom TCP config."""
        config = ZMQTCPConfig()
        comm = ZMQTCPCommunication(config=config)

        assert comm.config == config
        assert isinstance(comm.config, ZMQTCPConfig)

    def test_get_address_with_comm_address_enum(self):
        """Test get_address with CommAddress enum."""
        config = ZMQTCPConfig()
        comm = ZMQTCPCommunication(config=config)

        address = comm.get_address(CommAddress.EVENT_BUS_PROXY_FRONTEND)
        assert address is not None
        assert "tcp://" in address or "ipc://" in address

    def test_get_address_with_string(self):
        """Test get_address with string address."""
        comm = ZMQTCPCommunication()

        custom_address = "tcp://192.168.1.1:8888"
        address = comm.get_address(custom_address)
        assert address == custom_address

    @pytest.mark.parametrize(
        "client_type,bind",
        [
            (CommClientType.PUB, True),
            (CommClientType.SUB, False),
            (CommClientType.PUSH, True),
            (CommClientType.PULL, False),
            (CommClientType.REQUEST, False),
            (CommClientType.REPLY, True),
        ],
    )  # fmt: skip
    def test_create_client_returns_correct_type(self, client_type, bind):
        """Test that create_client returns the correct client type."""
        comm = ZMQTCPCommunication()
        address = "tcp://127.0.0.1:5555"

        client = comm.create_client(client_type, address, bind=bind)

        assert client is not None
        assert client.bind == bind

    def test_create_client_caches_clients(self):
        """Test that create_client caches clients."""
        comm = ZMQTCPCommunication()
        address = "tcp://127.0.0.1:5555"

        client1 = comm.create_client(CommClientType.PUB, address, bind=True)
        client2 = comm.create_client(CommClientType.PUB, address, bind=True)

        assert client1 is client2  # Same instance

    def test_create_client_different_params_creates_new(self):
        """Test that different parameters create new clients."""
        comm = ZMQTCPCommunication()
        address = "tcp://127.0.0.1:5555"

        client1 = comm.create_client(CommClientType.PUB, address, bind=True)
        client2 = comm.create_client(CommClientType.PUB, address, bind=False)

        assert client1 is not client2  # Different instances

    @pytest.mark.asyncio
    async def test_create_client_after_initialize_raises_error(self):
        """Test that creating client after initialize raises InvalidStateError."""
        comm = ZMQTCPCommunication()
        await comm.initialize()

        with pytest.raises(InvalidStateError, match="must be created before"):
            comm.create_client(CommClientType.PUB, "tcp://127.0.0.1:5555", bind=True)

        await comm.stop()
