# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for zmq_comms.py - ZMQ communication classes.
"""

import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import ZMQIPCConfig, ZMQTCPConfig
from aiperf.common.enums import CommAddress, CommClientType, LifecycleState
from aiperf.common.exceptions import InvalidStateError
from aiperf.zmq.zmq_comms import ZMQIPCCommunication, ZMQTCPCommunication


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


class TestZMQIPCCommunication:
    """Test ZMQIPCCommunication class."""

    def test_init_with_default_config(self):
        """Test initialization with default IPC config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ZMQIPCConfig(socket_dir=tmpdir)
            comm = ZMQIPCCommunication(config=config)

            assert comm.config is not None
            assert isinstance(comm.config, ZMQIPCConfig)
            assert comm.context is not None
            assert Path(tmpdir).exists()

    def test_init_creates_ipc_directory(self):
        """Test that initialization creates IPC directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipc_dir = Path(tmpdir) / "zmq_sockets"

            # Directory doesn't exist yet
            assert not ipc_dir.exists()

            # Config validator creates the directory
            config = ZMQIPCConfig(path=ipc_dir)

            # Directory should now exist (created by config validator)
            assert ipc_dir.exists()

            comm = ZMQIPCCommunication(config=config)

            assert ipc_dir.exists()
            assert comm._ipc_socket_dir == ipc_dir

    def test_init_with_existing_directory(self):
        """Test initialization when IPC directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipc_dir = Path(tmpdir) / "existing_sockets"
            ipc_dir.mkdir()

            config = ZMQIPCConfig(path=ipc_dir)
            comm = ZMQIPCCommunication(config=config)

            assert comm._ipc_socket_dir == ipc_dir

    @pytest.mark.asyncio
    async def test_cleanup_removes_ipc_files(self):
        """Test that cleanup removes .ipc socket files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipc_dir = Path(tmpdir) / "sockets"
            config = ZMQIPCConfig(path=ipc_dir)
            comm = ZMQIPCCommunication(config=config)

            # Create some .ipc files
            (ipc_dir / "socket1.ipc").touch()
            (ipc_dir / "socket2.ipc").touch()
            (ipc_dir / "other.txt").touch()

            assert (ipc_dir / "socket1.ipc").exists()
            assert (ipc_dir / "socket2.ipc").exists()

            # Cleanup
            comm._cleanup_ipc_sockets()

            # .ipc files should be removed
            assert not (ipc_dir / "socket1.ipc").exists()
            assert not (ipc_dir / "socket2.ipc").exists()
            # Other files should remain
            assert (ipc_dir / "other.txt").exists()

    def test_get_address_formats_correctly(self):
        """Test that IPC addresses are formatted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ZMQIPCConfig(path=tmpdir)
            comm = ZMQIPCCommunication(config=config)

            address = comm.get_address(CommAddress.EVENT_BUS_PROXY_FRONTEND)

            assert address is not None
            assert "ipc://" in address

    def test_create_client_with_socket_ops(self):
        """Test creating client with custom socket options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ZMQIPCConfig(path=tmpdir)
            comm = ZMQIPCCommunication(config=config)

            socket_ops = {1: 100}  # Custom socket option
            client = comm.create_client(
                CommClientType.PUB,
                "ipc:///tmp/test.ipc",
                bind=True,
                socket_ops=socket_ops,
            )

            assert client is not None
            assert client.socket_ops == socket_ops

    def test_create_client_with_max_pull_concurrency(self):
        """Test creating PULL client with max_pull_concurrency parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ZMQIPCConfig(path=tmpdir)
            comm = ZMQIPCCommunication(config=config)

            client = comm.create_client(
                CommClientType.PULL,
                "ipc:///tmp/test.ipc",
                bind=False,
                max_pull_concurrency=5,
            )

            assert client is not None
            # Verify the semaphore was configured
            assert client.semaphore._value == 5


class TestBaseZMQCommunicationEdgeCases:
    """Test edge cases in BaseZMQCommunication."""

    def test_context_is_singleton(self):
        """Test that ZMQ context is a singleton."""
        comm1 = ZMQTCPCommunication()
        comm2 = ZMQTCPCommunication()

        # Both should use the same context instance
        assert comm1.context is comm2.context

    def test_clients_cache_key_uniqueness(self):
        """Test that cache key correctly distinguishes clients."""
        comm = ZMQTCPCommunication()

        # Create clients with different parameters
        client1 = comm.create_client(
            CommClientType.PUB, "tcp://127.0.0.1:5555", bind=True
        )
        client2 = comm.create_client(
            CommClientType.PUB, "tcp://127.0.0.1:5555", bind=False
        )
        client3 = comm.create_client(
            CommClientType.SUB, "tcp://127.0.0.1:5555", bind=True
        )
        client4 = comm.create_client(
            CommClientType.PUB, "tcp://127.0.0.1:6666", bind=True
        )

        # All should be different instances
        assert client1 is not client2
        assert client1 is not client3
        assert client1 is not client4
        assert client2 is not client3

    @pytest.mark.asyncio
    async def test_lifecycle_propagates_to_clients(self):
        """Test that lifecycle methods propagate to child clients."""
        comm = ZMQTCPCommunication()

        # Create a client
        client = comm.create_client(
            CommClientType.PUB, "tcp://127.0.0.1:5555", bind=True
        )

        # Initialize communication (should initialize client too)
        await comm.initialize()

        assert comm.state == LifecycleState.INITIALIZED
        assert client.state == LifecycleState.INITIALIZED

        # Stop communication (should stop client too)
        await comm.stop()

        assert comm.state == LifecycleState.STOPPED
        assert client.state == LifecycleState.STOPPED
