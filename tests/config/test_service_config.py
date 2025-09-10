# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import cast

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.config.zmq_config import ZMQIPCConfig, ZMQTCPConfig
from aiperf.common.enums import CommunicationBackend


@pytest.fixture
def tcp_config():
    """Fixture providing a sample TCP configuration."""
    return ZMQTCPConfig()


@pytest.fixture
def ipc_config():
    """Fixture providing a sample IPC configuration."""
    return ZMQIPCConfig()


@pytest.fixture
def custom_tcp_config():
    """Fixture providing a TCP configuration with custom ports."""
    return ZMQTCPConfig(
        host="10.0.0.1",
        records_push_pull_port=6000,
        credit_drop_port=6001,
        credit_return_port=6002,
    )


def assert_comm_config_type_and_backend(config, expected_type, expected_backend):
    """Helper to assert communication configuration type and backend."""
    assert config._comm_config is not None
    assert isinstance(config._comm_config, expected_type)
    assert config._comm_config.comm_backend == expected_backend


class TestServiceConfigCommValidation:
    """Test communication configuration validation in ServiceConfig."""

    def test_default_uses_zmq_ipc_config(self):
        """Default configuration should use ZMQ IPC."""
        config = ServiceConfig()

        assert_comm_config_type_and_backend(
            config, ZMQIPCConfig, CommunicationBackend.ZMQ_IPC
        )
        comm_config = config.comm_config
        assert isinstance(comm_config, ZMQIPCConfig)
        # Make sure it is not using a hardcoded path (security measure)
        assert comm_config.path != Path("/tmp/aiperf")

        # Make sure a new config uses a different path
        config = ServiceConfig()
        assert comm_config.path != cast(ZMQIPCConfig, config.comm_config).path

    @pytest.mark.parametrize(
        "config_type,expected_type,expected_backend",
        [
            (CommunicationBackend.ZMQ_TCP, ZMQTCPConfig, CommunicationBackend.ZMQ_TCP),
            (CommunicationBackend.ZMQ_IPC, ZMQIPCConfig, CommunicationBackend.ZMQ_IPC),
        ],
    )
    def test_uses_provided_config_type(
        self, config_type, expected_type, expected_backend, tcp_config, ipc_config
    ):
        """Should use the provided configuration type."""
        if config_type == CommunicationBackend.ZMQ_TCP:
            config = ServiceConfig(zmq_tcp=tcp_config)
        else:
            config = ServiceConfig(zmq_ipc=ipc_config)

        assert_comm_config_type_and_backend(config, expected_type, expected_backend)

    def test_both_configs_raises_error(self, tcp_config, ipc_config):
        """Should raise error when both TCP and IPC configs are provided."""
        with pytest.raises(
            ValueError, match="Cannot use both ZMQ TCP and ZMQ IPC configuration"
        ):
            ServiceConfig(zmq_tcp=tcp_config, zmq_ipc=ipc_config)

    def test_comm_config_property_access(self, tcp_config):
        """Should return the configured communication config via property."""
        config = ServiceConfig(zmq_tcp=tcp_config)

        assert config.comm_config is config._comm_config
        comm_config = config.comm_config
        assert isinstance(comm_config, ZMQTCPConfig)
        assert comm_config.host == "127.0.0.1"

    def test_comm_config_property_error_when_unset(self):
        """Should raise error when accessing comm_config property if unset."""
        config = ServiceConfig()
        config._comm_config = None  # Simulate unset state

        with pytest.raises(ValueError, match="Communication configuration is not set"):
            _ = config.comm_config


class TestTCPConfiguration:
    """Test TCP-specific configuration behavior."""

    def test_custom_host_and_ports(self, custom_tcp_config):
        """Should use custom host and port settings."""
        config = ServiceConfig(zmq_tcp=custom_tcp_config)
        comm_config = config.comm_config
        assert isinstance(comm_config, ZMQTCPConfig)

        assert comm_config.host == "10.0.0.1"
        # Ensure the host gets filled in for the proxy configs
        assert comm_config.dataset_manager_proxy_config.host == "10.0.0.1"
        assert comm_config.event_bus_proxy_config.host == "10.0.0.1"
        assert comm_config.raw_inference_proxy_config.host == "10.0.0.1"
        assert comm_config.records_push_pull_port == 6000
        assert comm_config.credit_drop_port == 6001
        assert comm_config.credit_return_port == 6002

    def test_address_generation(self, custom_tcp_config):
        """Should generate correct TCP addresses."""
        config = ServiceConfig(zmq_tcp=custom_tcp_config)
        comm_config = config.comm_config

        expected_addresses = {
            "records_push_pull_address": "tcp://10.0.0.1:6000",
            "credit_drop_address": "tcp://10.0.0.1:6001",
            "credit_return_address": "tcp://10.0.0.1:6002",
        }

        for attr, expected in expected_addresses.items():
            assert getattr(comm_config, attr) == expected

    def test_default_ports(self):
        """Should use default ports when not specified."""
        config = ServiceConfig(zmq_tcp=ZMQTCPConfig())
        comm_config = config.comm_config
        assert isinstance(comm_config, ZMQTCPConfig)

        # Ensure it uses local host
        assert comm_config.host == "127.0.0.1"
        assert comm_config.dataset_manager_proxy_config.host == "127.0.0.1"
        assert comm_config.event_bus_proxy_config.host == "127.0.0.1"
        assert comm_config.raw_inference_proxy_config.host == "127.0.0.1"
        assert comm_config.records_push_pull_port == 5557
        assert comm_config.credit_drop_port == 5562
        assert comm_config.credit_return_port == 5563


class TestIPCConfiguration:
    """Test IPC-specific configuration behavior."""

    @pytest.mark.parametrize(
        "path,expected_addresses",
        [
            (
                Path("/tmp/aiperf"),  # Custom specified path
                {
                    "records_push_pull_address": "ipc:///tmp/aiperf/records_push_pull.ipc",
                    "credit_drop_address": "ipc:///tmp/aiperf/credit_drop.ipc",
                    "credit_return_address": "ipc:///tmp/aiperf/credit_return.ipc",
                },
            ),
        ],
    )
    def test_path_and_address_generation(self, path, expected_addresses):
        """Should generate correct IPC addresses for given paths."""
        ipc_config = ZMQIPCConfig(path=path)
        config = ServiceConfig(zmq_ipc=ipc_config)
        comm_config = config.comm_config
        assert isinstance(comm_config, ZMQIPCConfig)

        assert comm_config.path == path
        for attr, expected in expected_addresses.items():
            assert getattr(comm_config, attr) == expected

    def test_path_sets_proxy_paths(self):
        """Proxy paths should be set to the given root path."""
        ipc_config = ZMQIPCConfig()
        config = ServiceConfig(zmq_ipc=ipc_config)
        comm_config = config.comm_config
        assert isinstance(comm_config, ZMQIPCConfig)
        assert comm_config.dataset_manager_proxy_config.path == comm_config.path
        assert comm_config.event_bus_proxy_config.path == comm_config.path
        assert comm_config.raw_inference_proxy_config.path == comm_config.path


class TestServiceConfigSerialization:
    """Test ServiceConfig serialization."""

    def test_serialization_includes_comm_config(self, tcp_config, ipc_config):
        """Should include communication configuration in serialization."""
        config = ServiceConfig(zmq_tcp=tcp_config)
        config_dict = config.model_dump()

        assert config_dict["zmq_tcp"]["host"] == "127.0.0.1"
        assert config_dict["zmq_ipc"] is None

        config = ServiceConfig(zmq_ipc=ipc_config)
        config_dict = config.model_dump()

        assert config_dict["zmq_tcp"] is None
        assert str(config_dict["zmq_ipc"]["path"]).startswith(tempfile.gettempdir())
