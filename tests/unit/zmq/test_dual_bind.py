# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ZMQ dual-bind support (IPC + TCP).

Covers:
- ZMQDualBindProxyConfig address resolution
- ZMQDualBindConfig deployment-mode address selection
- ZMQDualBindCommunication initialization and plugin registration
- PullClient and StreamingRouterClient dual-bind @on_init binding
- BaseZMQProxy dual-bind socket binding during initialization
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.config import (
    EndpointConfig,
    ServiceConfig,
    UserConfig,
    ZMQDualBindConfig,
    ZMQDualBindProxyConfig,
    ZMQIPCProxyConfig,
    ZMQTCPConfig,
    ZMQTCPProxyConfig,
)
from aiperf.common.enums import CommAddress, LifecycleState
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.plugin import plugins
from aiperf.plugin.enums import CommunicationBackend, PluginType
from aiperf.records.records_manager import RecordsManager
from aiperf.zmq.pull_client import ZMQPullClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient
from aiperf.zmq.zmq_comms import ZMQDualBindCommunication
from aiperf.zmq.zmq_proxy_sockets import ZMQXPubXSubProxy

# =============================================================================
# ZMQDualBindProxyConfig
# =============================================================================


class TestZMQDualBindProxyConfig:
    """Test ZMQDualBindProxyConfig address construction and resolution."""

    @pytest.fixture
    def proxy_config(self, tmp_path: Path) -> ZMQDualBindProxyConfig:
        return ZMQDualBindProxyConfig(
            ipc_path=tmp_path,
            name="event_bus_proxy",
            tcp_host="0.0.0.0",
            tcp_frontend_port=5663,
            tcp_backend_port=5664,
        )

    def test_frontend_address_returns_ipc(
        self, proxy_config: ZMQDualBindProxyConfig
    ) -> None:
        addr = proxy_config.frontend_address
        assert addr.startswith("ipc://")
        assert "event_bus_proxy_frontend.ipc" in addr

    def test_backend_address_returns_ipc(
        self, proxy_config: ZMQDualBindProxyConfig
    ) -> None:
        addr = proxy_config.backend_address
        assert addr.startswith("ipc://")
        assert "event_bus_proxy_backend.ipc" in addr

    def test_frontend_tcp_address(self, proxy_config: ZMQDualBindProxyConfig) -> None:
        assert proxy_config.frontend_tcp_address == "tcp://0.0.0.0:5663"

    def test_backend_tcp_address(self, proxy_config: ZMQDualBindProxyConfig) -> None:
        assert proxy_config.backend_tcp_address == "tcp://0.0.0.0:5664"

    def test_control_address_disabled_by_default(
        self, proxy_config: ZMQDualBindProxyConfig
    ) -> None:
        assert proxy_config.control_address is None

    def test_capture_address_disabled_by_default(
        self, proxy_config: ZMQDualBindProxyConfig
    ) -> None:
        assert proxy_config.capture_address is None

    def test_control_address_when_enabled(self, tmp_path: Path) -> None:
        cfg = ZMQDualBindProxyConfig(
            ipc_path=tmp_path, name="test", enable_control=True
        )
        addr = cfg.control_address
        assert addr is not None
        assert "test_control.ipc" in addr

    def test_capture_address_when_enabled(self, tmp_path: Path) -> None:
        cfg = ZMQDualBindProxyConfig(
            ipc_path=tmp_path, name="test", enable_capture=True
        )
        addr = cfg.capture_address
        assert addr is not None
        assert "test_capture.ipc" in addr

    def test_ipc_addr_raises_without_path(self) -> None:
        cfg = ZMQDualBindProxyConfig(name="test")
        with pytest.raises(ValueError, match="IPC path is required"):
            _ = cfg.frontend_address

    @pytest.mark.parametrize(
        "remote_host,expected_prefix",
        [
            param(None, "ipc://", id="local-uses-ipc"),
            param("controller.svc.cluster.local", "tcp://controller.svc.cluster.local:", id="remote-uses-tcp"),
        ],
    )  # fmt: skip
    def test_resolve_frontend(
        self,
        proxy_config: ZMQDualBindProxyConfig,
        remote_host: str | None,
        expected_prefix: str,
    ) -> None:
        addr = proxy_config.resolve_frontend(remote_host)
        assert addr.startswith(expected_prefix)

    @pytest.mark.parametrize(
        "remote_host,expected_prefix",
        [
            param(None, "ipc://", id="local-uses-ipc"),
            param("10.0.0.5", "tcp://10.0.0.5:", id="remote-uses-tcp"),
        ],
    )  # fmt: skip
    def test_resolve_backend(
        self,
        proxy_config: ZMQDualBindProxyConfig,
        remote_host: str | None,
        expected_prefix: str,
    ) -> None:
        addr = proxy_config.resolve_backend(remote_host)
        assert addr.startswith(expected_prefix)

    def test_resolve_frontend_remote_uses_correct_port(
        self, proxy_config: ZMQDualBindProxyConfig
    ) -> None:
        addr = proxy_config.resolve_frontend("myhost")
        assert addr == "tcp://myhost:5663"

    def test_resolve_backend_remote_uses_correct_port(
        self, proxy_config: ZMQDualBindProxyConfig
    ) -> None:
        addr = proxy_config.resolve_backend("myhost")
        assert addr == "tcp://myhost:5664"


# =============================================================================
# ZMQDualBindConfig
# =============================================================================


class TestZMQDualBindConfig:
    """Test ZMQDualBindConfig deployment-mode address selection."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> ZMQDualBindConfig:
        return ZMQDualBindConfig(ipc_path=tmp_path, tcp_host="0.0.0.0")

    @pytest.fixture
    def remote_config(self, tmp_path: Path) -> ZMQDualBindConfig:
        return ZMQDualBindConfig(
            ipc_path=tmp_path,
            tcp_host="0.0.0.0",
            controller_host="controller.default.svc",
        )

    def test_default_creates_temp_path_when_ipc_path_none(self) -> None:
        cfg = ZMQDualBindConfig()
        assert cfg.ipc_path is not None

    def test_config_does_not_create_directory(self, tmp_path: Path) -> None:
        ipc_dir = tmp_path / "nonexistent" / "ipc"
        cfg = ZMQDualBindConfig(ipc_path=ipc_dir)
        assert cfg.ipc_path == ipc_dir
        assert not ipc_dir.exists()

    def test_validator_propagates_ipc_path_to_proxies(
        self, config: ZMQDualBindConfig
    ) -> None:
        for proxy in config.proxy_configs:
            assert proxy.ipc_path == config.ipc_path

    def test_validator_propagates_tcp_host_to_proxies(
        self, config: ZMQDualBindConfig
    ) -> None:
        for proxy in config.proxy_configs:
            assert proxy.tcp_host == "0.0.0.0"

    def test_proxy_configs_returns_three_proxies(
        self, config: ZMQDualBindConfig
    ) -> None:
        assert len(config.proxy_configs) == 3

    def test_default_tcp_ports(self, config: ZMQDualBindConfig) -> None:
        assert config.records_push_pull_tcp_port == 5557
        assert config.credit_router_tcp_port == 5564

    def test_default_proxy_tcp_ports(self, config: ZMQDualBindConfig) -> None:
        assert config.event_bus_proxy_config.tcp_frontend_port == 5663
        assert config.event_bus_proxy_config.tcp_backend_port == 5664
        assert config.dataset_manager_proxy_config.tcp_frontend_port == 5661
        assert config.dataset_manager_proxy_config.tcp_backend_port == 5662
        assert config.raw_inference_proxy_config.tcp_frontend_port == 5665
        assert config.raw_inference_proxy_config.tcp_backend_port == 5666

    # --- Local mode (controller_host=None) ---

    def test_records_address_local_uses_ipc(self, config: ZMQDualBindConfig) -> None:
        assert config.records_push_pull_address.startswith("ipc://")

    def test_credit_router_address_local_uses_ipc(
        self, config: ZMQDualBindConfig
    ) -> None:
        assert config.credit_router_address.startswith("ipc://")

    def test_get_address_local_returns_ipc_for_all(
        self, config: ZMQDualBindConfig
    ) -> None:
        for addr_type in CommAddress:
            addr = config.get_address(addr_type)
            assert addr.startswith("ipc://"), f"{addr_type} should be IPC in local mode"

    # --- Remote mode (controller_host set) ---

    def test_records_address_remote_uses_tcp(
        self, remote_config: ZMQDualBindConfig
    ) -> None:
        assert (
            remote_config.records_push_pull_address
            == "tcp://controller.default.svc:5557"
        )

    def test_credit_router_address_remote_uses_tcp(
        self, remote_config: ZMQDualBindConfig
    ) -> None:
        assert (
            remote_config.credit_router_address == "tcp://controller.default.svc:5564"
        )

    def test_get_address_remote_returns_tcp_for_all(
        self, remote_config: ZMQDualBindConfig
    ) -> None:
        for addr_type in CommAddress:
            addr = remote_config.get_address(addr_type)
            assert addr.startswith("tcp://"), (
                f"{addr_type} should be TCP in remote mode"
            )

    def test_get_address_remote_invalid_type_raises(
        self, remote_config: ZMQDualBindConfig
    ) -> None:
        with pytest.raises(ValueError, match="Invalid address type"):
            remote_config.get_address("nonexistent_address")

    def test_get_address_local_invalid_type_raises(
        self, config: ZMQDualBindConfig
    ) -> None:
        with pytest.raises(ValueError, match="Invalid address type"):
            config.get_address("nonexistent_address")

    # --- TCP bind addresses (controller-side) ---

    def test_credit_router_tcp_bind_address(self, config: ZMQDualBindConfig) -> None:
        assert config.credit_router_tcp_bind_address == "tcp://0.0.0.0:5564"

    def test_records_push_pull_tcp_bind_address(
        self, config: ZMQDualBindConfig
    ) -> None:
        assert config.records_push_pull_tcp_bind_address == "tcp://0.0.0.0:5557"

    # --- K8s usage (no mkdir on config construction) ---

    def test_k8s_path_does_not_create_dir(self) -> None:
        """Config construction must not mkdir — the path may only exist inside a pod."""
        pod_path = Path("/nonexistent/path/aiperf/ipc")
        cfg = ZMQDualBindConfig(ipc_path=pod_path, tcp_host="0.0.0.0")
        assert cfg.ipc_path == pod_path
        assert not pod_path.exists()
        for proxy in cfg.proxy_configs:
            assert proxy.ipc_path == pod_path
            assert proxy.tcp_host == "0.0.0.0"

    # --- comm_backend ClassVar ---

    def test_comm_backend_is_dual_bind(self) -> None:
        assert ZMQDualBindConfig.comm_backend == CommunicationBackend.ZMQ_DUAL_BIND


# =============================================================================
# ZMQDualBindCommunication
# =============================================================================


class TestZMQDualBindCommunication:
    """Test ZMQDualBindCommunication class."""

    def test_init_with_default_config(self) -> None:
        comm = ZMQDualBindCommunication()
        assert comm.config is not None
        assert isinstance(comm.config, ZMQDualBindConfig)
        assert comm.state == LifecycleState.CREATED

    def test_init_with_custom_config(self, tmp_path: Path) -> None:
        config = ZMQDualBindConfig(ipc_path=tmp_path)
        comm = ZMQDualBindCommunication(config=config)
        assert comm.config is config

    def test_plugin_registered(self) -> None:
        names = [e.name for e in plugins.iter_entries(PluginType.COMMUNICATION)]
        assert CommunicationBackend.ZMQ_DUAL_BIND in names

    def test_plugin_resolves_to_class(self) -> None:
        cls = plugins.get_class(
            PluginType.COMMUNICATION, CommunicationBackend.ZMQ_DUAL_BIND
        )
        assert cls is ZMQDualBindCommunication


# =============================================================================
# PullClient dual-bind
# =============================================================================


class TestPullClientDualBind:
    """Test ZMQPullClient additional_bind_address support."""

    def test_stores_additional_bind_address_when_bind_true(
        self, mock_zmq_context
    ) -> None:
        client = ZMQPullClient(
            address="ipc:///tmp/records.ipc",
            bind=True,
            additional_bind_address="tcp://0.0.0.0:5557",
        )
        assert client.additional_bind_address == "tcp://0.0.0.0:5557"

    def test_ignores_additional_bind_address_when_bind_false(
        self, mock_zmq_context
    ) -> None:
        client = ZMQPullClient(
            address="ipc:///tmp/records.ipc",
            bind=False,
            additional_bind_address="tcp://0.0.0.0:5557",
        )
        assert client.additional_bind_address is None

    def test_no_additional_bind_by_default(self, mock_zmq_context) -> None:
        client = ZMQPullClient(address="ipc:///tmp/records.ipc", bind=True)
        assert client.additional_bind_address is None

    @pytest.mark.asyncio
    async def test_on_init_binds_additional_address(
        self, mock_zmq_socket, mock_zmq_context
    ) -> None:
        client = ZMQPullClient(
            address="ipc:///tmp/records.ipc",
            bind=True,
            additional_bind_address="tcp://0.0.0.0:5557",
        )
        await client.initialize()

        # Primary bind + dual-bind
        bind_calls = [str(c) for c in mock_zmq_socket.bind.call_args_list]
        tcp_binds = [c for c in bind_calls if "5557" in c]
        assert len(tcp_binds) >= 1

    @pytest.mark.asyncio
    async def test_on_init_skips_when_no_additional(
        self, mock_zmq_socket, mock_zmq_context
    ) -> None:
        client = ZMQPullClient(address="ipc:///tmp/records.ipc", bind=True)
        await client.initialize()

        # Only primary bind, no TCP bind
        bind_calls = [str(c) for c in mock_zmq_socket.bind.call_args_list]
        tcp_binds = [c for c in bind_calls if "tcp://" in c]
        assert len(tcp_binds) == 0


# =============================================================================
# StreamingRouterClient dual-bind
# =============================================================================


class TestStreamingRouterClientDualBind:
    """Test ZMQStreamingRouterClient additional_bind_address support."""

    def test_stores_additional_bind_address_when_bind_true(
        self, mock_zmq_context
    ) -> None:
        client = ZMQStreamingRouterClient(
            address="ipc:///tmp/credit_router.ipc",
            bind=True,
            additional_bind_address="tcp://0.0.0.0:5564",
        )
        assert client.additional_bind_address == "tcp://0.0.0.0:5564"

    def test_ignores_additional_bind_address_when_bind_false(
        self, mock_zmq_context
    ) -> None:
        client = ZMQStreamingRouterClient(
            address="ipc:///tmp/credit_router.ipc",
            bind=False,
            additional_bind_address="tcp://0.0.0.0:5564",
        )
        assert client.additional_bind_address is None

    @pytest.mark.asyncio
    async def test_on_init_binds_additional_address(
        self, mock_zmq_socket, mock_zmq_context
    ) -> None:
        client = ZMQStreamingRouterClient(
            address="ipc:///tmp/credit_router.ipc",
            bind=True,
            additional_bind_address="tcp://0.0.0.0:5564",
        )
        await client.initialize()

        bind_calls = [str(c) for c in mock_zmq_socket.bind.call_args_list]
        tcp_binds = [c for c in bind_calls if "5564" in c]
        assert len(tcp_binds) >= 1


# =============================================================================
# BaseZMQProxy dual-bind
# =============================================================================


class TestProxyDualBind:
    """Test BaseZMQProxy dual-bind socket binding."""

    @pytest.mark.asyncio
    async def test_proxy_has_additional_bind_addresses(
        self, mock_zmq_context, tmp_path: Path
    ) -> None:
        proxy_config = ZMQDualBindProxyConfig(
            ipc_path=tmp_path,
            name="event_bus_proxy",
            tcp_host="0.0.0.0",
            tcp_frontend_port=5663,
            tcp_backend_port=5664,
        )
        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)
        assert proxy.config.additional_frontend_bind_address == "tcp://0.0.0.0:5663"
        assert proxy.config.additional_backend_bind_address == "tcp://0.0.0.0:5664"

    @pytest.mark.asyncio
    async def test_proxy_no_additional_bind_for_tcp_config(
        self, mock_zmq_context
    ) -> None:
        tcp_config = ZMQTCPConfig()
        proxy = ZMQXPubXSubProxy.from_config(tcp_config.event_bus_proxy_config)
        assert proxy.config.additional_frontend_bind_address is None
        assert proxy.config.additional_backend_bind_address is None

    @pytest.mark.asyncio
    async def test_proxy_initialize_binds_both_ipc_and_tcp(
        self, mock_zmq_socket, mock_zmq_context, tmp_path: Path
    ) -> None:
        proxy_config = ZMQDualBindProxyConfig(
            ipc_path=tmp_path,
            name="event_bus_proxy",
            tcp_host="0.0.0.0",
            tcp_frontend_port=5663,
            tcp_backend_port=5664,
        )
        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)
        await proxy.initialize()

        # Should have binds for IPC frontend, IPC backend, TCP frontend, TCP backend
        bind_calls = [str(c) for c in mock_zmq_socket.bind.call_args_list]
        ipc_binds = [c for c in bind_calls if "ipc://" in c]
        tcp_binds = [c for c in bind_calls if "tcp://" in c]

        assert len(ipc_binds) >= 2, f"Expected >= 2 IPC binds, got {ipc_binds}"
        assert len(tcp_binds) >= 2, f"Expected >= 2 TCP binds, got {tcp_binds}"

    @pytest.mark.asyncio
    async def test_proxy_dual_bind_uses_correct_tcp_addresses(
        self, mock_zmq_socket, mock_zmq_context, tmp_path: Path
    ) -> None:
        proxy_config = ZMQDualBindProxyConfig(
            ipc_path=tmp_path,
            name="event_bus_proxy",
            tcp_host="0.0.0.0",
            tcp_frontend_port=5663,
            tcp_backend_port=5664,
        )
        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)
        await proxy.initialize()

        bind_args = [
            c[0][0] if c[0] else c[1].get("addr", "")
            for c in mock_zmq_socket.bind.call_args_list
        ]
        assert "tcp://0.0.0.0:5663" in bind_args
        assert "tcp://0.0.0.0:5664" in bind_args


# =============================================================================
# BaseCommunication dual-bind plumbing
# =============================================================================


class TestBaseCommunicationDualBind:
    """Test that BaseCommunication passes additional_bind_address through."""

    def test_create_pull_client_passes_additional_bind(self) -> None:
        comm = ZMQDualBindCommunication()
        client = comm.create_pull_client(
            address="ipc:///tmp/records.ipc",
            bind=True,
            additional_bind_address="tcp://0.0.0.0:5557",
        )
        assert client.additional_bind_address == "tcp://0.0.0.0:5557"

    def test_create_streaming_router_client_passes_additional_bind(self) -> None:
        comm = ZMQDualBindCommunication()
        client = comm.create_streaming_router_client(
            address="ipc:///tmp/credit_router.ipc",
            bind=True,
            additional_bind_address="tcp://0.0.0.0:5564",
        )
        assert client.additional_bind_address == "tcp://0.0.0.0:5564"

    def test_create_pull_client_without_additional_bind(self) -> None:
        comm = ZMQDualBindCommunication()
        client = comm.create_pull_client(
            address="ipc:///tmp/records.ipc",
            bind=True,
        )
        assert client.additional_bind_address is None


# =============================================================================
# BaseZMQProxyConfig base resolve methods
# =============================================================================


class TestBaseZMQProxyConfigResolve:
    """Test that base resolve_frontend/resolve_backend return the address directly."""

    def test_resolve_backend_returns_backend_address(self) -> None:
        config = ZMQTCPProxyConfig(frontend_port=5555, backend_port=5556)
        assert config.resolve_backend() == config.backend_address

    def test_resolve_backend_ignores_remote_host(self) -> None:
        config = ZMQTCPProxyConfig(frontend_port=5555, backend_port=5556)
        assert config.resolve_backend("remote.host") == config.backend_address


# =============================================================================
# ZMQIPCProxyConfig
# =============================================================================


class TestZMQIPCProxyConfig:
    """Test ZMQIPCProxyConfig address construction."""

    def test_frontend_address(self, tmp_path: Path) -> None:
        cfg = ZMQIPCProxyConfig(path=tmp_path, name="test")
        assert cfg.frontend_address == f"ipc://{tmp_path / 'test'}_frontend.ipc"

    def test_backend_address(self, tmp_path: Path) -> None:
        cfg = ZMQIPCProxyConfig(path=tmp_path, name="test")
        assert cfg.backend_address == f"ipc://{tmp_path / 'test'}_backend.ipc"

    def test_control_address_disabled(self, tmp_path: Path) -> None:
        cfg = ZMQIPCProxyConfig(path=tmp_path, name="test")
        assert cfg.control_address is None

    def test_control_address_enabled(self, tmp_path: Path) -> None:
        cfg = ZMQIPCProxyConfig(path=tmp_path, name="test", enable_control=True)
        assert "test_control.ipc" in cfg.control_address

    def test_capture_address_disabled(self, tmp_path: Path) -> None:
        cfg = ZMQIPCProxyConfig(path=tmp_path, name="test")
        assert cfg.capture_address is None

    def test_capture_address_enabled(self, tmp_path: Path) -> None:
        cfg = ZMQIPCProxyConfig(path=tmp_path, name="test", enable_capture=True)
        assert "test_capture.ipc" in cfg.capture_address

    def test_addr_raises_when_path_is_none(self) -> None:
        cfg = ZMQIPCProxyConfig(name="test")
        with pytest.raises(ValueError, match="Path is required"):
            _ = cfg.frontend_address


# =============================================================================
# ZMQDualBindConfig._ipc_addr error path
# =============================================================================


class TestZMQDualBindConfigIPCAddrError:
    """Test _ipc_addr raises when ipc_path is cleared."""

    def test_ipc_addr_raises_when_ipc_path_cleared(self) -> None:
        cfg = ZMQDualBindConfig()
        cfg.ipc_path = None
        with pytest.raises(ValueError, match="IPC path is required"):
            _ = cfg.records_push_pull_address


# =============================================================================
# BaseZMQClient additional_bind_address edge cases
# =============================================================================


class TestBaseZMQClientAdditionalBind:
    """Test BaseZMQClient additional_bind_address warning and IPC cleanup."""

    def test_warns_when_bind_false_with_additional_address(
        self, mock_zmq_context
    ) -> None:
        client = ZMQPullClient(
            address="tcp://127.0.0.1:5555",
            bind=False,
            additional_bind_address="tcp://0.0.0.0:5557",
        )
        assert client.additional_bind_address is None

    @pytest.mark.asyncio
    async def test_cleanup_ipc_file_on_stop(
        self, mock_zmq_socket, mock_zmq_context, tmp_path: Path
    ) -> None:
        ipc_file = tmp_path / "test.ipc"
        ipc_file.touch()
        client = ZMQPullClient(
            address=f"ipc://{ipc_file}",
            bind=True,
        )
        await client.initialize()
        await client.stop()
        assert not ipc_file.exists()


# =============================================================================
# Service-level dual-bind wiring (RecordsManager + StickyRouter)
# =============================================================================


class _DualBindServiceFixtures:
    """Shared fixtures for service-level dual-bind tests."""

    @pytest.fixture
    def dual_bind_service_config(self, tmp_path: Path) -> ServiceConfig:
        config = ServiceConfig()
        config._comm_config = ZMQDualBindConfig(ipc_path=tmp_path, tcp_host="0.0.0.0")
        return config

    @pytest.fixture
    def remote_dual_bind_service_config(self, tmp_path: Path) -> ServiceConfig:
        config = ServiceConfig()
        config._comm_config = ZMQDualBindConfig(
            ipc_path=tmp_path,
            tcp_host="0.0.0.0",
            controller_host="controller.default.svc",
        )
        return config

    @pytest.fixture
    def user_config(self) -> UserConfig:
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


class TestRecordsManagerDualBind(_DualBindServiceFixtures):
    """Test RecordsManager passes additional_bind_address in dual-bind mode."""

    def test_controller_mode_passes_tcp_bind_address(
        self, dual_bind_service_config, user_config
    ) -> None:
        manager = RecordsManager(
            service_config=dual_bind_service_config,
            user_config=user_config,
        )
        assert manager.pull_client.additional_bind_address == "tcp://0.0.0.0:5557"

    def test_worker_mode_does_not_pass_tcp_bind_address(
        self, remote_dual_bind_service_config, user_config
    ) -> None:
        manager = RecordsManager(
            service_config=remote_dual_bind_service_config,
            user_config=user_config,
        )
        assert manager.pull_client.additional_bind_address is None

    def test_ipc_mode_does_not_pass_tcp_bind_address(self, user_config) -> None:
        config = ServiceConfig()
        manager = RecordsManager(
            service_config=config,
            user_config=user_config,
        )
        assert manager.pull_client.additional_bind_address is None


class TestStickyRouterDualBind(_DualBindServiceFixtures):
    """Test StickyCreditRouter passes additional_bind_address in dual-bind mode."""

    def test_controller_mode_passes_tcp_bind_address(
        self, dual_bind_service_config
    ) -> None:
        router = StickyCreditRouter(
            service_config=dual_bind_service_config,
            service_id="test-router",
        )
        assert router._router_client.additional_bind_address == "tcp://0.0.0.0:5564"

    def test_worker_mode_does_not_pass_tcp_bind_address(
        self, remote_dual_bind_service_config
    ) -> None:
        router = StickyCreditRouter(
            service_config=remote_dual_bind_service_config,
            service_id="test-router",
        )
        assert router._router_client.additional_bind_address is None

    def test_ipc_mode_does_not_pass_tcp_bind_address(self) -> None:
        config = ServiceConfig()
        router = StickyCreditRouter(
            service_config=config,
            service_id="test-router",
        )
        assert router._router_client.additional_bind_address is None
