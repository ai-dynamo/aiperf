# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `build_socket_address` — the Linux/macOS ipc:// vs Windows tcp:// helper.

Covers Bug 2: ZMQ ipc:// is not supported on Windows (pyzmq wheels disable it
due to crashes), so AIPerf falls back to tcp://127.0.0.1:<deterministic-port>
on Windows. Same path/filename inputs must hash to the same port so bind and
connect sides agree without coordination.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.comm.ipc import (
    _validate_no_port_collisions,
    build_socket_address,
)

# Default port range (matches Environment.SERVICE defaults; can be overridden
# at runtime via AIPERF_SERVICE_WINDOWS_TCP_{BASE_PORT,PORT_RANGE}).
_WINDOWS_TCP_BASE_PORT = Environment.SERVICE.WINDOWS_TCP_BASE_PORT
_WINDOWS_TCP_PORT_RANGE = Environment.SERVICE.WINDOWS_TCP_PORT_RANGE


class TestBuildSocketAddressLinux:
    """`build_socket_address` returns ipc:// when not on Windows."""

    @pytest.fixture(autouse=True)
    def _force_not_windows(self):
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", False):
            yield

    def test_returns_ipc_url_with_path_and_filename(self, tmp_path: Path) -> None:
        address = build_socket_address(tmp_path, "event_bus.ipc")
        assert address == f"ipc://{tmp_path / 'event_bus.ipc'}"

    def test_path_none_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match=r"[Pp]ath is required for socket address derivation"
        ):
            build_socket_address(None, "event_bus.ipc")


class TestBuildSocketAddressWindows:
    """`build_socket_address` returns tcp:// with deterministic port on Windows."""

    @pytest.fixture(autouse=True)
    def _force_windows(self):
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            yield

    def test_returns_tcp_loopback_url(self, tmp_path: Path) -> None:
        address = build_socket_address(tmp_path, "event_bus.ipc")
        assert address.startswith("tcp://127.0.0.1:")

    def test_port_within_configured_range(self, tmp_path: Path) -> None:
        address = build_socket_address(tmp_path, "event_bus.ipc")
        port = int(address.rsplit(":", 1)[1])
        assert (
            _WINDOWS_TCP_BASE_PORT
            <= port
            < _WINDOWS_TCP_BASE_PORT + _WINDOWS_TCP_PORT_RANGE
        )

    def test_same_inputs_produce_same_port(self, tmp_path: Path) -> None:
        addr1 = build_socket_address(tmp_path, "event_bus.ipc")
        addr2 = build_socket_address(tmp_path, "event_bus.ipc")
        assert addr1 == addr2

    def test_different_filenames_produce_different_ports(self, tmp_path: Path) -> None:
        addr1 = build_socket_address(tmp_path, "event_bus.ipc")
        addr2 = build_socket_address(tmp_path, "credit_router.ipc")
        assert addr1 != addr2

    def test_different_paths_produce_different_ports(self, tmp_path: Path) -> None:
        other_path = tmp_path / "other"
        other_path.mkdir()
        addr1 = build_socket_address(tmp_path, "event_bus.ipc")
        addr2 = build_socket_address(other_path, "event_bus.ipc")
        assert addr1 != addr2

    def test_path_none_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match=r"[Pp]ath is required for socket address derivation"
        ):
            build_socket_address(None, "event_bus.ipc")

    def test_salt_canonicalization_normalizes_separators_and_case(
        self, tmp_path: Path
    ) -> None:
        """Bind and connect must agree on the derived port regardless of
        how the path was constructed. Different stringifications of the
        same logical path (backslash vs forward-slash separators, trailing
        slash, casing) must produce the same hashed port. Pins F-02.
        """
        addr_forward = build_socket_address(Path("C:/Temp/aiperf"), "x.ipc")
        addr_backslash = build_socket_address(Path("C:\\Temp\\aiperf"), "x.ipc")
        addr_trailing = build_socket_address(Path("C:/Temp/aiperf/"), "x.ipc")
        addr_mixed_case = build_socket_address(Path("c:/temp/aiperf"), "x.ipc")
        assert addr_forward == addr_backslash == addr_trailing == addr_mixed_case

    @pytest.mark.parametrize(
        "filename",
        [
            param("event_bus_proxy_frontend.ipc", id="event_bus_frontend"),
            param("event_bus_proxy_backend.ipc", id="event_bus_backend"),
            param("records_push_pull.ipc", id="records"),
            param("credit_router.ipc", id="credit"),
            param("dataset_manager_proxy_frontend.ipc", id="dataset_frontend"),
            param("dataset_manager_proxy_backend.ipc", id="dataset_backend"),
            param("raw_inference_proxy_frontend.ipc", id="raw_inference_frontend"),
            param("raw_inference_proxy_backend.ipc", id="raw_inference_backend"),
        ],
    )
    def test_realistic_filenames_all_within_range(
        self, tmp_path: Path, filename: str
    ) -> None:
        address = build_socket_address(tmp_path, filename)
        port = int(address.rsplit(":", 1)[1])
        assert (
            _WINDOWS_TCP_BASE_PORT
            <= port
            < _WINDOWS_TCP_BASE_PORT + _WINDOWS_TCP_PORT_RANGE
        )


class TestBuildSocketAddressHashDistribution:
    """Sanity check that the hash distributes across the port range."""

    @pytest.fixture(autouse=True)
    def _force_windows(self):
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            yield

    def test_realistic_socket_set_has_no_collisions(self, tmp_path: Path) -> None:
        """Within a single AIPerf run, the 8 production socket filenames must hash to distinct ports.

        If this ever fails, increase _WINDOWS_TCP_PORT_RANGE or change a filename.
        Birthday paradox at RANGE=20000 with n=8 sockets: ~0.14% collision chance.
        """
        filenames = [
            "event_bus_proxy_frontend.ipc",
            "event_bus_proxy_backend.ipc",
            "records_push_pull.ipc",
            "credit_router.ipc",
            "dataset_manager_proxy_frontend.ipc",
            "dataset_manager_proxy_backend.ipc",
            "raw_inference_proxy_frontend.ipc",
            "raw_inference_proxy_backend.ipc",
        ]
        ports = {
            int(build_socket_address(tmp_path, fn).rsplit(":", 1)[1])
            for fn in filenames
        }
        assert len(ports) == len(filenames), (
            f"Hash collision detected for the production socket set: "
            f"{len(filenames)} sockets but only {len(ports)} unique ports. "
            f"Consider widening _WINDOWS_TCP_PORT_RANGE."
        )

    def test_port_range_avoids_common_service_ports(self, tmp_path: Path) -> None:
        """The TCP port range must not overlap common localhost service ports.

        Regression: an earlier range (5556-25556) overlapped vLLM, Prometheus,
        OTLP, Ollama, and HTTP backend defaults — Windows users with one of
        those services running would hit an opaque ``bind: address already in
        use`` ~0.7% of runs. The fix shifted the range above all common
        service ports while staying below the Windows ephemeral range (49152+).
        """
        # Non-exhaustive set of well-known localhost service ports. If a real
        # production service binds here, the user's aiperf run would fail at
        # startup with a confusing error — the range was chosen to avoid this.
        common_service_ports = {
            5000,  # Flask default
            5555,  # ZMQ canonical example
            5556,  # vLLM monitoring (and prior aiperf base)
            6379,  # Redis
            8000,
            8001,
            8080,
            8081,
            8082,  # HTTP backends (uvicorn, vLLM API, etc.)
            8443,  # HTTPS
            8765,  # aiperf mock server (per integration tests)
            9000,
            9090,
            9091,  # Prometheus
            9400,
            9401,  # DCGM exporter
            11211,  # memcached
            11434,  # Ollama
            27017,  # MongoDB
            50051,  # gRPC default
            4318,  # OTLP HTTP
        }
        port_range_top = _WINDOWS_TCP_BASE_PORT + _WINDOWS_TCP_PORT_RANGE
        overlaps = sorted(
            p
            for p in common_service_ports
            if _WINDOWS_TCP_BASE_PORT <= p < port_range_top
        )
        assert not overlaps, (
            f"Port range {_WINDOWS_TCP_BASE_PORT}-{port_range_top} overlaps "
            f"common service ports {overlaps}. Shift _WINDOWS_TCP_BASE_PORT "
            f"or narrow _WINDOWS_TCP_PORT_RANGE so no common service port "
            f"falls inside."
        )
        assert port_range_top <= 49152, (
            f"Port range top {port_range_top} enters Windows ephemeral range "
            f"(49152+); outbound connections may collide."
        )


class TestPortCollisionValidation:
    """Pins F-01: ``_validate_no_port_collisions`` raises a clear, actionable
    error when two derived TCP ports collide on Windows, and is a no-op on
    POSIX where ipc:// is used.
    """

    def test_no_collisions_is_silent(self) -> None:
        """The normal case (no two endpoints share a port) returns None."""
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            _validate_no_port_collisions(
                [
                    ("a", "tcp://127.0.0.1:30000"),
                    ("b", "tcp://127.0.0.1:30001"),
                    ("c", "tcp://127.0.0.1:30002"),
                ]
            )

    def test_collision_raises_with_actionable_message(self) -> None:
        """When two endpoints hash to the same port, the error names both
        endpoints and points the user at the env var escape hatch."""
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            with pytest.raises(ValueError, match=r"port collision") as exc:
                _validate_no_port_collisions(
                    [
                        ("records_push_pull", "tcp://127.0.0.1:30518"),
                        ("control", "tcp://127.0.0.1:30518"),
                    ]
                )
            msg = str(exc.value)
            assert "records_push_pull" in msg
            assert "control" in msg
            assert "30518" in msg
            assert "AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT" in msg

    def test_ipc_addresses_are_ignored(self) -> None:
        """Non-TCP addresses (POSIX ipc://) are skipped — the helper only
        cares about TCP loopback collisions."""
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            _validate_no_port_collisions(
                [
                    ("a", "ipc:///tmp/foo.ipc"),
                    ("b", "ipc:///tmp/bar.ipc"),
                ]
            )

    def test_posix_is_noop(self) -> None:
        """On POSIX the function returns immediately — ipc:// addresses
        cannot collide on a port the way TCP loopback ones can."""
        with patch("aiperf.config.comm.ipc.IS_WINDOWS", False):
            # Same port in both addresses; would raise on Windows.
            _validate_no_port_collisions(
                [
                    ("a", "tcp://127.0.0.1:30000"),
                    ("b", "tcp://127.0.0.1:30000"),
                ]
            )


class TestResolveCollisionSalt:
    """Pins the salt-retry behavior: ``_resolve_collision_salt`` rotates the
    salt until ``compute_addresses(salt)`` has no port collision, then
    returns that salt. This is what keeps Windows CI from flaking on the
    ~0.56% birthday-paradox collision rate.
    """

    def test_first_salt_wins_when_no_collision(self) -> None:
        """If the default-zero salt already has no collision, return ``""``
        immediately — don't waste retries when the first draw is good."""
        from aiperf.config.comm.ipc import _resolve_collision_salt

        def compute(salt: str) -> list[tuple[str, str]]:
            return [
                ("a", "tcp://127.0.0.1:30000"),
                ("b", "tcp://127.0.0.1:30001"),
            ]

        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            assert _resolve_collision_salt(compute) == ""

    def test_retries_until_no_collision(self) -> None:
        """If salts ``""`` and ``":1"`` collide but ``":2"`` doesn't, the
        loop returns ``":2"``. Proves the retry actually probes different
        salts rather than blindly returning the first one."""
        from aiperf.config.comm.ipc import _resolve_collision_salt

        attempts: list[str] = []

        def compute(salt: str) -> list[tuple[str, str]]:
            attempts.append(salt)
            if salt in ("", ":1"):
                return [
                    ("a", "tcp://127.0.0.1:30000"),
                    ("b", "tcp://127.0.0.1:30000"),
                ]
            return [
                ("a", "tcp://127.0.0.1:30000"),
                ("b", "tcp://127.0.0.1:30001"),
            ]

        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            assert _resolve_collision_salt(compute) == ":2"
        assert attempts == ["", ":1", ":2"]

    def test_posix_returns_empty_salt_no_iteration(self) -> None:
        """On POSIX salt rotation is irrelevant — ipc:// addresses cannot
        collide on a port. Return ``""`` immediately without calling
        ``compute_addresses`` even once (cheap fast path)."""
        from aiperf.config.comm.ipc import _resolve_collision_salt

        calls = 0

        def compute(salt: str) -> list[tuple[str, str]]:
            nonlocal calls
            calls += 1
            return [("a", "tcp://127.0.0.1:30000"), ("b", "tcp://127.0.0.1:30000")]

        with patch("aiperf.config.comm.ipc.IS_WINDOWS", False):
            assert _resolve_collision_salt(compute) == ""
        assert calls == 0

    def test_exhausting_retries_raises(self) -> None:
        """If every salt attempt collides (statistically impossible in
        practice, but possible if a caller bug means salt isn't actually
        in the hash input), raise so the failure is loud rather than
        silently shipping colliding ports."""
        from aiperf.config.comm.ipc import _resolve_collision_salt

        def compute(salt: str) -> list[tuple[str, str]]:
            return [
                ("a", "tcp://127.0.0.1:30000"),
                ("b", "tcp://127.0.0.1:30000"),
            ]

        with (
            patch("aiperf.config.comm.ipc.IS_WINDOWS", True),
            pytest.raises(ValueError, match=r"port collision"),
        ):
            _resolve_collision_salt(compute)


class TestZMQIPCConfigSaltThreading:
    """Pins the salt-retry integration on ``ZMQIPCConfig``: when constructed
    under ``IS_WINDOWS=True``, ``validate_path`` runs ``_resolve_collision_salt``,
    which calls ``_addresses_with_salt`` to derive every endpoint's address
    under each candidate salt. Without these tests, those helper methods (and
    the salt-threading wiring through property methods) are dead code from
    a coverage standpoint on the POSIX CI runner. Exercising them via mocked
    Windows brings the salt-retry code path under coverage even on Linux.
    """

    def test_windows_config_construction_runs_retry_loop(self, tmp_path: Path) -> None:
        """Constructing a ``ZMQIPCConfig`` on Windows must invoke
        ``_addresses_with_salt`` at least once and pick a salt that yields
        non-colliding tcp:// addresses for every endpoint.
        """
        from aiperf.config.comm.ipc import ZMQIPCConfig

        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            config = ZMQIPCConfig(path=tmp_path / "aiperf")
            # Access properties INSIDE the patch context — the IS_WINDOWS
            # check in ``build_socket_address`` fires at every call, not at
            # validate-time.
            addresses = [
                config.records_push_pull_address,
                config.credit_router_address,
                config.credit_return_router_address,
                config.control_address,
                config.group_lifecycle_address,
                config.dataset_manager_proxy_config.frontend_address,
                config.dataset_manager_proxy_config.backend_address,
                config.event_bus_proxy_config.frontend_address,
                config.event_bus_proxy_config.backend_address,
                config.raw_inference_proxy_config.frontend_address,
                config.raw_inference_proxy_config.backend_address,
            ]

        # Every endpoint should be tcp://127.0.0.1:<port> on Windows, with
        # distinct ports — the salt-retry loop guarantees no collision.
        for addr in addresses:
            assert addr.startswith("tcp://127.0.0.1:"), addr
        ports = [int(a.rsplit(":", 1)[1]) for a in addresses]
        assert len(set(ports)) == len(ports), (
            f"Port collision survived the retry loop: {ports}"
        )

    def test_windows_salt_propagates_to_proxy_configs(self, tmp_path: Path) -> None:
        """Whatever salt the parent picks must propagate to every proxy
        config so the proxy's own property methods derive ports under the
        SAME hash input. Otherwise bind and connect sides would disagree.
        """
        from aiperf.config.comm.ipc import ZMQIPCConfig

        with patch("aiperf.config.comm.ipc.IS_WINDOWS", True):
            config = ZMQIPCConfig(path=tmp_path / "aiperf")

        parent_salt = config._collision_salt
        for proxy in (
            config.dataset_manager_proxy_config,
            config.event_bus_proxy_config,
            config.raw_inference_proxy_config,
        ):
            assert proxy._collision_salt == parent_salt, (
                f"Proxy {proxy.name} salt {proxy._collision_salt!r} != "
                f"parent salt {parent_salt!r}"
            )
