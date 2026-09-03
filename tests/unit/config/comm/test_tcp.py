# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `ZMQTCPConfig` default port derivation.

Guards against off-by-relative-offset port collisions between distinct
logical ZMQ channels: every address a default ``ZMQTCPConfig()`` exposes
must resolve to a unique host:port, otherwise two different sockets try to
bind the same TCP port and one of them silently fails to start.
"""

from __future__ import annotations

from aiperf.config.comm.tcp import ZMQTCPConfig


def _all_addresses(config: ZMQTCPConfig) -> dict[str, str]:
    """Every distinct logical address/port field exposed by a `ZMQTCPConfig`."""
    return {
        "records_push_pull_address": config.records_push_pull_address,
        "credit_router_address": config.credit_router_address,
        "credit_return_router_address": config.credit_return_router_address,
        "credit_return_push_pull_address": config.credit_return_push_pull_address,
        "control_address": config.control_address,
        "group_lifecycle_address": config.group_lifecycle_address,
        "dataset_manager_proxy_frontend": config.dataset_manager_proxy_config.frontend_address,
        "dataset_manager_proxy_backend": config.dataset_manager_proxy_config.backend_address,
        "event_bus_proxy_frontend": config.event_bus_proxy_config.frontend_address,
        "event_bus_proxy_backend": config.event_bus_proxy_config.backend_address,
        "raw_inference_proxy_frontend": config.raw_inference_proxy_config.frontend_address,
        "raw_inference_proxy_backend": config.raw_inference_proxy_config.backend_address,
    }


class TestZMQTCPConfigDefaultPortUniqueness:
    """Default `ZMQTCPConfig()` must not assign the same address to two channels."""

    def test_default_addresses_are_all_unique(self) -> None:
        config = ZMQTCPConfig()
        addresses = _all_addresses(config)

        seen: dict[str, str] = {}
        collisions = []
        for label, address in addresses.items():
            if address in seen:
                collisions.append((seen[address], label, address))
            else:
                seen[address] = label

        assert not collisions, (
            f"Duplicate default addresses found in ZMQTCPConfig(): {collisions}. "
            f"Full address map: {addresses}"
        )

    def test_group_lifecycle_does_not_collide_with_credit_return_router(self) -> None:
        config = ZMQTCPConfig()
        assert config.group_lifecycle_address != config.credit_return_router_address
