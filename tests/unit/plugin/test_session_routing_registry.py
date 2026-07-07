# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.workers.session_routing import SessionRoutingBase


@pytest.mark.parametrize(
    "name",
    [
        param("dynamo_headers", id="dynamo_headers"),
        param("dynamo_nvext", id="dynamo_nvext"),
        param("smg_routing_key", id="smg_routing_key"),
        param("session_id_header", id="session_id_header"),
    ],
)  # fmt: skip
def test_session_routing_plugins_resolve(name):
    cls = plugins.get_class(PluginType.SESSION_ROUTING, name)
    assert issubclass(cls, SessionRoutingBase)


def test_session_routing_enum_generated():
    from aiperf.plugin.enums import SessionRoutingType

    assert SessionRoutingType.DYNAMO_HEADERS == "dynamo_headers"
    assert SessionRoutingType.SESSION_ID_HEADER == "session_id_header"
