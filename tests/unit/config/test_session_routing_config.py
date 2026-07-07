# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.config.endpoint import EndpointConfig
from aiperf.config.flags._converter_endpoint import _parse_routing_opts


def _config(**kwargs) -> EndpointConfig:
    return EndpointConfig(urls=["http://localhost:8000"], **kwargs)


def test_defaults_off():
    config = _config()
    assert config.session_routing is None
    assert config.session_routing_opts == {}


def test_mode_with_valid_opts():
    config = _config(
        session_routing="dynamo_nvext",
        session_routing_opts={"timeout_seconds": "600"},
    )
    assert str(config.session_routing) == "dynamo_nvext"


def test_opts_canonicalized_to_typed_values():
    config = _config(
        session_routing="dynamo_nvext",
        session_routing_opts={"timeout_seconds": "600"},
    )
    assert config.session_routing_opts == {"timeout_seconds": 600}
    assert isinstance(config.session_routing_opts["timeout_seconds"], int)


def test_canonicalization_idempotent():
    config = _config(
        session_routing="dynamo_nvext",
        session_routing_opts={"timeout_seconds": 600},
    )
    assert config.session_routing_opts == {"timeout_seconds": 600}


def test_opts_without_mode_rejected():
    with pytest.raises(ValueError, match="session-routing-opt"):
        _config(session_routing_opts={"header_name": "X-A"})


def test_unknown_opt_key_rejected():
    with pytest.raises(ValueError, match="timeout_secs"):
        _config(
            session_routing="dynamo_nvext",
            session_routing_opts={"timeout_secs": "600"},
        )


def test_invalid_opt_value_rejected():
    with pytest.raises(ValueError):
        _config(
            session_routing="dynamo_nvext",
            session_routing_opts={"timeout_seconds": "0"},
        )


def test_parameterless_mode_rejects_any_opt():
    with pytest.raises(ValueError):
        _config(
            session_routing="smg_routing_key",
            session_routing_opts={"anything": "x"},
        )


def test_parse_routing_opts_duplicate_key_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        _parse_routing_opts(["header_name=X-A", "header_name=X-B"])


@pytest.mark.parametrize(
    "item",
    [
        param("noequals", id="no_separator"),
        param("key=", id="empty_value"),
    ],
)  # fmt: skip
def test_parse_routing_opts_malformed_pair_rejected(item):
    with pytest.raises(ValueError, match="expected non-empty key=value"):
        _parse_routing_opts([item])
