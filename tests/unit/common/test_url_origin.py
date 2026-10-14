# tests/unit/common/test_url_origin.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.url_origin import (
    origin_of,
    same_origin,
    strip_credentials_for_foreign_origin,
)


def test_default_ports_normalize() -> None:
    assert origin_of("https://h.example/x") == ("https", "h.example", 443)
    assert same_origin("http://h:80/a", "http://h/b")
    assert not same_origin("https://h/a", "http://h/a")
    assert not same_origin("https://a.example/x", "https://b.example/x")
    assert origin_of("http://[bad") is None


def test_unparseable_port_fails_closed() -> None:
    assert origin_of("https://h.example:notaport/x") is None
    assert not same_origin(
        "https://h.example:notaport/x", "https://h.example:notaport/x"
    )


def test_strip_credentials_keeps_only_allowlisted_names_case_insensitively() -> None:
    headers = {
        "Authorization": "Bearer t",
        "x-acme-token": "s",
        "User-Agent": "ua",
        "Range": "bytes=0-1",
    }
    assert strip_credentials_for_foreign_origin(headers) == {"User-Agent": "ua"}
    assert strip_credentials_for_foreign_origin(
        headers, extra_allowed=frozenset({"range"})
    ) == {
        "User-Agent": "ua",
        "Range": "bytes=0-1",
    }
