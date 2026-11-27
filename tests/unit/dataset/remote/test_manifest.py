# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

import orjson
import pytest
from pytest import param

from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.manifest import check_compatibility, parse_manifest

REF = "hf://org/repo"
ROOT = {
    "aiperf_manifest_version": 1,
    "format": "mooncake_trace",
    "path": "data/t.jsonl",
}


def _raw(data: dict) -> bytes:
    return orjson.dumps(data)


def test_root_manifest_parses() -> None:
    m = parse_manifest(_raw({**ROOT, "block_size": 16}), ref_display=REF, kind="root")
    assert (m.format, m.path, m.block_size) == ("mooncake_trace", "data/t.jsonl", 16)


@pytest.mark.parametrize(
    "data,kind,error_kind,match",
    [
        param({"aiperf_manifest_version": 1, "path": "t.jsonl"}, "root", RemoteErrorKind.MANIFEST_INVALID, "format", id="root-needs-format"),
        param({"aiperf_manifest_version": 1, "format": "mooncake_trace"}, "root", RemoteErrorKind.MANIFEST_INVALID, "path", id="root-needs-path"),
        param({**ROOT}, "sidecar", RemoteErrorKind.MANIFEST_INVALID, "must not set `path`", id="sidecar-forbids-path"),
        param({**ROOT, "path": "../x.jsonl"}, "root", RemoteErrorKind.MANIFEST_INVALID, "inside", id="path-escape"),
        param({**ROOT, "path": "/abs.jsonl"}, "root", RemoteErrorKind.MANIFEST_INVALID, "inside", id="path-absolute"),
        param({**ROOT, "path": "C:/Windows/x"}, "root", RemoteErrorKind.MANIFEST_INVALID, "inside", id="path-drive-letter"),
        param({**ROOT, "path": "C:x"}, "root", RemoteErrorKind.MANIFEST_INVALID, "inside", id="path-drive-relative"),
        param({**ROOT, "path": "\\\\server\\share\\x"}, "root", RemoteErrorKind.MANIFEST_INVALID, "inside", id="path-unc"),
        param({**ROOT, "path": "\\x"}, "root", RemoteErrorKind.MANIFEST_INVALID, "inside", id="path-backslash-rooted"),
        param({**ROOT, "aiperf_manifest_version": 2.0}, "root", RemoteErrorKind.MANIFEST_INVALID, "aiperf_manifest_version", id="version-float"),
        param({**ROOT, "aiperf_manifest_version": "2"}, "root", RemoteErrorKind.MANIFEST_INVALID, "aiperf_manifest_version", id="version-string"),
        param({**ROOT, "aiperf_manifest_version": True}, "root", RemoteErrorKind.MANIFEST_INVALID, "aiperf_manifest_version", id="version-bool"),
        param({**ROOT, "aiperf_manifest_version": 2}, "root", RemoteErrorKind.INCOMPATIBLE, "Upgrade AIPerf", id="future-version"),
        param({**ROOT, "sha256": "xyz"}, "root", RemoteErrorKind.MANIFEST_INVALID, "sha256", id="bad-sha"),
    ],
)  # fmt: skip
def test_manifest_rejections(data, kind, error_kind, match) -> None:
    with pytest.raises(RemoteSourceError, match=match) as exc:
        parse_manifest(_raw(data), ref_display=REF, kind=kind)
    assert exc.value.kind == error_kind


def test_not_json_is_manifest_invalid() -> None:
    with pytest.raises(RemoteSourceError) as exc:
        parse_manifest(b"{nope", ref_display=REF, kind="root")
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


def test_unknown_fields_warn_not_fail(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        parse_manifest(_raw({**ROOT, "future_field": 1}), ref_display=REF, kind="root")
    assert "future_field" in caplog.text


def _manifest(**requires: object):
    return parse_manifest(
        _raw({**ROOT, "requires": requires}), ref_display=REF, kind="root"
    )


def _check(manifest, **overrides: object) -> None:
    kwargs = {
        "ref_display": REF,
        "endpoint_type": "chat",
        "max_context_length": None,
        "aiperf_version": "0.13.0",
    }
    check_compatibility(manifest, **{**kwargs, **overrides})


def test_unregistered_format_is_incompatible() -> None:
    m = parse_manifest(
        _raw({**ROOT, "format": "no_such_loader"}), ref_display=REF, kind="root"
    )
    with pytest.raises(RemoteSourceError, match="no_such_loader") as exc:
        _check(m)
    assert exc.value.kind == RemoteErrorKind.INCOMPATIBLE


def test_aiperf_version_requirement() -> None:
    _check(_manifest(aiperf=">=0.13"))
    with pytest.raises(RemoteSourceError, match="requires AIPerf >=99"):
        _check(_manifest(aiperf=">=99"))


def test_unknown_installed_version_skips_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _check(_manifest(aiperf=">=99"), aiperf_version="unknown")
    assert "requires.aiperf" in caplog.text


def test_endpoint_type_requirement() -> None:
    _check(_manifest(endpoint_types=["chat"]))
    with pytest.raises(RemoteSourceError, match="--endpoint-type"):
        _check(_manifest(endpoint_types=["completions"]))


def test_context_length_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _check(_manifest(max_context_tokens=262144), max_context_length=131072)
    assert "262144" in caplog.text
