# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote dataset references (``hf://``, ``s3://``, ``https://``) for ``FileDataset.path``."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from pydantic import ConfigDict, Field, model_serializer, model_validator

from aiperf.common.redact import has_sensitive_query_parameters, redact_url
from aiperf.config.base import BaseConfig

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")
SUPPORTED_SCHEMES: tuple[str, ...] = ("hf", "s3", "https", "http")
_HF_OTHER_REPO_TYPES = ("models/", "spaces/")


@dataclass(frozen=True, slots=True)
class ParsedRemoteRef:
    """Scheme-specific decomposition of a remote reference."""

    scheme: str
    """Lowercase scheme: hf, s3, https, or http."""
    location: str
    """HF repo id, S3 bucket, or the full URL for http(s)."""
    path: str | None
    """Path inside the repo or bucket; None for a repo/bucket root or an http(s) URL."""
    revision: str | None
    """HF revision or S3 versionId as written by the user."""
    is_container: bool
    """True when the syntax alone says this names a repo or prefix rather than one object."""


def is_remote_ref(value: object) -> bool:
    """Return True for strings of the form ``<scheme>://...``."""
    return isinstance(value, str) and _SCHEME_RE.match(value) is not None


def is_presigned(value: str) -> bool:
    """Return True when the URL carries a credential in its query string."""
    return has_sensitive_query_parameters(value)


def parse_remote_ref(value: str) -> ParsedRemoteRef:
    """Parse a remote reference; raise ``ValueError`` with an actionable message.

    Every message interpolates the redacted form of ``value``, never the raw one.
    """
    match = _SCHEME_RE.match(value)
    if match is None:
        raise ValueError(f"'{redact_url(value)}' is not a remote reference")
    scheme = match.group(1).lower()
    rest = value[match.end() :]
    if scheme == "hf":
        return _parse_hf(value, rest)
    if scheme == "s3":
        return _parse_s3(value, rest)
    if scheme in ("http", "https"):
        return _parse_http(value, scheme)
    supported = ", ".join(f"{s}://" for s in SUPPORTED_SCHEMES)
    raise ValueError(
        f"Unsupported dataset reference scheme '{scheme}://' in "
        f"'{redact_url(value)}'. Supported schemes: {supported}."
    )


def _parse_hf(value: str, rest: str) -> ParsedRemoteRef:
    if rest.startswith("datasets/"):
        rest = rest[len("datasets/") :]
    elif rest.startswith(_HF_OTHER_REPO_TYPES):
        raise ValueError(
            f"'{redact_url(value)}': only Hugging Face dataset repos are supported "
            "(hf://datasets/<org>/<repo> or hf://<org>/<repo>)"
        )
    segments = [s for s in rest.split("/") if s]
    if len(segments) < 2:
        raise ValueError(
            f"'{redact_url(value)}': a Hugging Face reference must name <org>/<repo>"
        )
    org, repo_part = segments[0], segments[1]
    repo, sep, revision = repo_part.partition("@")
    if not repo or (sep and not revision):
        raise ValueError(f"'{redact_url(value)}': empty repo name or revision")
    path = "/".join(segments[2:]) or None
    return ParsedRemoteRef(
        scheme="hf",
        location=f"{org}/{repo}",
        path=path,
        revision=unquote(revision) if revision else None,
        is_container=path is None,
    )


def _parse_s3(value: str, rest: str) -> ParsedRemoteRef:
    bucket, _, key_and_query = rest.partition("/")
    if not bucket:
        raise ValueError(f"'{redact_url(value)}': an S3 reference must name a bucket")
    # Keys are literal (no percent-decoding), matching the AWS CLI; '?' starts the query.
    key, _, query = key_and_query.partition("?")
    version: str | None = None
    if query:
        params = dict(parse_qsl(query, keep_blank_values=True))
        unknown = sorted(set(params) - {"versionId"})
        if unknown:
            raise ValueError(
                f"'{redact_url(value)}': unsupported S3 query parameter(s) "
                f"{', '.join(unknown)}; only versionId is allowed"
            )
        version = params.get("versionId") or None
    return ParsedRemoteRef(
        scheme="s3",
        location=bucket,
        path=key or None,
        revision=version,
        is_container=(not key) or key.endswith("/"),
    )


def _parse_http(value: str, scheme: str) -> ParsedRemoteRef:
    if not urlsplit(value).netloc:
        raise ValueError(f"'{redact_url(value)}': URL has no host")
    return ParsedRemoteRef(
        scheme=scheme, location=value, path=None, revision=None, is_container=False
    )


class RemoteRef(BaseConfig):
    """A dataset location on Hugging Face, S3, or an HTTP(S) server."""

    # Pydantic echoes the failing input in ValidationError text; the raw ref may
    # hold a presigned signature.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    ref: Annotated[
        str,
        Field(
            description="The reference as written by the user, e.g. `hf://org/repo` "
            "or `s3://bucket/key`. Held in full in memory; serialized redacted."
        ),
    ]
    resolved_version: Annotated[
        str | None,
        Field(
            default=None,
            description="Exact version pinned by the access preflight (HF commit, "
            "S3 version or ETag, HTTP ETag). None until resolved.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_ref(self) -> RemoteRef:
        parse_remote_ref(self.ref)
        return self

    @property
    def parsed(self) -> ParsedRemoteRef:
        return parse_remote_ref(self.ref)

    @property
    def scheme(self) -> str:
        return self.parsed.scheme

    @property
    def display(self) -> str:
        """The reference with credentials redacted; safe for logs and artifacts."""
        return redact_url(self.ref)

    @property
    def canonical(self) -> str:
        """Normalized identity used for dedupe and cache keys (no credentials)."""
        p = self.parsed
        if p.scheme == "hf":
            rev = f"@{p.revision}" if p.revision else ""
            sub = f"/{p.path}" if p.path else ""
            return f"hf://datasets/{p.location}{rev}{sub}"
        if p.scheme == "s3":
            ver = f"?versionId={p.revision}" if p.revision else ""
            return f"s3://{p.location}/{p.path or ''}{ver}"
        parts = urlsplit(self.ref)
        if is_presigned(self.ref):
            # Each signing changes the query; the object identity is scheme+host+path.
            return urlunsplit(parts._replace(query="", fragment=""))
        return urlunsplit(parts._replace(fragment=""))

    def with_version(self, version: str) -> RemoteRef:
        return self.model_copy(update={"resolved_version": version})

    # ``ref`` is held raw in memory, so the default pydantic repr/str would leak
    # a presigned signature into logs, tracebacks and ValidationError messages.
    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        yield "ref", self.display

    def __str__(self) -> str:
        return self.display

    @model_serializer(mode="plain")
    def _serialize(self) -> str | dict[str, str]:
        if self.resolved_version is None:
            return self.display
        return {"ref": self.display, "resolved_version": self.resolved_version}


def coerce_remote_ref(value: Any) -> Any:
    """Turn remote strings or serialized dicts into ``RemoteRef``; pass anything else through."""
    if isinstance(value, RemoteRef):
        return value
    if is_remote_ref(value):
        return RemoteRef(ref=value)
    if isinstance(value, dict) and "ref" in value:
        return RemoteRef.model_validate(value)
    return value
