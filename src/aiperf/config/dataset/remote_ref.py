# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote dataset references (``hf://``, ``s3://``, ``https://``) for ``FileDataset.path``."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

from pydantic import (
    ConfigDict,
    Field,
    SerializationInfo,
    model_serializer,
    model_validator,
)

from aiperf.common.redact import (
    has_sensitive_query_parameters,
    is_sensitive_query_parameter_name,
    redact_url,
)
from aiperf.config.base import BaseConfig

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")
SUPPORTED_SCHEMES: tuple[str, ...] = ("hf", "s3", "https", "http")
_HF_OTHER_REPO_TYPES = ("models/", "spaces/")
_S3_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")

# Presigned-URL parameters that change on every signing without naming a
# different object. Credential-bearing names are matched separately by
# ``is_sensitive_query_parameter_name``.
_SIGNING_ENVELOPE_PREFIXES = ("x-amz-", "x-goog-")
_SIGNING_ENVELOPE_NAMES = frozenset(
    {
        "expires",
        "se",
        "st",
        "sp",
        "sv",
        "sr",
        "spr",
        "sig",
        "skoid",
        "sktid",
        "skt",
        "ske",
        "sks",
        "skv",
    }
)


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
    """HF revision (percent-decoded) or S3 versionId."""
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


def _without_fragment(value: str) -> str:
    """Redacted ``value`` with any fragment dropped; redaction does not cover fragments."""
    return redact_url(value.partition("#")[0])


def _parse_hf(value: str, rest: str) -> ParsedRemoteRef:
    if rest.startswith("datasets/"):
        rest = rest[len("datasets/") :]
    elif rest.startswith(_HF_OTHER_REPO_TYPES):
        raise ValueError(
            f"'{redact_url(value)}': only Hugging Face dataset repos are supported "
            "(hf://datasets/<org>/<repo> or hf://<org>/<repo>)"
        )
    if "?" in rest or "#" in rest:
        raise ValueError(
            f"'{_without_fragment(value)}': a Hugging Face reference cannot "
            "carry a query string or fragment"
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
    if "#" in rest:
        raise ValueError(
            f"'{_without_fragment(value)}': an S3 reference cannot carry a fragment"
        )
    # The query ends the reference, so split it off before looking for the key.
    head, _, query = rest.partition("?")
    bucket, _, key = head.partition("/")
    if not bucket:
        raise ValueError(f"'{redact_url(value)}': an S3 reference must name a bucket")
    if not _S3_BUCKET_RE.fullmatch(bucket) or ".." in bucket:
        raise ValueError(
            f"'{redact_url(value)}': not a valid S3 bucket name (3-63 lowercase "
            "letters, digits, '.' or '-')"
        )
    # Keys are literal (no percent-decoding), matching the AWS CLI.
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


def _is_signing_parameter(pair: str) -> bool:
    """Return True for a raw ``name=value`` query pair that varies per signing."""
    name = pair.partition("=")[0]
    if is_sensitive_query_parameter_name(name):
        return True
    lowered = unquote(name).lower()
    return lowered in _SIGNING_ENVELOPE_NAMES or lowered.startswith(
        _SIGNING_ENVELOPE_PREFIXES
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
            # Re-encode so a '/' inside the revision cannot merge with the path.
            rev = f"@{quote(p.revision, safe='')}" if p.revision else ""
            sub = f"/{p.path}" if p.path else ""
            return f"hf://datasets/{p.location}{rev}{sub}"
        if p.scheme == "s3":
            ver = f"?versionId={quote(p.revision, safe='')}" if p.revision else ""
            return f"s3://{p.location}/{p.path or ''}{ver}"
        parts = urlsplit(self.ref)
        query = parts.query
        if is_presigned(self.ref):
            # Each signing changes the credential and envelope parameters; the
            # remaining ones can still select a different object.
            query = "&".join(
                pair for pair in query.split("&") if not _is_signing_parameter(pair)
            )
        # Userinfo is a credential and is redacted on serialization, so keeping it
        # would give a ref read back from run_config.json a different cache key.
        netloc = parts.netloc.rpartition("@")[2]
        return urlunsplit(parts._replace(netloc=netloc, query=query, fragment=""))

    def with_version(self, version: str) -> RemoteRef:
        return self.model_copy(update={"resolved_version": version})

    # ``ref`` is held raw in memory, so the default pydantic repr/str would leak
    # a presigned signature into logs, tracebacks and ValidationError messages.
    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        yield "ref", self.display

    def __str__(self) -> str:
        return self.display

    @model_serializer(mode="plain")
    def _serialize(self, info: SerializationInfo) -> str | dict[str, str]:
        """Redacted by default; raw when the caller asks for secrets.

        Sweep expansion rebuilds each run from an ``include_secrets`` dump, which
        must keep a presigned signature or the expanded runs could not fetch.
        """
        context = info.context
        include_secrets = isinstance(context, dict) and context.get("include_secrets")
        ref = self.ref if include_secrets else self.display
        if self.resolved_version is None:
            return ref
        return {"ref": ref, "resolved_version": self.resolved_version}


def coerce_remote_ref(value: Any) -> Any:
    """Turn remote strings or serialized dicts into ``RemoteRef``; pass anything else through."""
    if isinstance(value, RemoteRef):
        return value
    if is_remote_ref(value):
        return RemoteRef(ref=value)
    if isinstance(value, dict) and "ref" in value:
        return RemoteRef.model_validate(value)
    return value


class RemoteFixedScheduleCheck(BaseConfig):
    """Fixed-schedule auto-promotion deferred until a remote trace's first record is readable.

    Set by the CLI converter for a remote --input-file; consumed and cleared by the
    dataset access preflight.
    """

    model_config = ConfigDict(extra="forbid")

    promoted: Annotated[
        bool,
        Field(
            description="True when the CLI built the profiling phase as fixed_schedule ahead "
            "of the timestamp check; the access preflight demotes it if the trace has none."
        ),
    ]
    conflict_error: Annotated[
        str | None,
        Field(
            default=None,
            description="Error raised if the trace turns out to have timestamps (an "
            "incompatible rate-shaped mode or a parameter sweep was selected).",
        ),
    ]
